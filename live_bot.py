"""
live_bot.py — The Execution Daemon
====================================
Background process that:
  1. Reads active configuration saved by Streamlit (configs/live_config.json)
  2. Polls MT5 for live OHLCV data
  3. Applies trained HMM to rolling live data
  4. Evaluates multi-timeframe heuristics
  5. Routes real API calls to MT5 for 1% risk execution
  6. Enforces 48-hour cooldown and regime-flip emergency exits
"""

import json
import logging
import signal
import sys
import time
from datetime import datetime, timedelta
from pathlib import Path

import pandas as pd

from data_engine import MT5Connection
from hmm_engine import RegimeDetector
from strategies.base_strategy import CooldownTracker, Signal, calculate_lot_size
from strategies.custom_strat import RegimeSubordinatedStrategy

# ── Logging ──────────────────────────────────
LOG_DIR = Path(__file__).parent / "logs"
LOG_DIR.mkdir(exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[
        logging.FileHandler(LOG_DIR / "live_bot.log"),
        logging.StreamHandler(sys.stdout),
    ],
)
logger = logging.getLogger("LiveBot")

CONFIGS_DIR = Path(__file__).parent / "configs"
LIVE_CONFIG = CONFIGS_DIR / "live_config.json"
STATE_FILE = CONFIGS_DIR / "bot_state.json"


class LiveBot:
    """
    The live trading execution daemon.
    Designed to be launched via subprocess from Streamlit.
    """

    def __init__(self):
        self.running = False
        self.mt5: MT5Connection = None
        self.hmm = RegimeDetector()
        self.strategy = RegimeSubordinatedStrategy()
        self.cooldown = CooldownTracker()
        self.config = {}
        self.current_regime = "Unknown"
        self.poll_interval = 30  # seconds between M5 polls

    def load_config(self) -> dict:
        if not LIVE_CONFIG.exists():
            raise FileNotFoundError(f"No config at {LIVE_CONFIG}. Start from Streamlit first.")
        with open(LIVE_CONFIG) as f:
            self.config = json.load(f)
        logger.info("Config loaded: %s", self.config)

        # Load optimized params if available
        opt_file = self.config.get("params_file", "optimized_params.json")
        try:
            self.strategy.load_params(opt_file)
        except Exception:
            logger.info("Using default strategy params")

        return self.config

    def connect_mt5(self):
        self.mt5 = MT5Connection(
            login=int(self.config["mt5_login"]),
            password=self.config["mt5_password"],
            server=self.config.get("mt5_server", "ICMarketsSC-Demo"),
            path=self.config.get("mt5_path"),
        )
        if not self.mt5.connected:
            raise ConnectionError("Failed to connect to MT5")

    def train_hmm(self):
        """Train HMM on recent H1 data for the active symbol."""
        symbol = self.config["symbol"]
        logger.info("Training HMM on %s H1 data...", symbol)
        df = self.mt5.fetch_ohlcv(symbol, "H1", days_back=365)
        if df is None or len(df) < 200:
            raise RuntimeError(f"Insufficient H1 data for HMM training: {len(df) if df is not None else 0} bars")
        self.hmm.fit(df)
        self.hmm.save(f"hmm_live_{symbol}")
        logger.info("HMM trained and saved")

    def fetch_live_data(self) -> dict[str, pd.DataFrame]:
        """Fetch latest multi-TF data from MT5."""
        symbol = self.config["symbol"]
        data = {}
        tf_bars = {"D1": 300, "H4": 500, "H1": 500, "M5": 500}
        for tf, bars in tf_bars.items():
            df = self.mt5.fetch_ohlcv(symbol, tf, bars=bars)
            if df is not None and len(df) > 0:
                data[tf] = df
            else:
                logger.warning("No data for %s %s", symbol, tf)
        data["_symbol"] = symbol
        return data

    def get_current_regime(self, h1_data: pd.DataFrame) -> str:
        """Predict current regime from latest H1 data."""
        try:
            regime = self.hmm.predict_latest(h1_data)
            return regime
        except Exception as e:
            logger.error("Regime prediction failed: %s", e)
            return "Unknown"

    def check_regime_flip_exit(self):
        """Emergency close if regime flips to adverse direction."""
        symbol = self.config["symbol"]
        positions = self.mt5.get_positions(symbol=symbol)
        if not positions:
            return

        for pos in positions:
            should_close = False
            if pos["type"] == 0 and self.current_regime == "Bear Trend":  # BUY in Bear
                should_close = True
                logger.warning("REGIME FLIP: Closing LONG %s — regime is Bear", symbol)
            elif pos["type"] == 1 and self.current_regime == "Bull Trend":  # SELL in Bull
                should_close = True
                logger.warning("REGIME FLIP: Closing SHORT %s — regime is Bull", symbol)

            if should_close:
                result = self.mt5.close_position(
                    ticket=pos["ticket"],
                    symbol=symbol,
                    volume=pos["volume"],
                    position_type=pos["type"],
                )
                logger.info("Emergency close result: %s", result)
                self.cooldown.record_exit(pd.Timestamp.utcnow())

    def execute_signal(self, sig: Signal):
        """Route a validated signal to MT5."""
        symbol = self.config["symbol"]
        leverage = float(self.config.get("leverage", 1))

        # Get symbol info for lot sizing
        sym_info = self.mt5.symbol_info(symbol)
        if sym_info is None:
            logger.error("Cannot get symbol info for %s", symbol)
            return

        account = self.mt5.account_info()
        balance = account.get("balance", 0)
        tick_value = sym_info.get("trade_tick_value", 1.0)
        point = sym_info.get("point", 0.00001)
        min_lot = sym_info.get("volume_min", 0.01)
        max_lot = sym_info.get("volume_max", 100.0)
        lot_step = sym_info.get("volume_step", 0.01)

        lot_size = calculate_lot_size(
            balance=balance,
            risk_pct=self.strategy.params["risk_pct"],
            entry_price=sig.entry_price,
            stop_loss=sig.stop_loss,
            tick_value=tick_value,
            point=point,
            min_lot=min_lot,
            max_lot=max_lot,
            lot_step=lot_step,
        )

        logger.info(
            "EXECUTING: %s %s | Lots=%.2f | Entry=%.5f | SL=%.5f | TP=%.5f",
            sig.direction, symbol, lot_size, sig.entry_price, sig.stop_loss, sig.take_profit,
        )

        result = self.mt5.send_order(
            symbol=symbol,
            order_type=sig.direction,
            volume=lot_size,
            sl=sig.stop_loss,
            tp=sig.take_profit,
            comment=f"Regime_{self.current_regime[:4]}",
        )
        logger.info("Order result: %s", result)

    def save_state(self):
        """Persist bot state for Streamlit to read."""
        state = {
            "running": self.running,
            "regime": self.current_regime,
            "last_update": datetime.utcnow().isoformat(),
            "cooldown_active": self.cooldown.is_in_cooldown(pd.Timestamp.utcnow()),
            "symbol": self.config.get("symbol", ""),
        }

        positions = []
        if self.mt5 and self.mt5.connected:
            positions = self.mt5.get_positions(symbol=self.config.get("symbol"))
        state["open_positions"] = len(positions)
        state["positions"] = positions

        with open(STATE_FILE, "w") as f:
            json.dump(state, f, indent=2, default=str)

    def run(self):
        """Main loop."""
        self.load_config()
        self.connect_mt5()
        self.train_hmm()
        self.running = True

        # Graceful shutdown
        def _shutdown(signum, frame):
            logger.info("Shutdown signal received")
            self.running = False

        signal.signal(signal.SIGTERM, _shutdown)
        signal.signal(signal.SIGINT, _shutdown)

        logger.info("=" * 60)
        logger.info("LIVE BOT STARTED — %s", self.config["symbol"])
        logger.info("=" * 60)

        while self.running:
            try:
                # Refresh config (allows Streamlit to update settings)
                if LIVE_CONFIG.exists():
                    with open(LIVE_CONFIG) as f:
                        fresh = json.load(f)
                    if fresh.get("stop", False):
                        logger.info("Stop signal received from UI")
                        break

                # Fetch live data
                data = self.fetch_live_data()
                if "H1" not in data or "M5" not in data:
                    logger.warning("Insufficient live data, retrying...")
                    time.sleep(self.poll_interval)
                    continue

                # Determine regime
                self.current_regime = self.get_current_regime(data["H1"])
                logger.info("Current regime: %s", self.current_regime)

                # Check emergency exits
                self.check_regime_flip_exit()

                # Check cooldown
                now = pd.Timestamp.utcnow()
                if self.cooldown.is_in_cooldown(now):
                    logger.info("In cooldown — skipping signal evaluation")
                    self.save_state()
                    time.sleep(self.poll_interval)
                    continue

                # Check if already in a position
                positions = self.mt5.get_positions(symbol=self.config["symbol"])
                if positions:
                    logger.info("Position open (ticket %s) — monitoring", positions[0]["ticket"])
                    self.save_state()
                    time.sleep(self.poll_interval)
                    continue

                # Evaluate strategy (subordinated to regime)
                sig = self.strategy.evaluate(self.current_regime, data)
                if sig:
                    logger.info("SIGNAL: %s", sig)
                    self.execute_signal(sig)
                else:
                    logger.info("No signal — waiting")

                self.save_state()

            except Exception as e:
                logger.exception("Error in main loop: %s", e)

            time.sleep(self.poll_interval)

        # Cleanup
        self.running = False
        self.save_state()
        if self.mt5:
            self.mt5.disconnect()
        logger.info("Live bot stopped cleanly")


if __name__ == "__main__":
    bot = LiveBot()
    bot.run()
