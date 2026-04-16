"""
strategies/custom_strat.py — Multi-Timeframe Regime-Subordinated Strategy
==========================================================================
Implements the full LONGS / SHORTS heuristic across D1, H4, H1, M5,
with Fibonacci confluence, Donchian triggers, and deterministic SL/TP.
"""

import logging
from typing import Optional

import numpy as np
import pandas as pd

from strategies.base_strategy import (
    BaseStrategy,
    HyperoptSpace,
    Signal,
    donchian_high,
    donchian_low,
    ema,
    fibonacci_levels,
)

logger = logging.getLogger(__name__)


class RegimeSubordinatedStrategy(BaseStrategy):
    """
    Dynamic Strategy Subordination with Multi-Timeframe Confluence.

    LONGS (Bull Trend only):
      1. D1: Close > EMA(200)
      2. H4: Close > EMA(50)
      3. H1: Pullback into Fibo 0.50–0.618 zone of 24-bar impulse,
             EMA(50) within that band, Close <= EMA(50)
      4. M5: Close > Donchian High(6) → BUY trigger

    SHORTS (Bear Trend only):
      Inverted mirror of the LONGS logic.

    Risk:
      SL = lowest low (12 M5 bars) for longs / highest high for shorts
      TP = 0.0 Fibo level of the H1 impulse
      Position size = exactly 1% account risk
    """

    name = "RegimeSubordinated_MTF"
    base_timeframe = "M5"

    def default_params(self) -> dict:
        return {
            # Structural
            "d1_ema_period": 200,
            # Trend
            "h4_ema_period": 50,
            # Confluence
            "h1_ema_period": 50,
            "h1_impulse_bars": 24,
            "fibo_upper": 0.618,
            "fibo_lower": 0.50,
            # Trigger
            "m5_donchian_period": 6,
            # Risk
            "m5_sl_lookback": 12,
            "risk_pct": 0.01,
            # Cooldown
            "cooldown_hours": 48,
        }

    # ── LONGS ────────────────────────────────
    def evaluate_long(
        self,
        data: dict[str, pd.DataFrame],
        bar_index: int = -1,
    ) -> Optional[Signal]:
        p = self.params
        confluence_info = {}

        # --- 1. Structural Filter (D1) ---
        d1 = data.get("D1")
        if d1 is None or len(d1) < p["d1_ema_period"] + 1:
            return None

        d1_ema = ema(d1["Close"], p["d1_ema_period"])
        d1_close = d1["Close"].iloc[bar_index]
        d1_ema_val = d1_ema.iloc[bar_index]
        if d1_close <= d1_ema_val:
            return None
        confluence_info["d1_close"] = d1_close
        confluence_info["d1_ema200"] = d1_ema_val

        # --- 2. Trend Filter (H4) ---
        h4 = data.get("H4")
        if h4 is None or len(h4) < p["h4_ema_period"] + 1:
            return None

        h4_ema = ema(h4["Close"], p["h4_ema_period"])
        h4_close = h4["Close"].iloc[bar_index]
        h4_ema_val = h4_ema.iloc[bar_index]
        if h4_close <= h4_ema_val:
            return None
        confluence_info["h4_close"] = h4_close
        confluence_info["h4_ema50"] = h4_ema_val

        # --- 3. Confluence (H1) ---
        h1 = data.get("H1")
        if h1 is None or len(h1) < p["h1_impulse_bars"] + p["h1_ema_period"] + 1:
            return None

        # Impulse: High/Low of last N H1 candles
        impulse_slice = h1.iloc[bar_index - p["h1_impulse_bars"]:bar_index] if bar_index != -1 else h1.iloc[-p["h1_impulse_bars"] - 1:-1]
        if len(impulse_slice) < p["h1_impulse_bars"]:
            return None

        impulse_high = impulse_slice["High"].max()
        impulse_low = impulse_slice["Low"].min()

        # Fibonacci levels of the up-impulse
        fib = fibonacci_levels(impulse_high, impulse_low, direction="up")
        fib_upper = fib["0.50"] if p["fibo_lower"] == 0.50 else impulse_high - p["fibo_lower"] * (impulse_high - impulse_low)
        fib_lower = fib["0.618"] if p["fibo_upper"] == 0.618 else impulse_high - p["fibo_upper"] * (impulse_high - impulse_low)
        # Ensure fib_lower < fib_upper (both are retracement INTO the move)
        fib_band_top = max(fib_upper, fib_lower)
        fib_band_bot = min(fib_upper, fib_lower)

        h1_ema = ema(h1["Close"], p["h1_ema_period"])
        h1_ema_val = h1_ema.iloc[bar_index]
        h1_close = h1["Close"].iloc[bar_index]

        # EMA must be within the Fibo band
        if not (fib_band_bot <= h1_ema_val <= fib_band_top):
            return None

        # Price must pull back into zone (close <= EMA)
        if h1_close > h1_ema_val:
            return None

        confluence_info["h1_impulse_high"] = impulse_high
        confluence_info["h1_impulse_low"] = impulse_low
        confluence_info["h1_fib_band"] = (fib_band_bot, fib_band_top)
        confluence_info["h1_ema50"] = h1_ema_val
        confluence_info["h1_close"] = h1_close

        # --- 4. Trigger (M5) ---
        m5 = data.get("M5")
        if m5 is None or len(m5) < max(p["m5_donchian_period"], p["m5_sl_lookback"]) + 2:
            return None

        donch_hi = donchian_high(m5["High"], p["m5_donchian_period"])
        m5_close = m5["Close"].iloc[bar_index]
        prev_donch = donch_hi.iloc[bar_index - 1] if bar_index != -1 else donch_hi.iloc[-2]

        if m5_close <= prev_donch:
            return None

        # --- Risk Management ---
        entry_price = m5_close
        # SL = lowest low of last 12 M5 candles
        sl_slice = m5.iloc[bar_index - p["m5_sl_lookback"]:bar_index] if bar_index != -1 else m5.iloc[-p["m5_sl_lookback"] - 1:-1]
        stop_loss = sl_slice["Low"].min()

        # TP = the 0.0 Fibo level (top of the impulse)
        take_profit = impulse_high  # fib["0.0"]

        if stop_loss >= entry_price:
            return None  # invalid SL

        confluence_info["m5_donch_high"] = prev_donch
        confluence_info["m5_close"] = m5_close

        return Signal(
            direction="BUY",
            entry_price=entry_price,
            stop_loss=stop_loss,
            take_profit=take_profit,
            symbol=data.get("_symbol", ""),
            confluence=confluence_info,
            timestamp=m5.index[bar_index] if bar_index != -1 else m5.index[-1],
        )

    # ── SHORTS (Inverted Logic) ──────────────
    def evaluate_short(
        self,
        data: dict[str, pd.DataFrame],
        bar_index: int = -1,
    ) -> Optional[Signal]:
        p = self.params
        confluence_info = {}

        # --- 1. Structural Filter (D1): Close < EMA(200) ---
        d1 = data.get("D1")
        if d1 is None or len(d1) < p["d1_ema_period"] + 1:
            return None

        d1_ema = ema(d1["Close"], p["d1_ema_period"])
        d1_close = d1["Close"].iloc[bar_index]
        d1_ema_val = d1_ema.iloc[bar_index]
        if d1_close >= d1_ema_val:
            return None
        confluence_info["d1_close"] = d1_close
        confluence_info["d1_ema200"] = d1_ema_val

        # --- 2. Trend Filter (H4): Close < EMA(50) ---
        h4 = data.get("H4")
        if h4 is None or len(h4) < p["h4_ema_period"] + 1:
            return None

        h4_ema = ema(h4["Close"], p["h4_ema_period"])
        h4_close = h4["Close"].iloc[bar_index]
        h4_ema_val = h4_ema.iloc[bar_index]
        if h4_close >= h4_ema_val:
            return None
        confluence_info["h4_close"] = h4_close
        confluence_info["h4_ema50"] = h4_ema_val

        # --- 3. Confluence (H1): Downward impulse, rally into Fibo ---
        h1 = data.get("H1")
        if h1 is None or len(h1) < p["h1_impulse_bars"] + p["h1_ema_period"] + 1:
            return None

        impulse_slice = h1.iloc[bar_index - p["h1_impulse_bars"]:bar_index] if bar_index != -1 else h1.iloc[-p["h1_impulse_bars"] - 1:-1]
        if len(impulse_slice) < p["h1_impulse_bars"]:
            return None

        impulse_high = impulse_slice["High"].max()
        impulse_low = impulse_slice["Low"].min()

        # Fibonacci retracement of a DOWN move
        fib = fibonacci_levels(impulse_high, impulse_low, direction="down")
        fib_upper_val = impulse_low + p["fibo_upper"] * (impulse_high - impulse_low)
        fib_lower_val = impulse_low + p["fibo_lower"] * (impulse_high - impulse_low)
        fib_band_top = max(fib_upper_val, fib_lower_val)
        fib_band_bot = min(fib_upper_val, fib_lower_val)

        h1_ema = ema(h1["Close"], p["h1_ema_period"])
        h1_ema_val = h1_ema.iloc[bar_index]
        h1_close = h1["Close"].iloc[bar_index]

        # EMA must be within the Fibo band
        if not (fib_band_bot <= h1_ema_val <= fib_band_top):
            return None

        # Price must rally into zone (close >= EMA)
        if h1_close < h1_ema_val:
            return None

        confluence_info["h1_impulse_high"] = impulse_high
        confluence_info["h1_impulse_low"] = impulse_low
        confluence_info["h1_fib_band"] = (fib_band_bot, fib_band_top)
        confluence_info["h1_ema50"] = h1_ema_val
        confluence_info["h1_close"] = h1_close

        # --- 4. Trigger (M5): Close < Donchian Low(6) ---
        m5 = data.get("M5")
        if m5 is None or len(m5) < max(p["m5_donchian_period"], p["m5_sl_lookback"]) + 2:
            return None

        donch_lo = donchian_low(m5["Low"], p["m5_donchian_period"])
        m5_close = m5["Close"].iloc[bar_index]
        prev_donch = donch_lo.iloc[bar_index - 1] if bar_index != -1 else donch_lo.iloc[-2]

        if m5_close >= prev_donch:
            return None

        # --- Risk Management ---
        entry_price = m5_close
        # SL = highest high of last 12 M5 candles
        sl_slice = m5.iloc[bar_index - p["m5_sl_lookback"]:bar_index] if bar_index != -1 else m5.iloc[-p["m5_sl_lookback"] - 1:-1]
        stop_loss = sl_slice["High"].max()

        # TP = the 0.0 Fibo level (bottom of the impulse)
        take_profit = impulse_low  # fib["0.0"] for down move

        if stop_loss <= entry_price:
            return None  # invalid SL

        confluence_info["m5_donch_low"] = prev_donch
        confluence_info["m5_close"] = m5_close

        return Signal(
            direction="SELL",
            entry_price=entry_price,
            stop_loss=stop_loss,
            take_profit=take_profit,
            symbol=data.get("_symbol", ""),
            confluence=confluence_info,
            timestamp=m5.index[bar_index] if bar_index != -1 else m5.index[-1],
        )

    # ── Hyperopt Space ───────────────────────
    def hyperopt_space(self) -> list[HyperoptSpace]:
        """
        Define optimization boundaries.
        Allows loosening structural parameters if trade count is too low.
        """
        return [
            HyperoptSpace("d1_ema_period", low=100, high=300, step=10, param_type="int"),
            HyperoptSpace("h4_ema_period", low=20, high=100, step=5, param_type="int"),
            HyperoptSpace("h1_ema_period", low=20, high=100, step=5, param_type="int"),
            HyperoptSpace("h1_impulse_bars", low=12, high=48, step=2, param_type="int"),
            HyperoptSpace("fibo_upper", low=0.382, high=0.786, step=0.01, param_type="float"),
            HyperoptSpace("fibo_lower", low=0.382, high=0.786, step=0.01, param_type="float"),
            HyperoptSpace("m5_donchian_period", low=4, high=12, step=1, param_type="int"),
            HyperoptSpace("m5_sl_lookback", low=6, high=24, step=1, param_type="int"),
        ]

    # ── Vectorized signal generation for backtesting ──
    def generate_signals_vectorized(
        self,
        data: dict[str, pd.DataFrame],
        regime_series: pd.Series,
    ) -> pd.DataFrame:
        """
        Generate entry/exit signals across the entire M5 dataset for vectorbt.
        Returns a DataFrame aligned to M5 index with columns:
          entries_long, exits_long, entries_short, exits_short, sl, tp
        """
        p = self.params
        base_tf = self.base_timeframe
        m5 = data[base_tf].copy()
        h1 = data["H1"].copy()
        h4 = data["H4"].copy()
        d1 = data["D1"].copy()

        n = len(m5)
        entries_long = pd.Series(False, index=m5.index)
        entries_short = pd.Series(False, index=m5.index)
        sl_prices = pd.Series(np.nan, index=m5.index)
        tp_prices = pd.Series(np.nan, index=m5.index)

        # Precompute indicators
        d1["ema_struct"] = ema(d1["Close"], p["d1_ema_period"])
        h4["ema_trend"] = ema(h4["Close"], p["h4_ema_period"])
        h1["ema_conf"] = ema(h1["Close"], p["h1_ema_period"])
        m5["donch_hi"] = donchian_high(m5["High"], p["m5_donchian_period"]).shift(1)
        m5["donch_lo"] = donchian_low(m5["Low"], p["m5_donchian_period"]).shift(1)
        m5["sl_low"] = m5["Low"].rolling(p["m5_sl_lookback"]).min().shift(1)
        m5["sl_high"] = m5["High"].rolling(p["m5_sl_lookback"]).max().shift(1)

        # Rolling H1 impulse
        h1["impulse_high"] = h1["High"].rolling(p["h1_impulse_bars"]).max().shift(1)
        h1["impulse_low"] = h1["Low"].rolling(p["h1_impulse_bars"]).min().shift(1)

        # Align higher TF to M5 using forward-fill (as-of join)
        m5["regime"] = regime_series.reindex(m5.index, method="ffill")

        # Forward-fill higher TFs to M5 index
        for col in ["ema_struct"]:
            m5[f"d1_{col}"] = d1[col].reindex(m5.index, method="ffill")
        m5["d1_close"] = d1["Close"].reindex(m5.index, method="ffill")

        for col in ["ema_trend"]:
            m5[f"h4_{col}"] = h4[col].reindex(m5.index, method="ffill")
        m5["h4_close"] = h4["Close"].reindex(m5.index, method="ffill")

        for col in ["ema_conf", "impulse_high", "impulse_low"]:
            m5[f"h1_{col}"] = h1[col].reindex(m5.index, method="ffill")
        m5["h1_close"] = h1["Close"].reindex(m5.index, method="ffill")

        # ── Vectorized LONG conditions ──
        bull_regime = m5["regime"] == "Bull Trend"

        d1_long = m5["d1_close"] > m5["d1_ema_struct"]
        h4_long = m5["h4_close"] > m5["h4_ema_trend"]

        # H1 confluence
        fib_range = m5["h1_impulse_high"] - m5["h1_impulse_low"]
        fib_band_top_long = m5["h1_impulse_high"] - p["fibo_lower"] * fib_range
        fib_band_bot_long = m5["h1_impulse_high"] - p["fibo_upper"] * fib_range
        h1_ema_in_band_long = (m5["h1_ema_conf"] >= fib_band_bot_long) & (m5["h1_ema_conf"] <= fib_band_top_long)
        h1_pullback_long = m5["h1_close"] <= m5["h1_ema_conf"]

        m5_trigger_long = m5["Close"] > m5["donch_hi"]
        valid_sl_long = m5["sl_low"] < m5["Close"]

        entries_long = (
            bull_regime & d1_long & h4_long &
            h1_ema_in_band_long & h1_pullback_long &
            m5_trigger_long & valid_sl_long
        )

        # ── Vectorized SHORT conditions ──
        bear_regime = m5["regime"] == "Bear Trend"

        d1_short = m5["d1_close"] < m5["d1_ema_struct"]
        h4_short = m5["h4_close"] < m5["h4_ema_trend"]

        # H1 confluence (inverted)
        fib_band_top_short = m5["h1_impulse_low"] + p["fibo_upper"] * fib_range
        fib_band_bot_short = m5["h1_impulse_low"] + p["fibo_lower"] * fib_range
        h1_ema_in_band_short = (m5["h1_ema_conf"] >= fib_band_bot_short) & (m5["h1_ema_conf"] <= fib_band_top_short)
        h1_rally_short = m5["h1_close"] >= m5["h1_ema_conf"]

        m5_trigger_short = m5["Close"] < m5["donch_lo"]
        valid_sl_short = m5["sl_high"] > m5["Close"]

        entries_short = (
            bear_regime & d1_short & h4_short &
            h1_ema_in_band_short & h1_rally_short &
            m5_trigger_short & valid_sl_short
        )

        # Build result
        signals = pd.DataFrame(index=m5.index)
        signals["entries_long"] = entries_long.fillna(False)
        signals["entries_short"] = entries_short.fillna(False)
        signals["sl_long"] = np.where(entries_long, m5["sl_low"], np.nan)
        signals["tp_long"] = np.where(entries_long, m5["h1_impulse_high"], np.nan)
        signals["sl_short"] = np.where(entries_short, m5["sl_high"], np.nan)
        signals["tp_short"] = np.where(entries_short, m5["h1_impulse_low"], np.nan)
        signals["regime"] = m5["regime"]

        return signals
