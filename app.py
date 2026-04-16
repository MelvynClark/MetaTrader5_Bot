"""
app.py — Streamlit Dashboard (Control Center)
===============================================
Interactive UI for:
  - MT5 credential input
  - Strategy / Pair / Leverage selection
  - Start / Stop live trading daemon
  - Live display: HMM regime, signal state, PnL
"""

import json
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

CONFIGS_DIR = Path(__file__).parent / "configs"
CONFIGS_DIR.mkdir(exist_ok=True)
LIVE_CONFIG = CONFIGS_DIR / "live_config.json"
STATE_FILE = CONFIGS_DIR / "bot_state.json"
PID_FILE = CONFIGS_DIR / "bot_pid.txt"

PAIRS = [
    "EURUSD", "GBPUSD", "USDJPY", "GBPJPY", "EURJPY",
    "AUDUSD", "USDCAD", "USDCHF", "NZDUSD", "EURGBP",
]
STRATEGIES = ["RegimeSubordinated_MTF"]

# ── Page Config ──────────────────────────────
st.set_page_config(
    page_title="Regime Trading Bot",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)


def main():
    st.title("Regime-Based Forex Trading System")
    st.caption("HMM Regime Detection | Multi-Timeframe Subordination | MT5 Execution")

    # ── Sidebar: Configuration ───────────────
    with st.sidebar:
        st.header("MT5 Connection")
        mt5_login = st.text_input("MT5 Login (Account #)", type="default")
        mt5_password = st.text_input("MT5 Password", type="password")
        mt5_server = st.text_input("MT5 Server", value="ICMarketsSC-Demo")
        mt5_path = st.text_input("MT5 Terminal Path (optional)", value="")

        st.divider()
        st.header("Trading Config")
        symbol = st.selectbox("Pair", PAIRS, index=0)
        strategy_name = st.selectbox("Strategy", STRATEGIES, index=0)
        leverage = st.slider("Leverage", min_value=1, max_value=100, value=1, step=1)
        risk_pct = st.slider("Risk per Trade (%)", min_value=0.5, max_value=5.0, value=1.0, step=0.5)

        st.divider()
        params_file = st.text_input(
            "Optimized Params File",
            value="optimized_params.json",
            help="JSON from Hyperopt. Stored in configs/",
        )

    # ── Main Area ────────────────────────────
    col1, col2, col3 = st.columns(3)

    with col1:
        start_btn = st.button("Start Real Trading", type="primary", use_container_width=True)
    with col2:
        stop_btn = st.button("Stop Bot", type="secondary", use_container_width=True)
    with col3:
        refresh_btn = st.button("Refresh Status", use_container_width=True)

    # ── Start Trading ────────────────────────
    if start_btn:
        if not mt5_login or not mt5_password:
            st.error("Please enter MT5 credentials")
        else:
            config = {
                "mt5_login": mt5_login,
                "mt5_password": mt5_password,
                "mt5_server": mt5_server,
                "mt5_path": mt5_path if mt5_path else None,
                "symbol": symbol,
                "strategy": strategy_name,
                "leverage": leverage,
                "risk_pct": risk_pct / 100,
                "params_file": params_file,
                "stop": False,
                "started_at": datetime.utcnow().isoformat(),
            }
            with open(LIVE_CONFIG, "w") as f:
                json.dump(config, f, indent=2)

            # Launch live_bot.py as detached subprocess
            bot_script = Path(__file__).parent / "live_bot.py"
            proc = subprocess.Popen(
                [sys.executable, str(bot_script)],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                creationflags=subprocess.CREATE_NO_WINDOW if sys.platform == "win32" else 0,
            )
            PID_FILE.write_text(str(proc.pid))
            st.success(f"Bot launched (PID: {proc.pid}) — trading {symbol}")

    # ── Stop Bot ─────────────────────────────
    if stop_btn:
        _stop_bot()
        st.warning("Stop signal sent to bot")

    # ── Status Display ───────────────────────
    st.divider()
    _display_status()


def _stop_bot():
    """Send stop signal via config and attempt to kill process."""
    if LIVE_CONFIG.exists():
        with open(LIVE_CONFIG) as f:
            cfg = json.load(f)
        cfg["stop"] = True
        with open(LIVE_CONFIG, "w") as f:
            json.dump(cfg, f, indent=2)

    if PID_FILE.exists():
        import os
        try:
            pid = int(PID_FILE.read_text().strip())
            os.kill(pid, 15)  # SIGTERM
        except (ProcessLookupError, ValueError, PermissionError):
            pass
        PID_FILE.unlink(missing_ok=True)


def _display_status():
    """Read bot state and display live dashboard."""
    if not STATE_FILE.exists():
        st.info("No bot state found. Start trading to see live data.")
        return

    try:
        with open(STATE_FILE) as f:
            state = json.load(f)
    except json.JSONDecodeError:
        st.warning("State file corrupted — waiting for next update")
        return

    # ── Key Metrics Row ──────────────────────
    c1, c2, c3, c4 = st.columns(4)

    regime = state.get("regime", "Unknown")
    regime_colors = {
        "Bull Trend": "🟢",
        "Bear Trend": "🔴",
        "Neutral": "⚪",
    }

    with c1:
        st.metric("Status", "RUNNING" if state.get("running") else "STOPPED")
    with c2:
        st.metric("HMM Regime", f"{regime_colors.get(regime, '❓')} {regime}")
    with c3:
        st.metric("Open Positions", state.get("open_positions", 0))
    with c4:
        cd = state.get("cooldown_active", False)
        st.metric("Cooldown", "ACTIVE (48h)" if cd else "Ready")

    # ── Symbol & Last Update ─────────────────
    st.caption(
        f"Symbol: **{state.get('symbol', 'N/A')}** | "
        f"Last Update: {state.get('last_update', 'N/A')}"
    )

    # ── Open Positions Detail ────────────────
    positions = state.get("positions", [])
    if positions:
        st.subheader("Open Positions")
        pos_df = pd.DataFrame(positions)
        display_cols = [
            c for c in ["ticket", "symbol", "type", "volume", "price_open",
                         "sl", "tp", "profit", "time"]
            if c in pos_df.columns
        ]
        if display_cols:
            pos_df["type"] = pos_df["type"].map({0: "BUY", 1: "SELL"})
            st.dataframe(pos_df[display_cols], use_container_width=True)

            # Live PnL gauge
            total_pnl = pos_df["profit"].sum() if "profit" in pos_df.columns else 0
            fig = go.Figure(go.Indicator(
                mode="number+delta",
                value=total_pnl,
                title={"text": "Live Unrealized PnL"},
                delta={"reference": 0, "valueformat": ".2f"},
                number={"prefix": "$", "valueformat": ".2f"},
            ))
            fig.update_layout(height=200, template="plotly_dark")
            st.plotly_chart(fig, use_container_width=True)

    # ── Recent Log Tail ──────────────────────
    log_path = Path(__file__).parent / "logs" / "live_bot.log"
    if log_path.exists():
        with st.expander("Recent Bot Logs (last 30 lines)"):
            lines = log_path.read_text().splitlines()
            st.code("\n".join(lines[-30:]), language="log")


if __name__ == "__main__":
    main()
