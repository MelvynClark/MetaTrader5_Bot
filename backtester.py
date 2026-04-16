"""
backtester.py — Trade-by-Trade Backtesting Engine
===================================================
Correct Forex PnL math:
  PnL = (Exit - Entry) * LotSize * ContractSize * FxConversionRate
  Leverage is NOT in the PnL formula — it only determines margin.
  Cross-pair support via dynamic quote-to-USD conversion rates.
  Dynamic IC Markets Raw Spread commission by asset class.
Outputs: Freqtrade-style analytics (SQN, Expectancy, Profit Factor, etc.)
"""

import logging
from typing import Optional

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from data_engine import CONTRACT_SIZE

logger = logging.getLogger(__name__)


# ──────────────────────────────────────────────
# IC Markets Raw Spread Commission
# ──────────────────────────────────────────────
_METALS = {"XAUUSD", "XAGUSD", "XAUEUR", "XAGEUR"}
_ZERO_COMMISSION = {"US500", "US30", "US100", "UK100", "DE40", "JP225",
                    "BTCUSD", "ETHUSD", "LTCUSD", "XRPUSD", "BCHUSD"}


def classify_symbol(symbol: str) -> str:
    """
    Classify a symbol into IC Markets commission categories.
    Returns: 'forex_metals', 'stock_cfd', or 'index_crypto'.
    """
    s = symbol.upper().replace("/", "")
    if s in _METALS:
        return "forex_metals"
    if s in _ZERO_COMMISSION:
        return "index_crypto"
    # Standard 6-char Forex pairs (EURUSD, GBPJPY, etc.)
    if len(s) == 6 and s.isalpha():
        return "forex_metals"
    # Stock CFDs: short alpha symbols (AAPL, TSLA, AMZN, etc.)
    if len(s) <= 5 and s.isalpha():
        return "stock_cfd"
    # Anything with digits in name is likely an index
    if any(c.isdigit() for c in s):
        return "index_crypto"
    return "forex_metals"  # conservative default


def calculate_commission(symbol: str, lot_size: float,
                         notional_usd: float) -> float:
    """
    Calculate round-turn commission for IC Markets Raw Spread accounts.

    Forex & Metals: $7.00 per 1.0 standard lot round-turn.
    Stock CFDs:     0.1% per side (0.2% round-turn) of notional value.
    Indices/Crypto: $0 commission (cost embedded in spread).
    """
    cat = classify_symbol(symbol)
    if cat == "forex_metals":
        return lot_size * 7.00
    elif cat == "stock_cfd":
        return notional_usd * 0.001 * 2
    else:
        return 0.0


# ──────────────────────────────────────────────
# Overnight Swap (Financing) Engine
# ──────────────────────────────────────────────
_JPY_PAIRS = {"USDJPY", "EURJPY", "GBPJPY", "AUDJPY", "NZDJPY", "CADJPY", "CHFJPY"}

def _get_point_size(symbol: str) -> float:
    """
    Return the minimum price increment (point) for a symbol.
    5-digit Forex: 0.00001, 3-digit JPY pairs: 0.001,
    Gold: 0.01, Silver: 0.001.
    """
    s = symbol.upper().replace("/", "")
    if s in _JPY_PAIRS:
        return 0.001
    if s == "XAUUSD":
        return 0.01
    if s == "XAGUSD":
        return 0.001
    return 0.00001


def _count_swap_days(entry_time: pd.Timestamp, exit_time: pd.Timestamp) -> int:
    """
    Count the number of swap-eligible rollovers between entry and exit.
    Forex rollover occurs daily at 22:00 UTC. Each crossing = 1 swap day.
    Triple Swap Wednesday: crossing Wednesday 22:00 UTC counts as 3 days
    (covers Saturday + Sunday when markets are closed).

    Returns total swap days including the Wednesday triple.
    """
    # Normalize to UTC
    entry_utc = entry_time.tz_convert("UTC") if entry_time.tzinfo else entry_time
    exit_utc = exit_time.tz_convert("UTC") if exit_time.tzinfo else exit_time

    # Find the first rollover after entry
    rollover_hour = 22
    if entry_utc.hour < rollover_hour:
        first_rollover = entry_utc.normalize() + pd.Timedelta(hours=rollover_hour)
    else:
        first_rollover = (entry_utc.normalize() + pd.Timedelta(days=1)
                          + pd.Timedelta(hours=rollover_hour))

    if first_rollover >= exit_utc:
        return 0  # trade closed before any rollover

    swap_days = 0
    current = first_rollover
    while current < exit_utc:
        if current.weekday() == 2:  # Wednesday
            swap_days += 3
        else:
            swap_days += 1
        current += pd.Timedelta(days=1)

    return swap_days


# ──────────────────────────────────────────────
# Main Backtest Engine
# ──────────────────────────────────────────────
def run_backtest(
    signals: pd.DataFrame,
    ohlc: pd.DataFrame,
    symbol: str = "EURUSD",
    capital: float = 10_000,
    risk_pct: float = 0.01,
    regime_series: Optional[pd.Series] = None,
    fx_rate: Optional[pd.Series] = None,
    swap_long: float = -5.0,
    swap_short: float = -2.0,
) -> dict:
    """
    Run a trade-by-trade backtest from pre-computed signals.

    Parameters
    ----------
    signals : DataFrame with columns entries_long, entries_short,
              sl_long, tp_long, sl_short, tp_short, regime
    ohlc : DataFrame with Open, High, Low, Close columns aligned to signals.
           Intra-bar High/Low are used for accurate SL/TP evaluation.
    symbol : Trading symbol (used for commission classification)
    capital : starting equity (in account currency, e.g. USD)
    risk_pct : fraction of balance risked per trade (for position sizing)
    regime_series : Optional full regime Series for chart overlay
    fx_rate : Series converting quote currency to account currency (USD).
              1.0 for USD-quoted pairs (EURUSD). For GBPJPY this would be
              1/USDJPY so that PnL_in_JPY * fx_rate = PnL_in_USD.
              If None, defaults to 1.0 (assumes USD-quoted pair).
    swap_long : Overnight swap rate in points for BUY positions (typically negative).
    swap_short : Overnight swap rate in points for SELL positions (typically negative).

    Returns
    -------
    dict with keys: metrics, trades, equity_curve, close, regime_series
    """
    ohlc = ohlc.loc[signals.index].copy()
    close = ohlc["Close"]
    point = _get_point_size(symbol)

    if fx_rate is None:
        fx_rate = pd.Series(1.0, index=close.index, name="fx_rate")
    else:
        fx_rate = fx_rate.reindex(close.index, method="ffill").bfill().fillna(1.0)

    # ── Trade-by-trade simulation ──
    trades_log = []
    equity = [capital]
    balance = capital
    in_position = False
    position_dir = None
    entry_price = 0.0
    sl = 0.0
    tp = 0.0
    lot_size = 0.0
    entry_time = None
    entry_fx = 1.0
    cooldown_until = None

    for i in range(len(close)):
        ts = close.index[i]
        c = close.iloc[i]
        bar_high = ohlc["High"].iloc[i]
        bar_low = ohlc["Low"].iloc[i]
        rate = fx_rate.iloc[i]

        if cooldown_until is not None and ts < cooldown_until:
            equity.append(balance)
            continue

        if in_position:
            hit_sl = False
            hit_tp = False
            exit_price = c

            if position_dir == "BUY":
                if bar_low <= sl:
                    hit_sl = True
                    exit_price = sl
                elif bar_high >= tp:
                    hit_tp = True
                    exit_price = tp
            else:
                if bar_high >= sl:
                    hit_sl = True
                    exit_price = sl
                elif bar_low <= tp:
                    hit_tp = True
                    exit_price = tp

            current_regime = signals["regime"].iloc[i] if "regime" in signals.columns else None
            regime_flip = False
            if current_regime:
                if position_dir == "BUY" and current_regime == "Bear Trend":
                    regime_flip = True
                elif position_dir == "SELL" and current_regime == "Bull Trend":
                    regime_flip = True

            if hit_sl or hit_tp or regime_flip:
                if regime_flip:
                    exit_price = c

                if position_dir == "BUY":
                    pnl_quote = (exit_price - entry_price) * lot_size * CONTRACT_SIZE
                else:
                    pnl_quote = (entry_price - exit_price) * lot_size * CONTRACT_SIZE

                pnl = pnl_quote * entry_fx

                # Dynamic IC Markets commission
                notional_usd = lot_size * CONTRACT_SIZE * c * rate
                commission_cost = calculate_commission(symbol, lot_size, notional_usd)
                pnl -= commission_cost

                # Overnight swap (financing) fee
                swap_days = _count_swap_days(entry_time, ts)
                tick_value = point * CONTRACT_SIZE * entry_fx
                swap_points = swap_long if position_dir == "BUY" else swap_short
                swap_cost = swap_points * tick_value * lot_size * swap_days
                pnl += swap_cost  # swap_points are typically negative, so this deducts

                balance += pnl

                trades_log.append({
                    "entry_time": entry_time,
                    "exit_time": ts,
                    "direction": position_dir,
                    "entry_price": entry_price,
                    "exit_price": exit_price,
                    "sl": sl,
                    "tp": tp,
                    "lot_size": lot_size,
                    "pnl": round(pnl, 2),
                    "pnl_pct": round(pnl / (balance - pnl) * 100, 2) if (balance - pnl) != 0 else 0.0,
                    "balance_after": round(balance, 2),
                    "exit_reason": "SL" if hit_sl else ("TP" if hit_tp else "Regime Flip"),
                    "commission": round(commission_cost, 2),
                    "swap": round(swap_cost, 2),
                    "fx_rate": round(entry_fx, 6),
                })

                in_position = False
                cooldown_until = ts + pd.Timedelta(hours=1)

            equity.append(balance)
            continue

        long_entry = bool(signals["entries_long"].iloc[i]) if "entries_long" in signals.columns else False
        short_entry = bool(signals["entries_short"].iloc[i]) if "entries_short" in signals.columns else False

        if long_entry and not np.isnan(signals["sl_long"].iloc[i]):
            entry_price = c
            sl = signals["sl_long"].iloc[i]
            tp = signals["tp_long"].iloc[i]
            position_dir = "BUY"
            entry_fx = rate

            pip_distance = abs(entry_price - sl)
            if pip_distance > 0:
                risk_amount = balance * risk_pct
                distance_value = pip_distance * CONTRACT_SIZE * rate
                lot_size = risk_amount / distance_value
                lot_size = max(0.01, round(lot_size, 2))
            else:
                lot_size = 0.01

            in_position = True
            entry_time = ts
            cooldown_until = None

        elif short_entry and not np.isnan(signals["sl_short"].iloc[i]):
            entry_price = c
            sl = signals["sl_short"].iloc[i]
            tp = signals["tp_short"].iloc[i]
            position_dir = "SELL"
            entry_fx = rate

            pip_distance = abs(sl - entry_price)
            if pip_distance > 0:
                risk_amount = balance * risk_pct
                distance_value = pip_distance * CONTRACT_SIZE * rate
                lot_size = risk_amount / distance_value
                lot_size = max(0.01, round(lot_size, 2))
            else:
                lot_size = 0.01

            in_position = True
            entry_time = ts
            cooldown_until = None

        equity.append(balance)

    equity_series = pd.Series(equity[:len(close)], index=close.index)

    trades_df = pd.DataFrame(trades_log) if trades_log else pd.DataFrame()

    # Add trade duration column
    if len(trades_df) > 0:
        trades_df["duration"] = pd.to_datetime(trades_df["exit_time"]) - pd.to_datetime(trades_df["entry_time"])

    metrics = _compute_metrics(equity_series, trades_df, capital, close)

    return {
        "metrics": metrics,
        "trades": trades_df,
        "equity_curve": equity_series,
        "close": close,
        "regime_series": regime_series,
    }


# ──────────────────────────────────────────────
# Freqtrade-Style Metrics Engine
# ──────────────────────────────────────────────
def _compute_metrics(
    equity: pd.Series,
    trades: pd.DataFrame,
    initial_capital: float,
    close: pd.Series,
) -> dict:
    """
    Compute exhaustive Freqtrade-style performance metrics.
    All fields needed for the three output tables are computed here.
    """
    returns = equity.pct_change().dropna()
    total_return = (equity.iloc[-1] / initial_capital - 1) * 100
    abs_profit = equity.iloc[-1] - initial_capital

    # Buy & Hold
    bh_return = (close.iloc[-1] / close.iloc[0] - 1) * 100

    # Annualized ratios (M5 bars ~ 252*288 per year)
    bars_per_year = 252 * 288
    downside = returns[returns < 0]
    downside_std = downside.std() * np.sqrt(bars_per_year) if len(downside) > 0 else 1e-9
    mean_return = returns.mean() * bars_per_year
    sortino = mean_return / downside_std if downside_std > 0 else 0.0
    std_return = returns.std() * np.sqrt(bars_per_year) if returns.std() > 0 else 1e-9
    sharpe = mean_return / std_return

    # Max Drawdown
    cummax = equity.cummax()
    drawdown_series = equity - cummax
    drawdown_pct_series = drawdown_series / cummax
    max_dd_abs = drawdown_series.min()
    max_dd_pct = drawdown_pct_series.min() * 100

    # Backtest period
    bt_start = close.index[0]
    bt_end = close.index[-1]
    bt_duration = bt_end - bt_start

    n_trades = len(trades)
    if n_trades == 0:
        return _empty_metrics(initial_capital, total_return, bh_return,
                              sortino, sharpe, max_dd_abs, max_dd_pct,
                              bt_start, bt_end, bt_duration)

    wins = trades[trades["pnl"] > 0]
    losses = trades[trades["pnl"] < 0]
    draws = trades[trades["pnl"] == 0]
    n_wins = len(wins)
    n_losses = len(losses)
    n_draws = len(draws)
    win_rate = n_wins / n_trades * 100

    longs = trades[trades["direction"] == "BUY"]
    shorts = trades[trades["direction"] == "SELL"]

    # Profit Factor
    gross_profit = wins["pnl"].sum() if n_wins > 0 else 0.0
    gross_loss = abs(losses["pnl"].sum()) if n_losses > 0 else 0.0
    profit_factor = gross_profit / gross_loss if gross_loss > 0 else float("inf")

    # Expectancy (average PnL per trade)
    expectancy = trades["pnl"].mean()
    # Expectancy ratio: expectancy / avg_loss (Van Tharp style)
    avg_loss_abs = abs(losses["pnl"].mean()) if n_losses > 0 else 1.0
    expectancy_ratio = expectancy / avg_loss_abs if avg_loss_abs > 0 else 0.0

    # SQN = System Quality Number (Van Tharp)
    # SQN = sqrt(N) * expectancy / stdev(pnl)
    pnl_std = trades["pnl"].std()
    sqn = np.sqrt(n_trades) * expectancy / pnl_std if pnl_std > 0 else 0.0

    # Durations
    avg_duration = trades["duration"].mean() if "duration" in trades.columns else pd.Timedelta(0)
    avg_win_duration = wins["duration"].mean() if n_wins > 0 and "duration" in trades.columns else pd.Timedelta(0)
    avg_loss_duration = losses["duration"].mean() if n_losses > 0 and "duration" in trades.columns else pd.Timedelta(0)

    # Commission & swap totals
    total_commission = trades["commission"].sum() if "commission" in trades.columns else 0.0
    total_swap = trades["swap"].sum() if "swap" in trades.columns else 0.0

    # Per-direction stats
    def _direction_stats(df: pd.DataFrame) -> dict:
        if len(df) == 0:
            return {"trades": 0, "avg_pnl_pct": 0.0, "tot_pnl": 0.0,
                    "avg_duration": pd.Timedelta(0),
                    "wins": 0, "draws": 0, "losses": 0, "win_pct": 0.0}
        w = df[df["pnl"] > 0]
        l = df[df["pnl"] < 0]
        d = df[df["pnl"] == 0]
        return {
            "trades": len(df),
            "avg_pnl_pct": df["pnl_pct"].mean() if "pnl_pct" in df.columns else 0.0,
            "tot_pnl": df["pnl"].sum(),
            "avg_duration": df["duration"].mean() if "duration" in df.columns else pd.Timedelta(0),
            "wins": len(w), "draws": len(d), "losses": len(l),
            "win_pct": len(w) / len(df) * 100 if len(df) > 0 else 0.0,
        }

    # Per exit-reason stats
    def _exit_reason_stats(df: pd.DataFrame) -> dict:
        result = {}
        if len(df) == 0:
            return result
        for reason in df["exit_reason"].unique():
            sub = df[df["exit_reason"] == reason]
            result[reason] = _direction_stats(sub)
        return result

    return {
        # Core
        "total_return_pct": round(total_return, 2),
        "abs_profit": round(abs_profit, 2),
        "buy_hold_pct": round(bh_return, 2),
        "final_equity": round(equity.iloc[-1], 2),
        "initial_capital": initial_capital,

        # Risk ratios
        "sortino": round(sortino, 3),
        "sharpe": round(sharpe, 3),
        "sqn": round(sqn, 2),
        "profit_factor": round(profit_factor, 2),
        "expectancy": round(expectancy, 2),
        "expectancy_ratio": round(expectancy_ratio, 2),

        # Drawdown
        "max_dd_abs": round(max_dd_abs, 2),
        "max_dd_pct": round(max_dd_pct, 2),

        # Trade counts
        "total_trades": n_trades,
        "wins": n_wins,
        "draws": n_draws,
        "losses": n_losses,
        "win_rate": round(win_rate, 1),

        # Per-trade averages
        "avg_profit_pct": round(trades["pnl_pct"].mean(), 2) if "pnl_pct" in trades.columns else 0.0,
        "avg_win": round(wins["pnl"].mean(), 2) if n_wins > 0 else 0.0,
        "avg_loss": round(losses["pnl"].mean(), 2) if n_losses > 0 else 0.0,

        # Durations
        "avg_duration": avg_duration,
        "avg_win_duration": avg_win_duration,
        "avg_loss_duration": avg_loss_duration,

        # Commission & Swap
        "total_commission": round(total_commission, 2),
        "total_swap": round(total_swap, 2),

        # Direction breakdown
        "long_stats": _direction_stats(longs),
        "short_stats": _direction_stats(shorts),

        # Exit reason breakdown
        "exit_stats": _exit_reason_stats(trades),

        # Backtest period
        "bt_start": bt_start,
        "bt_end": bt_end,
        "bt_duration": bt_duration,
    }


def _empty_metrics(initial_capital, total_return, bh_return,
                   sortino, sharpe, max_dd_abs, max_dd_pct,
                   bt_start, bt_end, bt_duration) -> dict:
    """Return a metrics dict when there are no trades."""
    zero_td = pd.Timedelta(0)
    return {
        "total_return_pct": round(total_return, 2),
        "abs_profit": 0.0,
        "buy_hold_pct": round(bh_return, 2),
        "final_equity": initial_capital,
        "initial_capital": initial_capital,
        "sortino": round(sortino, 3), "sharpe": round(sharpe, 3),
        "sqn": 0.0, "profit_factor": 0.0,
        "expectancy": 0.0, "expectancy_ratio": 0.0,
        "max_dd_abs": round(max_dd_abs, 2), "max_dd_pct": round(max_dd_pct, 2),
        "total_trades": 0, "wins": 0, "draws": 0, "losses": 0, "win_rate": 0.0,
        "avg_profit_pct": 0.0, "avg_win": 0.0, "avg_loss": 0.0,
        "avg_duration": zero_td, "avg_win_duration": zero_td, "avg_loss_duration": zero_td,
        "total_commission": 0.0,
        "total_swap": 0.0,
        "long_stats": {"trades": 0, "avg_pnl_pct": 0.0, "tot_pnl": 0.0,
                       "avg_duration": zero_td, "wins": 0, "draws": 0, "losses": 0, "win_pct": 0.0},
        "short_stats": {"trades": 0, "avg_pnl_pct": 0.0, "tot_pnl": 0.0,
                        "avg_duration": zero_td, "wins": 0, "draws": 0, "losses": 0, "win_pct": 0.0},
        "exit_stats": {},
        "bt_start": bt_start, "bt_end": bt_end, "bt_duration": bt_duration,
    }


# ──────────────────────────────────────────────
# Freqtrade-Style Rich Table Renderers
# ──────────────────────────────────────────────
def _fmt_duration(td) -> str:
    """Format a timedelta as compact '2d 05:30:00' or '0:45:00'."""
    if pd.isna(td) or td == pd.Timedelta(0):
        return "-"
    total_secs = int(td.total_seconds())
    days = total_secs // 86400
    remainder = total_secs % 86400
    hours = remainder // 3600
    minutes = (remainder % 3600) // 60
    if days > 0:
        return f"{days}d {hours:02d}:{minutes:02d}"
    return f"{hours}:{minutes:02d}"


def _pnl_color(val: float) -> str:
    """Return colored string for PnL values."""
    if val > 0:
        return f"[green]{val:,.2f}[/]"
    elif val < 0:
        return f"[red]{val:,.2f}[/]"
    return f"{val:,.2f}"


def _pct_color(val: float) -> str:
    """Return colored string for percentage values."""
    if val > 0:
        return f"[green]{val:.2f}%[/]"
    elif val < 0:
        return f"[red]{val:.2f}%[/]"
    return f"{val:.2f}%"


def print_backtest_report(metrics: dict, trades: pd.DataFrame, console):
    """
    Print three Freqtrade-style Rich tables:
      1. BACKTESTING REPORT (Long vs Short breakdown)
      2. EXIT REASON STATS
      3. SUMMARY METRICS
    """
    from rich.table import Table

    # ── Table 1: BACKTESTING REPORT ──
    t1 = Table(title="BACKTESTING REPORT", show_lines=True,
               title_style="bold cyan", border_style="cyan")
    t1.add_column("", style="bold")
    t1.add_column("Trades", justify="right")
    t1.add_column("Avg Profit %", justify="right")
    t1.add_column("Tot Profit ($)", justify="right")
    t1.add_column("Avg Duration", justify="right")
    t1.add_column("Win  Draw  Loss", justify="center")
    t1.add_column("Win %", justify="right")

    for label, stats in [("Long", metrics["long_stats"]),
                         ("Short", metrics["short_stats"])]:
        color = "green" if label == "Long" else "red"
        wdl = f"{stats['wins']}    {stats['draws']}    {stats['losses']}"
        t1.add_row(
            f"[{color}]{label}[/]",
            str(stats["trades"]),
            _pct_color(stats["avg_pnl_pct"]),
            _pnl_color(stats["tot_pnl"]),
            _fmt_duration(stats["avg_duration"]),
            wdl,
            f"{stats['win_pct']:.1f}%",
        )

    # TOTAL row
    m = metrics
    total_wdl = f"{m['wins']}    {m['draws']}    {m['losses']}"
    t1.add_row(
        "[bold]TOTAL[/]",
        f"[bold]{m['total_trades']}[/]",
        f"[bold]{_pct_color(m['avg_profit_pct'])}[/]",
        f"[bold]{_pnl_color(m['abs_profit'])}[/]",
        f"[bold]{_fmt_duration(m['avg_duration'])}[/]",
        f"[bold]{total_wdl}[/]",
        f"[bold]{m['win_rate']:.1f}%[/]",
    )

    console.print(t1)

    # ── Table 2: EXIT REASON STATS ──
    t2 = Table(title="EXIT REASON STATS", show_lines=True,
               title_style="bold yellow", border_style="yellow")
    t2.add_column("Exit Reason", style="bold")
    t2.add_column("Trades", justify="right")
    t2.add_column("Avg Profit %", justify="right")
    t2.add_column("Tot Profit ($)", justify="right")
    t2.add_column("Avg Duration", justify="right")
    t2.add_column("Win  Draw  Loss", justify="center")
    t2.add_column("Win %", justify="right")

    exit_stats = m.get("exit_stats", {})
    for reason in ["TP", "SL", "Regime Flip"]:
        if reason not in exit_stats:
            continue
        s = exit_stats[reason]
        reason_color = "green" if reason == "TP" else ("red" if reason == "SL" else "yellow")
        wdl = f"{s['wins']}    {s['draws']}    {s['losses']}"
        t2.add_row(
            f"[{reason_color}]{reason}[/]",
            str(s["trades"]),
            _pct_color(s["avg_pnl_pct"]),
            _pnl_color(s["tot_pnl"]),
            _fmt_duration(s["avg_duration"]),
            wdl,
            f"{s['win_pct']:.1f}%",
        )

    console.print(t2)

    # ── Table 3: SUMMARY METRICS ──
    t3 = Table(title="SUMMARY METRICS", show_lines=True,
               title_style="bold magenta", border_style="magenta")
    t3.add_column("Metric", style="bold")
    t3.add_column("Value", justify="right")

    rows = [
        ("Backtesting from", str(m["bt_start"])[:10]),
        ("Backtesting to", str(m["bt_end"])[:10]),
        ("Backtest Duration", _fmt_duration(m["bt_duration"])),
        ("", ""),
        ("Total/Daily Avg Trades", f"{m['total_trades']}"),
        ("Starting Balance", f"${m['initial_capital']:,.2f}"),
        ("Final Balance", f"${m['final_equity']:,.2f}"),
        ("Absolute Profit", _pnl_color(m["abs_profit"])),
        ("Total Return", _pct_color(m["total_return_pct"])),
        ("Buy & Hold Return", _pct_color(m["buy_hold_pct"])),
        ("", ""),
        ("Max Drawdown (Abs)", _pnl_color(m["max_dd_abs"])),
        ("Max Drawdown (%)", _pct_color(m["max_dd_pct"])),
        ("", ""),
        ("Sharpe Ratio", f"{m['sharpe']:.3f}"),
        ("Sortino Ratio", f"{m['sortino']:.3f}"),
        ("SQN", f"{m['sqn']:.2f}"),
        ("Profit Factor", f"{m['profit_factor']:.2f}"),
        ("Expectancy ($)", _pnl_color(m["expectancy"])),
        ("Expectancy Ratio", f"{m['expectancy_ratio']:.2f}"),
        ("", ""),
        ("Win Rate", f"{m['win_rate']:.1f}%"),
        ("Avg Win ($)", _pnl_color(m["avg_win"])),
        ("Avg Loss ($)", _pnl_color(m["avg_loss"])),
        ("Long Profit ($)", _pnl_color(m["long_stats"]["tot_pnl"])),
        ("Short Profit ($)", _pnl_color(m["short_stats"]["tot_pnl"])),
        ("", ""),
        ("Avg Trade Duration", _fmt_duration(m["avg_duration"])),
        ("Avg Win Duration", _fmt_duration(m["avg_win_duration"])),
        ("Avg Loss Duration", _fmt_duration(m["avg_loss_duration"])),
        ("", ""),
        ("Total Commission", f"[dim]${m['total_commission']:,.2f}[/]"),
        ("Total Swap Fees", f"[dim]${m['total_swap']:,.2f}[/]"),
    ]

    for label, val in rows:
        if label == "" and val == "":
            t3.add_row("─" * 25, "─" * 15)
        else:
            t3.add_row(label, val)

    console.print(t3)


# ──────────────────────────────────────────────
# Terminal Chart (fast, instant)
# ──────────────────────────────────────────────
def build_terminal_chart(result: dict, symbol: str = "", width: int = 120, height: int = 20):
    """
    Render price + equity directly in the terminal using plotext.
    Downsamples to ~width points for clean rendering. Instant output.
    """
    import sys
    import plotext as plt

    if sys.platform == "win32":
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    close = result["close"]
    equity = result["equity_curve"]

    step = max(1, len(close) // width)
    close_ds = close.iloc[::step]
    equity_ds = equity.iloc[::step]

    n_labels = 6
    label_step = max(1, len(close_ds) // n_labels)
    labels = [close_ds.index[i].strftime("%Y-%m-%d") for i in range(0, len(close_ds), label_step)]
    label_positions = list(range(1, len(close_ds) + 1, label_step))

    plt.clear_figure()
    plt.subplots(2, 1)

    plt.subplot(1, 1)
    plt.plot(list(close_ds.values), label="Close", color="blue")
    plt.title(f"{symbol} Price")
    plt.xticks(label_positions, labels)
    plt.plotsize(width, height)

    trades = result.get("trades")
    if trades is not None and len(trades) > 0:
        for _, t in trades.iterrows():
            entry_idx = close_ds.index.get_indexer([t["entry_time"]], method="nearest")[0] + 1
            if t["direction"] == "BUY":
                plt.scatter([entry_idx], [t["entry_price"]], marker="dot", color="green")
            else:
                plt.scatter([entry_idx], [t["entry_price"]], marker="dot", color="red")

    plt.subplot(2, 1)
    eq_color = "green" if equity_ds.iloc[-1] >= equity_ds.iloc[0] else "red"
    plt.plot(list(equity_ds.values), label="Equity", color=eq_color)
    plt.title("Equity Curve")
    plt.xticks(label_positions, labels)
    plt.plotsize(width, height)

    plt.show()


# ──────────────────────────────────────────────
# Plotly Visualization (optional HTML export)
# ──────────────────────────────────────────────
def build_backtest_chart(result: dict, symbol: str = "") -> go.Figure:
    """
    Plotly HTML chart (optional export). For daily use, prefer build_terminal_chart().
    Downsamples data to prevent browser lag on large M5 datasets.
    """
    close = result["close"]
    equity = result["equity_curve"]
    trades = result["trades"]
    regime_series = result.get("regime_series")

    max_points = 5000
    step = max(1, len(close) // max_points)
    close = close.iloc[::step]
    equity = equity.iloc[::step]

    fig = make_subplots(
        rows=2, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.03,
        row_heights=[0.6, 0.4],
        subplot_titles=[f"{symbol} Price + Regime Overlay", "Equity Curve"],
    )

    fig.add_trace(
        go.Scatter(
            x=close.index, y=close.values,
            name="Close", line=dict(color="#2196F3", width=1),
        ),
        row=1, col=1,
    )

    fig.add_trace(
        go.Scatter(
            x=equity.index, y=equity.values,
            name="Equity", line=dict(color="#FF9800", width=2),
            fill="tozeroy", fillcolor="rgba(255,152,0,0.1)",
        ),
        row=2, col=1,
    )

    if regime_series is not None:
        regime_aligned = regime_series.reindex(close.index, method="ffill")
        _add_regime_shading(fig, regime_aligned, row=1, max_blocks=200)
        _add_regime_shading(fig, regime_aligned, row=2, max_blocks=200)

    if len(trades) > 0:
        buys = trades[trades["direction"] == "BUY"]
        sells = trades[trades["direction"] == "SELL"]

        if len(buys) > 0:
            fig.add_trace(go.Scatter(
                x=buys["entry_time"], y=buys["entry_price"],
                mode="markers", name="Long Entry",
                marker=dict(symbol="triangle-up", size=10, color="lime"),
            ), row=1, col=1)

        if len(sells) > 0:
            fig.add_trace(go.Scatter(
                x=sells["entry_time"], y=sells["entry_price"],
                mode="markers", name="Short Entry",
                marker=dict(symbol="triangle-down", size=10, color="red"),
            ), row=1, col=1)

    fig.update_layout(
        template="plotly_dark",
        height=900,
        showlegend=True,
        title=f"Regime-Based Backtest: {symbol}",
        xaxis_rangeslider_visible=False,
    )
    return fig


def _add_regime_shading(fig: go.Figure, regime_series: pd.Series, row: int, max_blocks: int = 200):
    """Add colored background rectangles for each regime block."""
    colors = {
        "Bull Trend": "rgba(0,200,0,0.08)",
        "Bear Trend": "rgba(200,0,0,0.08)",
        "Neutral": "rgba(128,128,128,0.06)",
    }

    if len(regime_series) == 0:
        return

    changes = regime_series != regime_series.shift(1)
    block_starts = regime_series.index[changes]

    if len(block_starts) > max_blocks:
        step = len(block_starts) // max_blocks
        block_starts = block_starts[::step]

    for i, start in enumerate(block_starts):
        end = block_starts[i + 1] if i + 1 < len(block_starts) else regime_series.index[-1]
        regime = regime_series.loc[start]
        color = colors.get(regime, "rgba(128,128,128,0.05)")
        fig.add_vrect(
            x0=start, x1=end,
            fillcolor=color, layer="below", line_width=0,
            row=row, col=1,
        )
