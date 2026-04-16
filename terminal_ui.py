"""
terminal_ui.py — Terminal Operations (Backtest & Hyperopt)
===========================================================
Rich terminal interface for:
  Tab 1: Backtesting with OOS split + Freqtrade-style reports
  Tab 2: Optuna hyperparameter optimization (regime-conditional, OOS)

Run:  python terminal_ui.py
"""

import json
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("TerminalUI")

CONFIGS_DIR = Path(__file__).parent / "configs"
CONFIGS_DIR.mkdir(exist_ok=True)


def print_banner():
    from rich.console import Console
    from rich.panel import Panel

    console = Console()
    console.print(Panel.fit(
        "[bold cyan]Regime-Based Forex Trading System[/]\n"
        "[dim]Backtest & Hyperopt Terminal[/]",
        border_style="cyan",
    ))


def select_menu():
    """Main menu selection."""
    from rich.console import Console

    console = Console()
    console.print("\n[bold]Select Operation:[/]")
    console.print("  [1] Backtest")
    console.print("  [2] Hyperopt (Optuna)")
    console.print("  [3] Exit")
    choice = input("\n> ").strip()
    return choice


# ──────────────────────────────────────────────
# OOS Data Splitting
# ──────────────────────────────────────────────
def _split_oos(data: dict, train_ratio: float) -> tuple[dict, dict]:
    """
    Chronologically split multi-TF data into train (IS) and test (OOS).
    The split point is determined by the H1 timeframe, then all TFs are
    clipped to their corresponding date boundaries.

    Returns (data_train, data_test) dicts keyed by timeframe.
    """
    h1 = data["H1"]
    split_idx = int(len(h1) * train_ratio)
    split_time = h1.index[split_idx]

    data_train = {}
    data_test = {}
    for tf, df in data.items():
        data_train[tf] = df.loc[df.index < split_time].copy()
        data_test[tf] = df.loc[df.index >= split_time].copy()

    return data_train, data_test


# ──────────────────────────────────────────────
# Tab 1: BACKTEST
# ──────────────────────────────────────────────
def run_backtest_terminal():
    from rich.console import Console
    from rich.table import Table
    from rich.prompt import Prompt, FloatPrompt, IntPrompt

    console = Console()
    console.print("\n[bold green]═══ BACKTEST ═══[/]\n")

    # Gather parameters
    symbol = Prompt.ask("Symbol", default="EURUSD")
    capital = FloatPrompt.ask("Starting Capital ($)", default=10000.0)
    days_back = IntPrompt.ask("Days of history", default=365)
    risk_pct = FloatPrompt.ask("Risk per trade (%)", default=1.0) / 100
    swap_long = FloatPrompt.ask("Swap Long (points)", default=-5.0)
    swap_short = FloatPrompt.ask("Swap Short (points)", default=-2.0)
    train_split = FloatPrompt.ask("Train/Test split (0.0=full IS, 0.7=70% train)", default=0.7)

    # Load optimized params?
    use_opt = Prompt.ask("Load optimized params? (y/n)", default="n")
    params = None
    if use_opt.lower() == "y":
        opt_file = Prompt.ask("Params filename", default="optimized_params.json")
        opt_path = CONFIGS_DIR / opt_file
        if opt_path.exists():
            with open(opt_path) as f:
                params = json.load(f).get("params")
            console.print(f"[green]Loaded params from {opt_path}[/]")
        else:
            console.print(f"[yellow]File not found, using defaults[/]")

    console.print("\n[dim]Fetching data from Dukascopy...[/]")
    console.print("[dim](First download may take several minutes — subsequent runs use local parquet cache)[/]")
    from data_engine import fetch_historical, fetch_quote_to_usd_rate, get_quote_currency
    from hmm_engine import RegimeDetector
    from strategies.custom_strat import RegimeSubordinatedStrategy
    from backtester import (
        run_backtest, build_backtest_chart, build_terminal_chart,
        print_backtest_report, classify_symbol,
    )

    # Instantiate strategy to discover its base timeframe
    strategy = RegimeSubordinatedStrategy(params=params)
    base_tf = strategy.base_timeframe

    # Fetch multi-timeframe data (Dukascopy has no intraday limits)
    timeframes = list(dict.fromkeys(["D1", "H4", "H1", base_tf]))
    data = {}
    for tf in timeframes:
        try:
            console.print(f"  Fetching {symbol} {tf}...")
            data[tf] = fetch_historical(symbol, tf, days_back=days_back)
            console.print(f"    [green]{len(data[tf])} bars[/]")
        except Exception as e:
            console.print(f"    [red]{e}[/]")

    if "H1" not in data or base_tf not in data:
        console.print(f"[red]Cannot run backtest without H1 and {base_tf} data[/]")
        return

    # ── OOS Split ──
    use_oos = 0.0 < train_split < 1.0
    if use_oos:
        data_train, data_test = _split_oos(data, train_split)
        pct_train = int(train_split * 100)
        pct_test = 100 - pct_train
        console.print(f"\n[bold cyan]Out-of-Sample Mode: {pct_train}% Train / {pct_test}% Test[/]")
        console.print(f"  Train H1: {len(data_train['H1'])} bars  [{data_train['H1'].index[0].date()} -> {data_train['H1'].index[-1].date()}]")
        console.print(f"  Test  H1: {len(data_test['H1'])} bars  [{data_test['H1'].index[0].date()} -> {data_test['H1'].index[-1].date()}]")

        # HMM trains ONLY on in-sample
        console.print(f"\n[dim]Training HMM on IN-SAMPLE data ({pct_train}%)...[/]")
        hmm = RegimeDetector()
        hmm.fit(data_train["H1"])

        # Predict regimes on the FULL H1 (train + test)
        # The HMM model was only fitted on train, but can predict on any data
        regime_series_full = hmm.predict(data["H1"])

        # Show regime summary for train portion
        summary = hmm.regime_summary(data_train["H1"])
    else:
        console.print("\n[bold yellow]Full In-Sample Mode (no OOS split)[/]")
        data_test = data
        console.print("\n[dim]Training HMM (3-state)...[/]")
        hmm = RegimeDetector()
        hmm.fit(data["H1"])
        regime_series_full = hmm.predict(data["H1"])
        summary = hmm.regime_summary(data["H1"])

    # Regime summary table
    regime_table = Table(title="Regime Distribution (H1 Train)")
    regime_table.add_column("Regime", style="bold")
    regime_table.add_column("Count")
    regime_table.add_column("%")
    regime_table.add_column("Mean Return")
    regime_table.add_column("Std Return")
    for regime, row in summary.iterrows():
        color = "green" if regime == "Bull Trend" else ("red" if regime == "Bear Trend" else "white")
        regime_table.add_row(
            f"[{color}]{regime}[/]",
            str(int(row["count"])),
            f"{row['pct']:.1f}%",
            f"{row['mean_return']:.6f}",
            f"{row['std_return']:.6f}",
        )
    console.print(regime_table)

    # Fetch quote-to-USD conversion rate (on test data range)
    quote_ccy = get_quote_currency(symbol)
    if quote_ccy != "USD":
        console.print(f"  [cyan]Quote currency: {quote_ccy} -- fetching conversion rate...[/]")
    fx_rate = fetch_quote_to_usd_rate(symbol, data_test[base_tf].index, days_back=days_back)
    if quote_ccy != "USD":
        console.print(f"    [green]FX rate range: {fx_rate.min():.6f} - {fx_rate.max():.6f}[/]")

    # Strategy signals — generated on TEST data only
    # Regime series must be sliced to test period
    regime_test = regime_series_full.loc[regime_series_full.index >= data_test["H1"].index[0]]

    console.print("\n[dim]Generating signals on OUT-OF-SAMPLE data...[/]" if use_oos
                  else "\n[dim]Generating signals...[/]")
    signals = strategy.generate_signals_vectorized(data_test, regime_test)

    n_long = signals["entries_long"].sum()
    n_short = signals["entries_short"].sum()
    console.print(f"  Signals: [green]{n_long} longs[/] | [red]{n_short} shorts[/]")

    # Commission info
    sym_class = classify_symbol(symbol)
    comm_label = {"forex_metals": "$7/lot RT", "stock_cfd": "0.1%/side", "index_crypto": "$0 (spread)"}
    console.print(f"  Commission: [dim]{comm_label.get(sym_class, '?')} ({sym_class})[/]")

    if n_long + n_short == 0:
        console.print("[yellow]No trades generated — consider loosening parameters via Hyperopt[/]")
        return

    # Run backtest on OOS data
    console.print("\n[dim]Running backtest...[/]")
    result = run_backtest(
        signals=signals,
        ohlc=data_test[base_tf],
        symbol=symbol,
        capital=capital,
        risk_pct=risk_pct,
        regime_series=regime_test,
        fx_rate=fx_rate,
        swap_long=swap_long,
        swap_short=swap_short,
    )

    # ── Freqtrade-style output ──
    console.print("")
    print_backtest_report(result["metrics"], result["trades"], console)

    # Trade log
    trades = result["trades"]
    if len(trades) > 0:
        console.print(f"\n[bold]Trade Log ({len(trades)} trades):[/]")
        trade_table = Table(show_lines=True)
        for col in ["entry_time", "exit_time", "direction", "entry_price",
                     "exit_price", "lot_size", "pnl", "commission", "swap", "exit_reason"]:
            trade_table.add_column(col)
        for _, t in trades.iterrows():
            pnl_style = "green" if t["pnl"] > 0 else "red"
            swap_val = t.get("swap", 0.0)
            trade_table.add_row(
                str(t["entry_time"])[:19],
                str(t["exit_time"])[:19],
                t["direction"],
                f"{t['entry_price']:.5f}",
                f"{t['exit_price']:.5f}",
                f"{t['lot_size']:.2f}",
                f"[{pnl_style}]{t['pnl']:.2f}[/]",
                f"${t['commission']:.2f}",
                f"${swap_val:.2f}",
                t["exit_reason"],
            )
        console.print(trade_table)

    # Terminal chart
    console.print("")
    build_terminal_chart(result, symbol=symbol)

    # Optional HTML export
    export = Prompt.ask("\nExport interactive HTML chart? (y/n)", default="n")
    if export.lower() == "y":
        console.print("[dim]Generating Plotly chart...[/]")
        fig = build_backtest_chart(result, symbol=symbol)
        chart_path = Path(__file__).parent / "backtest_chart.html"
        fig.write_html(str(chart_path))
        console.print(f"[green]Chart saved -> {chart_path}[/]")
        import webbrowser
        webbrowser.open(str(chart_path))


# ──────────────────────────────────────────────
# Tab 2: HYPEROPT (Optuna)
# ──────────────────────────────────────────────
def run_hyperopt_terminal():
    from rich.console import Console
    from rich.table import Table
    from rich.prompt import Prompt, FloatPrompt, IntPrompt

    console = Console()
    console.print("\n[bold magenta]═══ HYPEROPT (Optuna) ═══[/]\n")

    symbol = Prompt.ask("Symbol", default="EURUSD")
    capital = FloatPrompt.ask("Starting Capital ($)", default=10000.0)
    days_back = IntPrompt.ask("Days of history", default=365)
    n_trials = IntPrompt.ask("Optuna trials", default=100)
    train_split = FloatPrompt.ask("Train/Test split (0.7 = optimize on 70%)", default=0.7)

    regime_filter = Prompt.ask(
        "Optimize for regime (all/bull/bear)",
        default="all",
    ).lower()

    min_trades = IntPrompt.ask("Min acceptable trade count", default=5)
    swap_long = FloatPrompt.ask("Swap Long (points)", default=-5.0)
    swap_short = FloatPrompt.ask("Swap Short (points)", default=-2.0)
    output_file = Prompt.ask("Output params file", default="optimized_params.json")

    console.print("\n[dim]Fetching data from Dukascopy...[/]")
    console.print("[dim](First download may take several minutes — subsequent runs use local parquet cache)[/]")
    from data_engine import fetch_historical, fetch_quote_to_usd_rate, get_quote_currency
    from hmm_engine import RegimeDetector
    from strategies.custom_strat import RegimeSubordinatedStrategy
    from backtester import run_backtest, print_backtest_report

    # Instantiate strategy to discover its base timeframe
    strategy_template = RegimeSubordinatedStrategy()
    base_tf = strategy_template.base_timeframe

    # Fetch data
    data = {}
    for tf in list(dict.fromkeys(["D1", "H4", "H1", base_tf])):
        try:
            data[tf] = fetch_historical(symbol, tf, days_back=days_back)
            console.print(f"  {tf}: {len(data[tf])} bars")
        except Exception as e:
            console.print(f"  [red]{tf}: {e}[/]")

    if "H1" not in data or base_tf not in data:
        console.print(f"[red]Insufficient data for optimization (need H1 and {base_tf})[/]")
        return

    # ── OOS Split ──
    use_oos = 0.0 < train_split < 1.0
    if use_oos:
        data_train, data_test = _split_oos(data, train_split)
        pct_train = int(train_split * 100)
        console.print(f"\n[bold cyan]OOS: Optimizing on {pct_train}% IS, validating on {100 - pct_train}% OOS[/]")
        console.print(f"  Train: {data_train['H1'].index[0].date()} -> {data_train['H1'].index[-1].date()}")
        console.print(f"  Test:  {data_test['H1'].index[0].date()} -> {data_test['H1'].index[-1].date()}")
    else:
        data_train = data
        data_test = data
        console.print("\n[bold yellow]Full In-Sample optimization (no OOS)[/]")

    # Fetch FX rate for train data (used during optimization)
    quote_ccy = get_quote_currency(symbol)
    if quote_ccy != "USD":
        console.print(f"  [cyan]Quote currency: {quote_ccy} -- fetching conversion rate...[/]")
    fx_rate_train = fetch_quote_to_usd_rate(symbol, data_train[base_tf].index, days_back=days_back)
    fx_rate_test = fetch_quote_to_usd_rate(symbol, data_test[base_tf].index, days_back=days_back) if use_oos else fx_rate_train

    # Train HMM on IS data only
    console.print("\n[dim]Training HMM on IN-SAMPLE data...[/]")
    hmm = RegimeDetector()
    hmm.fit(data_train["H1"])

    # Predict regimes on train and full data
    regime_train = hmm.predict(data_train["H1"])
    regime_full = hmm.predict(data["H1"])
    regime_test = regime_full.loc[regime_full.index >= data_test["H1"].index[0]] if use_oos else regime_full

    if regime_filter == "bull":
        console.print("[green]Optimizing for BULL regime only[/]")
    elif regime_filter == "bear":
        console.print("[red]Optimizing for BEAR regime only[/]")
    else:
        console.print("[cyan]Optimizing across ALL regimes[/]")

    # Optuna objective — runs on TRAIN data
    import optuna

    space = strategy_template.hyperopt_space()

    def objective(trial: optuna.Trial) -> float:
        params = strategy_template.default_params()

        for sp in space:
            if sp.param_type == "int":
                params[sp.name] = trial.suggest_int(sp.name, int(sp.low), int(sp.high), step=int(sp.step))
            else:
                params[sp.name] = trial.suggest_float(sp.name, sp.low, sp.high, step=sp.step)

        if params["fibo_lower"] > params["fibo_upper"]:
            params["fibo_lower"], params["fibo_upper"] = params["fibo_upper"], params["fibo_lower"]

        strategy = RegimeSubordinatedStrategy(params=params)

        if regime_filter == "bull":
            filtered_regime = regime_train.copy()
            filtered_regime[filtered_regime == "Bear Trend"] = "Neutral"
        elif regime_filter == "bear":
            filtered_regime = regime_train.copy()
            filtered_regime[filtered_regime == "Bull Trend"] = "Neutral"
        else:
            filtered_regime = regime_train

        try:
            signals = strategy.generate_signals_vectorized(data_train, filtered_regime)
        except Exception:
            return -1e6

        n_entries = signals["entries_long"].sum() + signals["entries_short"].sum()
        if n_entries < min_trades:
            return -100 + n_entries * 0.1

        result = run_backtest(
            signals=signals,
            ohlc=data_train[base_tf],
            symbol=symbol,
            capital=capital,
            risk_pct=params.get("risk_pct", 0.01),
            regime_series=filtered_regime,
            fx_rate=fx_rate_train,
            swap_long=swap_long,
            swap_short=swap_short,
        )

        m = result["metrics"]
        total_return = m["total_return_pct"]
        max_dd = abs(m["max_dd_pct"])

        if max_dd == 0:
            max_dd = 0.01
        score = total_return / max_dd

        n_trades = m["total_trades"]
        if n_trades >= min_trades:
            score += n_trades * 0.01

        if max_dd > 30:
            score -= (max_dd - 30) * 0.5

        trial.set_user_attr("total_return", total_return)
        trial.set_user_attr("max_dd", max_dd)
        trial.set_user_attr("n_trades", n_trades)
        trial.set_user_attr("sortino", m["sortino"])

        return score

    # Run optimization
    console.print(f"\n[bold]Running {n_trials} Optuna trials (TPE sampler)...[/]\n")
    optuna.logging.set_verbosity(optuna.logging.WARNING)

    study = optuna.create_study(
        direction="maximize",
        sampler=optuna.samplers.TPESampler(seed=42),
        study_name=f"{symbol}_{regime_filter}_opt",
    )
    study.optimize(objective, n_trials=n_trials, show_progress_bar=True)

    # Results
    best = study.best_trial
    console.print(f"\n[bold green]Best Trial #{best.number} (In-Sample)[/]")
    console.print(f"  Score: {best.value:.4f}")
    console.print(f"  Return: {best.user_attrs.get('total_return', 'N/A')}%")
    console.print(f"  Max DD: {best.user_attrs.get('max_dd', 'N/A')}%")
    console.print(f"  Trades: {best.user_attrs.get('n_trades', 'N/A')}")
    console.print(f"  Sortino: {best.user_attrs.get('sortino', 'N/A')}")

    # Best params table
    params_table = Table(title="Optimized Parameters")
    params_table.add_column("Parameter", style="bold")
    params_table.add_column("Value", justify="right")
    params_table.add_column("Default", justify="right", style="dim")

    defaults = strategy_template.default_params()
    best_params = {**defaults, **best.params}
    if best_params["fibo_lower"] > best_params["fibo_upper"]:
        best_params["fibo_lower"], best_params["fibo_upper"] = best_params["fibo_upper"], best_params["fibo_lower"]

    for k, v in best_params.items():
        default_v = defaults.get(k, "—")
        changed = v != default_v
        style = "bold yellow" if changed else ""
        params_table.add_row(
            k,
            f"[{style}]{v}[/]" if style else str(v),
            str(default_v),
        )
    console.print(params_table)

    # ── OOS Validation ──
    if use_oos:
        console.print(f"\n[bold cyan]═══ OUT-OF-SAMPLE VALIDATION ({100 - pct_train}%) ═══[/]")
        best_strategy = RegimeSubordinatedStrategy(params=best_params)

        if regime_filter == "bull":
            oos_regime = regime_test.copy()
            oos_regime[oos_regime == "Bear Trend"] = "Neutral"
        elif regime_filter == "bear":
            oos_regime = regime_test.copy()
            oos_regime[oos_regime == "Bull Trend"] = "Neutral"
        else:
            oos_regime = regime_test

        try:
            oos_signals = best_strategy.generate_signals_vectorized(data_test, oos_regime)
            oos_result = run_backtest(
                signals=oos_signals,
                ohlc=data_test[base_tf],
                symbol=symbol,
                capital=capital,
                risk_pct=best_params.get("risk_pct", 0.01),
                regime_series=oos_regime,
                fx_rate=fx_rate_test,
                swap_long=swap_long,
                swap_short=swap_short,
            )

            print_backtest_report(oos_result["metrics"], oos_result["trades"], console)
        except Exception as e:
            console.print(f"[red]OOS validation failed: {e}[/]")

    # Save params
    output_path = CONFIGS_DIR / output_file
    payload = {
        "strategy": "RegimeSubordinated_MTF",
        "symbol": symbol,
        "regime_filter": regime_filter,
        "trials": n_trials,
        "train_split": train_split,
        "best_score": best.value,
        "best_metrics": {
            "total_return": best.user_attrs.get("total_return"),
            "max_dd": best.user_attrs.get("max_dd"),
            "n_trades": best.user_attrs.get("n_trades"),
            "sortino": best.user_attrs.get("sortino"),
        },
        "params": best_params,
    }
    with open(output_path, "w") as f:
        json.dump(payload, f, indent=2)
    console.print(f"\n[green]Params saved -> {output_path}[/]")

    # Top 5 trials
    console.print("\n[bold]Top 5 Trials (In-Sample):[/]")
    top_table = Table(show_lines=True)
    top_table.add_column("#")
    top_table.add_column("Score")
    top_table.add_column("Return %")
    top_table.add_column("Max DD %")
    top_table.add_column("Trades")
    top_table.add_column("Sortino")

    sorted_trials = sorted(study.trials, key=lambda t: t.value if t.value else -1e9, reverse=True)[:5]
    for t in sorted_trials:
        top_table.add_row(
            str(t.number),
            f"{t.value:.4f}" if t.value else "N/A",
            f"{t.user_attrs.get('total_return', 'N/A')}",
            f"{t.user_attrs.get('max_dd', 'N/A')}",
            f"{t.user_attrs.get('n_trades', 'N/A')}",
            f"{t.user_attrs.get('sortino', 'N/A')}",
        )
    console.print(top_table)


# ──────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────
def main():
    print_banner()

    while True:
        choice = select_menu()
        if choice == "1":
            run_backtest_terminal()
        elif choice == "2":
            run_hyperopt_terminal()
        elif choice == "3":
            print("Goodbye.")
            break
        else:
            print("Invalid choice. Try again.")


if __name__ == "__main__":
    main()
