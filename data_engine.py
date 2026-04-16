"""
data_engine.py — Data & Exchange Connection
=============================================
Handles historical data ingestion (Dukascopy) and live MT5 broker connection.
Dukascopy provides deep M1 history (10+ years). Data is cached locally as
parquet files and resampled to M5/H1/H4/D1 on demand.
"""

import datetime as dt
import logging
import struct
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from io import BytesIO
from lzma import LZMADecompressor, LZMAError, FORMAT_AUTO
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import requests

logger = logging.getLogger(__name__)

# ──────────────────────────────────────────────
# Constants
# ──────────────────────────────────────────────
DATA_HISTORY_DIR = Path(__file__).parent / "data_history"
DATA_HISTORY_DIR.mkdir(exist_ok=True)

CACHE_DIR = Path(__file__).parent / "cache"
CACHE_DIR.mkdir(exist_ok=True)

CONTRACT_SIZE = 100_000  # 1 standard lot

# Dukascopy bi5 URL template (months are 0-indexed)
_DUKA_URL = (
    "https://www.dukascopy.com/datafeed/{symbol}/{year}/{month:02d}/{day:02d}/{hour:02d}h_ticks.bi5"
)
_DUKA_HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
}
_DUKA_TICK_SIZE = 20  # bytes per tick record in bi5 format
_DUKA_MAX_RETRIES = 3
_DUKA_TIMEOUT = 30  # seconds per HTTP request
_DUKA_THREADS = 8   # parallel hour downloads

# Resample rules for each target timeframe
_RESAMPLE_RULES = {
    "M1": "1min",
    "M5": "5min",
    "M15": "15min",
    "M30": "30min",
    "H1": "1h",
    "H4": "4h",
    "D1": "1D",
}

# MT5 timeframe constants (resolved at import time if MT5 available)
MT5_TF = {}

# ──────────────────────────────────────────────
# Quote currency -> USD conversion lookup
# ──────────────────────────────────────────────
# For each quote currency, the tuple is (dukascopy_pair, invert).
#   invert=False: rate = pair price  (e.g. GBPUSD price = USD per GBP)
#   invert=True:  rate = 1 / pair price (e.g. USDJPY price = JPY per USD)
_QUOTE_TO_USD = {
    "USD": None,
    "EUR": ("EURUSD", False),
    "GBP": ("GBPUSD", False),
    "AUD": ("AUDUSD", False),
    "NZD": ("NZDUSD", False),
    "JPY": ("USDJPY", True),
    "CHF": ("USDCHF", True),
    "CAD": ("USDCAD", True),
    "SGD": ("USDSGD", True),
    "HKD": ("USDHKD", True),
    "SEK": ("USDSEK", True),
    "NOK": ("USDNOK", True),
    "DKK": ("USDDKK", True),
    "ZAR": ("USDZAR", True),
    "MXN": ("USDMXN", True),
    "TRY": ("USDTRY", True),
    "CNH": ("USDCNH", True),
    "PLN": ("USDPLN", True),
    "HUF": ("USDHUF", True),
    "CZK": ("USDCZK", True),
}

try:
    import MetaTrader5 as mt5

    MT5_TF = {
        "M1": mt5.TIMEFRAME_M1,
        "M5": mt5.TIMEFRAME_M5,
        "M15": mt5.TIMEFRAME_M15,
        "M30": mt5.TIMEFRAME_M30,
        "H1": mt5.TIMEFRAME_H1,
        "H4": mt5.TIMEFRAME_H4,
        "D1": mt5.TIMEFRAME_D1,
        "W1": mt5.TIMEFRAME_W1,
    }
except ImportError:
    mt5 = None
    logger.warning("MetaTrader5 package not installed — live trading disabled")


# ──────────────────────────────────────────────
# Dukascopy bi5 low-level functions
# ──────────────────────────────────────────────
def _decompress_lzma(data: bytes) -> bytes:
    """Decompress LZMA/bi5 data, handling multi-stream files."""
    results = []
    while True:
        decomp = LZMADecompressor(FORMAT_AUTO, None, None)
        try:
            res = decomp.decompress(data)
        except LZMAError:
            if results:
                break
            else:
                raise
        results.append(res)
        data = decomp.unused_data
        if not data:
            break
        if not decomp.eof:
            raise LZMAError("Compressed data ended before end-of-stream marker")
    return b"".join(results)


def _parse_bi5_ticks(day: dt.date, hour: int, raw: bytes) -> list[tuple]:
    """
    Parse decompressed bi5 binary into tick tuples.
    Each 20-byte record: (ms_offset: u32, ask: u32, bid: u32, ask_vol: f32, bid_vol: f32)
    Returns list of (datetime_utc, ask, bid, ask_volume, bid_volume).
    """
    if len(raw) == 0:
        return []

    n_ticks = len(raw) // _DUKA_TICK_SIZE
    ticks = []
    base_dt = dt.datetime(day.year, day.month, day.day, hour, 0, 0)

    for i in range(n_ticks):
        offset = i * _DUKA_TICK_SIZE
        ms, ask_raw, bid_raw, ask_vol, bid_vol = struct.unpack(
            "!IIIff", raw[offset : offset + _DUKA_TICK_SIZE]
        )
        tick_time = base_dt + dt.timedelta(milliseconds=ms)
        ask = ask_raw / 100_000
        bid = bid_raw / 100_000
        ticks.append((tick_time, ask, bid, round(ask_vol * 1_000_000), round(bid_vol * 1_000_000)))

    return ticks


def _fetch_hour(symbol: str, day: dt.date, hour: int) -> list[tuple]:
    """Fetch and parse one hour of tick data from Dukascopy."""
    url = _DUKA_URL.format(
        symbol=symbol,
        year=day.year,
        month=day.month - 1,  # Dukascopy months are 0-indexed
        day=day.day,
        hour=hour,
    )

    for attempt in range(_DUKA_MAX_RETRIES):
        try:
            resp = requests.get(url, headers=_DUKA_HEADERS, timeout=_DUKA_TIMEOUT)
            if resp.status_code == 200 and len(resp.content) > 0:
                decompressed = _decompress_lzma(resp.content)
                return _parse_bi5_ticks(day, hour, decompressed)
            elif resp.status_code == 404:
                return []  # No data for this hour (weekend, holiday)
            else:
                logger.debug("Dukascopy %d for %s hour %d (attempt %d)", resp.status_code, day, hour, attempt + 1)
        except (requests.RequestException, LZMAError) as e:
            logger.debug("Dukascopy fetch error %s hour %d: %s (attempt %d)", day, hour, e, attempt + 1)
            if attempt < _DUKA_MAX_RETRIES - 1:
                time.sleep(0.5 * (attempt + 1))

    return []


def _fetch_day_ticks(symbol: str, day: dt.date) -> list[tuple]:
    """Fetch all 24 hours of tick data for one day, in parallel."""
    all_ticks = []

    with ThreadPoolExecutor(max_workers=_DUKA_THREADS) as pool:
        futures = {pool.submit(_fetch_hour, symbol, day, h): h for h in range(24)}
        results = {}
        for future in as_completed(futures):
            h = futures[future]
            try:
                results[h] = future.result()
            except Exception as e:
                logger.debug("Hour %d failed: %s", h, e)
                results[h] = []

    # Reassemble in order
    for h in range(24):
        all_ticks.extend(results.get(h, []))

    return all_ticks


def _ticks_to_m1(ticks: list[tuple]) -> pd.DataFrame:
    """Aggregate tick data into M1 OHLCV candles."""
    if not ticks:
        return pd.DataFrame(columns=["Open", "High", "Low", "Close", "Volume"])

    df = pd.DataFrame(ticks, columns=["datetime", "ask", "bid", "ask_vol", "bid_vol"])
    # Use mid-price for OHLC
    df["price"] = (df["ask"] + df["bid"]) / 2
    df["volume"] = df["ask_vol"] + df["bid_vol"]
    df["datetime"] = pd.to_datetime(df["datetime"], utc=True)
    df.set_index("datetime", inplace=True)

    m1 = df["price"].resample("1min").agg(
        Open="first", High="max", Low="min", Close="last"
    )
    m1["Volume"] = df["volume"].resample("1min").sum()
    m1.dropna(subset=["Open"], inplace=True)

    return m1


# ──────────────────────────────────────────────
# Dukascopy symbol mapping
# ──────────────────────────────────────────────
def _dukascopy_symbol(symbol: str) -> str:
    """
    Convert IC Markets symbol to Dukascopy format.
    Dukascopy uses uppercase without slashes (e.g. EURUSD).
    Most Forex/CFD symbols are already in this format.
    """
    return symbol.upper().replace("/", "").replace("=X", "")


# ──────────────────────────────────────────────
# Local Parquet Cache
# ──────────────────────────────────────────────
def _cache_path(symbol: str, year: int, month: int) -> Path:
    """Return the parquet cache path for a symbol/year/month."""
    sym_dir = DATA_HISTORY_DIR / symbol.upper()
    sym_dir.mkdir(exist_ok=True)
    return sym_dir / f"{symbol}_{year}_{month:02d}_M1.parquet"


def _load_cached_month(symbol: str, year: int, month: int) -> Optional[pd.DataFrame]:
    """Load cached M1 data for a given month, or None if not cached."""
    path = _cache_path(symbol, year, month)
    if path.exists():
        try:
            df = pd.read_parquet(path)
            if len(df) > 0:
                return df
        except Exception as e:
            logger.warning("Corrupt cache file %s: %s — will re-download", path, e)
    return None


def _save_cached_month(symbol: str, year: int, month: int, df: pd.DataFrame):
    """Save M1 data to parquet cache."""
    if df is None or len(df) == 0:
        return
    path = _cache_path(symbol, year, month)
    df.to_parquet(path, engine="pyarrow")
    logger.info("Cached %d M1 bars -> %s", len(df), path.name)


def _is_month_complete(year: int, month: int) -> bool:
    """Check if a month is fully in the past (safe to cache permanently)."""
    now = dt.datetime.utcnow()
    if year < now.year:
        return True
    if year == now.year and month < now.month:
        return True
    return False


def _month_range(start: dt.date, end: dt.date) -> list[tuple[int, int]]:
    """Generate list of (year, month) tuples covering the date range."""
    months = []
    current = dt.date(start.year, start.month, 1)
    while current <= end:
        months.append((current.year, current.month))
        if current.month == 12:
            current = dt.date(current.year + 1, 1, 1)
        else:
            current = dt.date(current.year, current.month + 1, 1)
    return months


# ──────────────────────────────────────────────
# Main Dukascopy Fetcher
# ──────────────────────────────────────────────
def fetch_dukascopy(
    symbol: str,
    timeframe: str = "H1",
    start: Optional[str] = None,
    end: Optional[str] = None,
    days_back: int = 720,
) -> pd.DataFrame:
    """
    Fetch OHLCV data from Dukascopy with local parquet caching.

    Downloads M1 tick data, caches it month-by-month as parquet, then
    resamples to the requested timeframe.

    Parameters
    ----------
    symbol : str
        Trading pair (e.g. "EURUSD", "GBPJPY").
    timeframe : str
        Target timeframe: M1, M5, M15, M30, H1, H4, D1.
    start, end : str, optional
        Date strings "YYYY-MM-DD". If None, uses days_back from today.
    days_back : int
        How many days of history to fetch (default 720).

    Returns
    -------
    pd.DataFrame
        OHLCV DataFrame indexed by UTC datetime.
    """
    duka_sym = _dukascopy_symbol(symbol)
    resample_rule = _RESAMPLE_RULES.get(timeframe)
    if resample_rule is None:
        raise ValueError(f"Unsupported timeframe: {timeframe}. Use one of {list(_RESAMPLE_RULES.keys())}")

    # Resolve date range
    if start is None:
        start_date = (dt.datetime.utcnow() - dt.timedelta(days=days_back)).date()
    else:
        start_date = dt.datetime.strptime(start, "%Y-%m-%d").date()

    if end is None:
        end_date = dt.datetime.utcnow().date()
    else:
        end_date = dt.datetime.strptime(end, "%Y-%m-%d").date()

    logger.info("Dukascopy: %s %s [%s -> %s]", duka_sym, timeframe, start_date, end_date)

    # Collect M1 data month by month (cached or fresh)
    months = _month_range(start_date, end_date)
    all_m1 = []

    for year, month in months:
        # Check if current month (needs partial re-download)
        is_complete = _is_month_complete(year, month)

        cached = _load_cached_month(duka_sym, year, month)
        if cached is not None and is_complete:
            all_m1.append(cached)
            logger.debug("Cache hit: %s %04d-%02d (%d bars)", duka_sym, year, month, len(cached))
            continue

        # Need to download this month
        import calendar
        _, days_in_month = calendar.monthrange(year, month)
        month_start = dt.date(year, month, 1)
        month_end = dt.date(year, month, days_in_month)

        # Clip to requested range
        fetch_start = max(month_start, start_date)
        fetch_end = min(month_end, end_date)

        # For incomplete months with existing cache, only fetch new days
        if not is_complete and cached is not None and len(cached) > 0:
            last_cached_date = cached.index[-1].date()
            incremental_start = last_cached_date + dt.timedelta(days=1)
            if incremental_start > fetch_end:
                # Cache already covers everything requested
                all_m1.append(cached)
                logger.debug("Cache covers %s %04d-%02d (up to %s)", duka_sym, year, month, last_cached_date)
                continue
            fetch_start = max(fetch_start, incremental_start)
            logger.info("Incremental download %s %04d-%02d [%s -> %s]", duka_sym, year, month, fetch_start, fetch_end)
        else:
            logger.info("Downloading %s %04d-%02d from Dukascopy...", duka_sym, year, month)

        month_ticks = []
        current_day = fetch_start
        while current_day <= fetch_end:
            # Skip weekends (Forex market closed Sat-Sun)
            if current_day.weekday() < 5:  # Mon=0 .. Fri=4
                day_ticks = _fetch_day_ticks(duka_sym, current_day)
                month_ticks.extend(day_ticks)
            current_day += dt.timedelta(days=1)

        m1_month = _ticks_to_m1(month_ticks)

        if len(m1_month) > 0:
            # If the month is complete, merge with any existing cache
            if is_complete:
                if cached is not None:
                    m1_month = pd.concat([cached, m1_month])
                    m1_month = m1_month[~m1_month.index.duplicated(keep="last")]
                    m1_month.sort_index(inplace=True)
                _save_cached_month(duka_sym, year, month, m1_month)
            else:
                # Current month: merge but don't mark as permanent
                if cached is not None:
                    m1_month = pd.concat([cached, m1_month])
                    m1_month = m1_month[~m1_month.index.duplicated(keep="last")]
                    m1_month.sort_index(inplace=True)
                _save_cached_month(duka_sym, year, month, m1_month)

            all_m1.append(m1_month)
        elif cached is not None:
            all_m1.append(cached)

    if not all_m1:
        raise RuntimeError(f"No Dukascopy data for {duka_sym} in [{start_date} -> {end_date}]")

    # Combine all months
    m1_full = pd.concat(all_m1)
    m1_full = m1_full[~m1_full.index.duplicated(keep="last")]
    m1_full.sort_index(inplace=True)

    # Clip to exact requested range
    start_ts = pd.Timestamp(start_date, tz="UTC")
    end_ts = pd.Timestamp(end_date, tz="UTC") + pd.Timedelta(days=1)
    m1_full = m1_full.loc[start_ts:end_ts]

    # Resample to target timeframe
    if timeframe == "M1":
        return m1_full

    return _resample_ohlcv(m1_full, resample_rule)


# ──────────────────────────────────────────────
# OHLCV Resampling
# ──────────────────────────────────────────────
def _resample_ohlcv(df: pd.DataFrame, rule: str) -> pd.DataFrame:
    """Resample OHLCV to a coarser timeframe."""
    return df.resample(rule).agg({
        "Open": "first",
        "High": "max",
        "Low": "min",
        "Close": "last",
        "Volume": "sum",
    }).dropna(subset=["Open"])


# ──────────────────────────────────────────────
# Quote currency helpers
# ──────────────────────────────────────────────
def get_quote_currency(symbol: str) -> str:
    """Extract the quote currency (last 3 chars) from a Forex symbol."""
    s = symbol.upper().replace("/", "").replace("=X", "")
    if len(s) >= 6:
        return s[3:6]
    return "USD"


def fetch_quote_to_usd_rate(
    symbol: str,
    target_index: pd.DatetimeIndex,
    days_back: int = 730,
) -> pd.Series:
    """
    Return a time-varying Series that converts the quote currency of `symbol`
    into USD.  The Series is forward-filled and aligned to `target_index`.

    For USD-quoted pairs (e.g. EURUSD) this returns all 1.0.
    For JPY-quoted pairs (e.g. GBPJPY) this fetches USDJPY and returns 1/rate.
    """
    quote = get_quote_currency(symbol)

    if quote == "USD":
        return pd.Series(1.0, index=target_index, name="fx_rate")

    lookup = _QUOTE_TO_USD.get(quote)
    if lookup is None:
        logger.warning(
            "Unknown quote currency '%s' for symbol '%s' — defaulting fx_rate=1.0. "
            "PnL will be in quote currency, not USD.",
            quote, symbol,
        )
        return pd.Series(1.0, index=target_index, name="fx_rate")

    conv_pair, invert = lookup
    logger.info(
        "Fetching conversion pair %s for %s->USD (invert=%s)",
        conv_pair, quote, invert,
    )

    try:
        conv_df = fetch_dukascopy(conv_pair, timeframe="H1", days_back=days_back)
        conv_rate = conv_df["Close"]
        if invert:
            conv_rate = 1.0 / conv_rate
    except Exception as e:
        logger.warning("H1 fetch failed for %s, trying D1: %s", conv_pair, e)
        try:
            conv_df = fetch_dukascopy(conv_pair, timeframe="D1", days_back=days_back)
            conv_rate = conv_df["Close"]
            if invert:
                conv_rate = 1.0 / conv_rate
        except Exception as e2:
            logger.error(
                "Cannot fetch conversion pair %s: %s — using static fallback",
                conv_pair, e2,
            )
            return pd.Series(1.0, index=target_index, name="fx_rate")

    # Align to target index via forward-fill
    rate_aligned = conv_rate.reindex(target_index, method="ffill")
    rate_aligned = rate_aligned.bfill()
    if rate_aligned.isna().any():
        fallback = conv_rate.iloc[-1] if len(conv_rate) > 0 else 1.0
        rate_aligned = rate_aligned.fillna(fallback)

    rate_aligned.name = "fx_rate"
    return rate_aligned


# ──────────────────────────────────────────────
# Unified Historical Fetcher
# ──────────────────────────────────────────────
def fetch_historical(
    symbol: str,
    timeframe: str,
    start: Optional[str] = None,
    end: Optional[str] = None,
    days_back: int = 720,
    prefer_mt5: bool = False,
) -> pd.DataFrame:
    """
    Unified entry point. Tries MT5 first (if prefer_mt5 and connected),
    falls back to Dukascopy for deep history.
    """
    if prefer_mt5 and mt5 is not None:
        try:
            conn = MT5Connection()
            if conn.connected:
                df = conn.fetch_ohlcv(symbol, timeframe, days_back=days_back)
                if df is not None and len(df) > 100:
                    return df
        except Exception as e:
            logger.warning("MT5 fetch failed, falling back to Dukascopy: %s", e)

    return fetch_dukascopy(symbol, timeframe, start=start, end=end, days_back=days_back)


def fetch_multi_timeframe(
    symbol: str,
    timeframes: list[str] = ("D1", "H4", "H1", "M5"),
    days_back: int = 720,
    prefer_mt5: bool = False,
) -> dict[str, pd.DataFrame]:
    """Fetch multiple timeframes for one symbol. Returns dict keyed by TF."""
    data = {}
    for tf in timeframes:
        try:
            data[tf] = fetch_historical(
                symbol, tf, days_back=days_back, prefer_mt5=prefer_mt5,
            )
            logger.info("%s %s: %d bars", symbol, tf, len(data[tf]))
        except Exception as e:
            logger.error("Failed %s %s: %s", symbol, tf, e)
    return data


# ──────────────────────────────────────────────
# Live MT5 Connection Class
# ──────────────────────────────────────────────
class MT5Connection:
    """
    Manages a live connection to MetaTrader 5 via the official Python library.
    Broker: ICMarkets (configurable).
    """

    def __init__(
        self,
        login: Optional[int] = None,
        password: Optional[str] = None,
        server: str = "ICMarketsSC-Demo",
        path: Optional[str] = None,
    ):
        if mt5 is None:
            raise ImportError("MetaTrader5 package is required for live trading")
        self.login = login
        self.password = password
        self.server = server
        self.path = path
        self.connected = False
        if login is not None:
            self.connect()

    def connect(self) -> bool:
        kwargs = {"login": self.login, "password": self.password, "server": self.server}
        if self.path:
            kwargs["path"] = self.path
        if not mt5.initialize(**kwargs):
            err = mt5.last_error()
            logger.error("MT5 init failed: %s", err)
            self.connected = False
            return False
        self.connected = True
        info = mt5.account_info()
        logger.info(
            "MT5 connected: %s | Balance: %.2f %s",
            info.name, info.balance, info.currency,
        )
        return True

    def disconnect(self):
        if self.connected:
            mt5.shutdown()
            self.connected = False

    def account_info(self) -> dict:
        info = mt5.account_info()
        if info is None:
            return {}
        return info._asdict()

    def fetch_ohlcv(
        self,
        symbol: str,
        timeframe: str = "H1",
        bars: int = 10_000,
        days_back: Optional[int] = None,
    ) -> pd.DataFrame:
        """Fetch OHLCV from MT5. Returns UTC-indexed DataFrame."""
        tf = MT5_TF.get(timeframe)
        if tf is None:
            raise ValueError(f"Unknown MT5 timeframe: {timeframe}")

        if days_back:
            utc_from = dt.datetime.utcnow() - dt.timedelta(days=days_back)
            rates = mt5.copy_rates_from(symbol, tf, utc_from, bars)
        else:
            rates = mt5.copy_rates_from_pos(symbol, tf, 0, bars)

        if rates is None or len(rates) == 0:
            logger.warning("No MT5 data for %s %s", symbol, timeframe)
            return pd.DataFrame()

        df = pd.DataFrame(rates)
        df["datetime"] = pd.to_datetime(df["time"], unit="s", utc=True)
        df.set_index("datetime", inplace=True)
        df.rename(columns={
            "open": "Open", "high": "High", "low": "Low",
            "close": "Close", "tick_volume": "Volume",
        }, inplace=True)
        return df[["Open", "High", "Low", "Close", "Volume"]]

    def fetch_tick(self, symbol: str) -> Optional[dict]:
        """Get latest tick for a symbol."""
        tick = mt5.symbol_info_tick(symbol)
        if tick is None:
            return None
        return tick._asdict()

    def symbol_info(self, symbol: str) -> Optional[dict]:
        """Return symbol specification (pip value, digits, etc)."""
        info = mt5.symbol_info(symbol)
        if info is None:
            return None
        return info._asdict()

    # ── Order Execution ──────────────────────
    def send_order(
        self,
        symbol: str,
        order_type: str,
        volume: float,
        price: Optional[float] = None,
        sl: Optional[float] = None,
        tp: Optional[float] = None,
        comment: str = "RegimeBot",
        magic: int = 234000,
    ) -> dict:
        """
        Send a market order to MT5.
        order_type: 'BUY' or 'SELL'
        """
        sym_info = mt5.symbol_info(symbol)
        if sym_info is None:
            raise RuntimeError(f"Symbol {symbol} not found in MT5")
        if not sym_info.visible:
            mt5.symbol_select(symbol, True)

        tick = mt5.symbol_info_tick(symbol)
        if order_type.upper() == "BUY":
            mt5_type = mt5.ORDER_TYPE_BUY
            exec_price = tick.ask if price is None else price
        else:
            mt5_type = mt5.ORDER_TYPE_SELL
            exec_price = tick.bid if price is None else price

        request = {
            "action": mt5.TRADE_ACTION_DEAL,
            "symbol": symbol,
            "volume": round(volume, 2),
            "type": mt5_type,
            "price": exec_price,
            "sl": sl or 0.0,
            "tp": tp or 0.0,
            "deviation": 20,
            "magic": magic,
            "comment": comment,
            "type_time": mt5.ORDER_TIME_GTC,
            "type_filling": mt5.ORDER_FILLING_IOC,
        }
        result = mt5.order_send(request)
        if result is None:
            raise RuntimeError(f"Order send returned None: {mt5.last_error()}")
        res = result._asdict()
        if result.retcode != mt5.TRADE_RETCODE_DONE:
            logger.error("Order failed: %s (code %d)", result.comment, result.retcode)
        else:
            logger.info(
                "Order OK: %s %s %.2f lots @ %.5f | SL=%.5f TP=%.5f",
                order_type, symbol, volume, exec_price, sl or 0, tp or 0,
            )
        return res

    def close_position(self, ticket: int, symbol: str, volume: float, position_type: int) -> dict:
        """Close an open position by ticket."""
        tick = mt5.symbol_info_tick(symbol)
        if position_type == mt5.ORDER_TYPE_BUY:
            close_type = mt5.ORDER_TYPE_SELL
            price = tick.bid
        else:
            close_type = mt5.ORDER_TYPE_BUY
            price = tick.ask

        request = {
            "action": mt5.TRADE_ACTION_DEAL,
            "symbol": symbol,
            "volume": round(volume, 2),
            "type": close_type,
            "position": ticket,
            "price": price,
            "deviation": 20,
            "magic": 234000,
            "comment": "RegimeBot_Close",
            "type_time": mt5.ORDER_TIME_GTC,
            "type_filling": mt5.ORDER_FILLING_IOC,
        }
        result = mt5.order_send(request)
        return result._asdict() if result else {}

    def get_positions(self, symbol: Optional[str] = None, magic: int = 234000) -> list[dict]:
        """Get open positions filtered by symbol and magic number."""
        if symbol:
            positions = mt5.positions_get(symbol=symbol)
        else:
            positions = mt5.positions_get()
        if positions is None:
            return []
        return [p._asdict() for p in positions if p.magic == magic]
