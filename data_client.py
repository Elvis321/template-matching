import logging
import numpy as np
import pandas as pd
import yfinance as yf
from config import config

logger = logging.getLogger("data_client")


class StaleDataError(Exception):
    pass


def get_recent_bars(symbol=None, interval=None, n_bars=None, max_flat_run=8):
    """
    Fetch recent bars and run a basic sanity check for stale/duplicated data
    (the same issue that produced the ~95% same-session-revisit artifact earlier).
    Raises StaleDataError if too many consecutive bars show zero price change.
    """
    symbol = symbol or config.SYMBOL
    interval = interval or config.INTERVAL
    n_bars = n_bars or config.BARS_LOOKBACK

    period = _period_for_interval(interval)
    df = yf.download(symbol, period=period, interval=interval,
                      auto_adjust=True, progress=False)
    if df is None or len(df) == 0:
        raise RuntimeError(f"No data returned for {symbol} @ {interval}")

    df = df.dropna()
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = [c[0] for c in df.columns]

    df = df.tail(n_bars).copy()

    closes = df["Close"].values
    unchanged = (np.diff(closes) == 0).astype(int)
    # longest run of consecutive zero-change bars at the END of the series
    tail_run = 0
    for v in unchanged[::-1]:
        if v == 1:
            tail_run += 1
        else:
            break
    if tail_run >= max_flat_run:
        raise StaleDataError(
            f"Last {tail_run} bars for {symbol} show zero price change -- "
            f"data feed likely stale. Skipping this cycle."
        )

    df = df.reset_index()
    time_col = "Datetime" if "Datetime" in df.columns else df.columns[0]
    df = df.rename(columns={
        time_col: "time", "Open": "open", "High": "high",
        "Low": "low", "Close": "close", "Volume": "volume",
    })
    return df[["time", "open", "high", "low", "close", "volume"]]


def get_current_price(symbol=None):
    symbol = symbol or config.SYMBOL
    bars = get_recent_bars(symbol=symbol, n_bars=2)
    return float(bars["close"].iloc[-1])


def _period_for_interval(interval):
    # yfinance caps how far back intraday intervals can go
    mapping = {
        "1m": "5d", "2m": "60d", "5m": "60d", "15m": "60d",
        "30m": "60d", "60m": "730d", "1h": "730d", "1d": "5y",
    }
    return mapping.get(interval, "60d")