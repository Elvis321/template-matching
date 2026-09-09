import os
from dotenv import load_dotenv

load_dotenv()


def _bool(val, default=False):
    if val is None:
        return default
    return val.strip().lower() in ("1", "true", "yes", "on")


class Config:
    # Market data
    SYMBOL = os.getenv("SYMBOL", "EURUSD=X")
    INTERVAL = os.getenv("INTERVAL", "5m")
    BARS_LOOKBACK = int(os.getenv("BARS_LOOKBACK", "200"))

    # Virtual account
    STARTING_BALANCE = float(os.getenv("STARTING_BALANCE", "10000"))
    POSITION_SIZE_USD = float(os.getenv("POSITION_SIZE_USD", "1000"))
    SPREAD_PCT = float(os.getenv("SPREAD_PCT", "0.02"))
    SLIPPAGE_PCT = float(os.getenv("SLIPPAGE_PCT", "0.01"))

    # Feature extraction params -- MUST match training
    LEG_IN_LOOKBACK = int(os.getenv("LEG_IN_LOOKBACK", "10"))
    CONTEXT_LOOKBACK = int(os.getenv("CONTEXT_LOOKBACK", "50"))
    BASE_LEN_MIN = int(os.getenv("BASE_LEN_MIN", "3"))
    BASE_LEN_MAX = int(os.getenv("BASE_LEN_MAX", "10"))
    ATR_WINDOW = int(os.getenv("ATR_WINDOW", "14"))
    CONTRACTION_RATIO = float(os.getenv("CONTRACTION_RATIO", "0.5"))
    DEDUP_GAP = int(os.getenv("DEDUP_GAP", "5"))

    # Strategy / trading
    CONFIDENCE_THRESHOLD = float(os.getenv("CONFIDENCE_THRESHOLD", "0.65"))
    HORIZON_BARS = int(os.getenv("HORIZON_BARS", "10"))
    BREAKOUT_THRESHOLD_PCT = float(os.getenv("BREAKOUT_THRESHOLD_PCT", "0.15"))

    # Loop behavior
    POLL_INTERVAL_SECONDS = int(os.getenv("POLL_INTERVAL_SECONDS", "60"))
    RETRAIN_EVERY_N_OUTCOMES = int(os.getenv("RETRAIN_EVERY_N_OUTCOMES", "25"))
    ENABLE_RETRAINING = _bool(os.getenv("ENABLE_RETRAINING"), default=False)

    # Paths
    MODEL_PATH = os.getenv("MODEL_PATH", "models/model.joblib")
    FEATURE_COLS_PATH = os.getenv("FEATURE_COLS_PATH", "models/feature_cols.json")
    TRADE_LOG_PATH = os.getenv("TRADE_LOG_PATH", "logs/trade_log.csv")
    EQUITY_LOG_PATH = os.getenv("EQUITY_LOG_PATH", "logs/equity_log.csv")


config = Config()