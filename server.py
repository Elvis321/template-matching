import asyncio
import json
import logging
from pathlib import Path
from contextlib import asynccontextmanager

import xgboost as xgb
import pandas as pd
from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from typing import Optional

from config import config
from data_client import get_recent_bars, StaleDataError
from feature_engineering import get_most_recent_base
from trading_loop import TradingEngine, run_forever

logger = logging.getLogger("server")

engine: Optional[TradingEngine] = None
_loop_task: Optional[asyncio.Task] = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global engine, _loop_task
    engine = TradingEngine()
    _loop_task = asyncio.create_task(run_forever(engine))
    logger.info("Background trading loop started.")
    yield
    if _loop_task is not None:
        _loop_task.cancel()


app = FastAPI(title="Zone Trader — Paper Trading", lifespan=lifespan)


@app.get("/status")
def status():
    log_path = Path(config.TRADE_LOG_PATH)
    if not log_path.exists() or engine is None:
        return {"status": "starting up"}

    df = pd.read_csv(log_path)
    closed = df[df["status"] == "closed"]
    open_trades = df[df["status"] == "open"]

    current_prices = {}
    for symbol in config.SYMBOLS:
        try:
            bars = get_recent_bars(symbol=symbol, n_bars=2)
            current_prices[symbol] = float(bars["close"].iloc[-1])
        except Exception:
            pass

    per_symbol = {}
    for symbol in config.SYMBOLS:
        sym_closed = closed[closed.get("symbol") == symbol] if "symbol" in closed.columns else closed.iloc[0:0]
        per_symbol[symbol] = {
            "current_price": current_prices.get(symbol),
            "has_open_position": engine.broker.has_open_position(symbol),
            "closed_trades": len(sym_closed),
            "accuracy": float(sym_closed["correct"].astype(float).mean()) if len(sym_closed) > 0 else None,
        }

    return {
        "symbols": config.SYMBOLS,
        "interval": config.INTERVAL,
        "paused": engine.paused,
        "confidence_threshold": config.CONFIDENCE_THRESHOLD,
        "retraining_enabled": config.ENABLE_RETRAINING,
        "starting_balance": config.STARTING_BALANCE,
        "current_balance": engine.broker.balance,
        "current_equity": engine.broker.equity(current_prices),
        "open_positions": list(engine.broker.open_positions.keys()),
        "total_trades": len(df),
        "open_trades": len(open_trades),
        "closed_trades": len(closed),
        "overall_accuracy": float(closed["correct"].astype(float).mean()) if len(closed) > 0 else None,
        "recent_accuracy_last_50": (
            float(closed.tail(50)["correct"].astype(float).mean()) if len(closed) > 0 else None
        ),
        "per_symbol": per_symbol,
    }


@app.get("/history")
def history(limit: int = 50, symbol: str = None):
    log_path = Path(config.TRADE_LOG_PATH)
    if not log_path.exists():
        return {"trades": []}
    df = pd.read_csv(log_path)
    if symbol and "symbol" in df.columns:
        df = df[df["symbol"] == symbol]
    cols = ["symbol", "timestamp", "ticket", "direction", "entry_price", "exit_price",
            "predicted_proba_up", "predicted_class", "outcome", "correct", "status"]
    cols = [c for c in cols if c in df.columns]
    return {"trades": df[cols].tail(limit).to_dict(orient="records")}


@app.get("/equity_curve")
def equity_curve(limit: int = 500):
    path = Path(config.EQUITY_LOG_PATH)
    if not path.exists():
        return {"points": []}
    df = pd.read_csv(path)
    return {"points": df.tail(limit).to_dict(orient="records")}


class PredictionResponse(BaseModel):
    symbol: Optional[str] = None
    setup_found: bool
    leg_in_direction: Optional[str] = None
    predicted_direction: Optional[str] = None
    proba_up: Optional[float] = None
    confidence: Optional[float] = None
    would_trade: Optional[bool] = None


@app.get("/predict_now", response_model=PredictionResponse)
def predict_now(symbol: str = None):
    symbol = symbol or config.SYMBOLS[0]
    n_bars_needed = max(
        config.BARS_LOOKBACK,
        config.CONTEXT_LOOKBACK + config.LEG_IN_LOOKBACK + config.BASE_LEN_MAX + 10,
    )
    try:
        bars = get_recent_bars(symbol=symbol, n_bars=n_bars_needed)
    except StaleDataError:
        return PredictionResponse(symbol=symbol, setup_found=False)

    close = bars["close"].values
    high = bars["high"].values
    low = bars["low"].values
    volume = bars["volume"].values

    setup = get_most_recent_base(close, high, low, volume, recency_bars=3)
    if setup is None:
        return PredictionResponse(symbol=symbol, setup_found=False)

    with open(config.FEATURE_COLS_PATH) as f:
        feature_cols = json.load(f)
    model = xgb.XGBClassifier()
    model.load_model(config.MODEL_PATH)

    X = pd.DataFrame([setup["features"]])[feature_cols]
    proba_up = float(model.predict_proba(X)[0, 1])
    confidence = max(proba_up, 1 - proba_up)

    return PredictionResponse(
        symbol=symbol,
        setup_found=True,
        leg_in_direction="up" if setup["features"]["leg_in_direction"] == 1 else "down",
        predicted_direction="up" if proba_up > 0.5 else "down",
        proba_up=proba_up,
        confidence=confidence,
        would_trade=confidence >= config.CONFIDENCE_THRESHOLD,
    )


@app.get("/predict_all")
def predict_all():
    """Convenience endpoint: predict_now for every configured symbol in one call."""
    return {symbol: predict_now(symbol=symbol) for symbol in config.SYMBOLS}


@app.post("/pause")
def pause():
    if engine is None:
        return {"ok": False, "reason": "engine not started"}
    engine.paused = True
    return {"ok": True, "paused": True}


@app.post("/resume")
def resume():
    if engine is None:
        return {"ok": False, "reason": "engine not started"}
    engine.paused = False
    return {"ok": True, "paused": False}


@app.get("/health")
def health():
    return {"ok": True}


# keep this LAST so it doesn't shadow the API routes above
app.mount("/", StaticFiles(directory="static", html=True), name="static")