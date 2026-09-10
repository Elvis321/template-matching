import asyncio
import logging
import pandas as pd
from datetime import datetime, timezone

from config import config
from data_client import get_recent_bars, StaleDataError
from feature_engineering import get_most_recent_base
from paper_broker import PaperBroker
from online_learner import OnlineZoneLearner

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
)
logger = logging.getLogger("trading_loop")


class TradingEngine:
    def __init__(self):
        self.broker = PaperBroker(
            starting_balance=config.STARTING_BALANCE,
            position_size_usd=config.POSITION_SIZE_USD,
            spread_pct=config.SPREAD_PCT,
            slippage_pct=config.SLIPPAGE_PCT,
            equity_log_path=config.EQUITY_LOG_PATH,
        )
        self.learner = OnlineZoneLearner(
            model_path=config.MODEL_PATH,
            feature_cols_path=config.FEATURE_COLS_PATH,
            trade_log_path=config.TRADE_LOG_PATH,
            retrain_every=config.RETRAIN_EVERY_N_OUTCOMES,
        )
        self.paused = False
        self._next_ticket = 1
        self._open_row_index = {}  # symbol -> row index of its open trade in the learner log

    def _resolve_open_trade(self, symbol, bars):
        if not self.broker.has_open_position(symbol):
            return

        current_price = float(bars["close"].iloc[-1])
        position = self.broker.open_positions[symbol]
        entry_price = position["entry_price"]
        direction = position["direction"]
        opened_at = pd.Timestamp(position["opened_at"])

        bars_since_entry = (bars["time"] > opened_at).sum()
        move_pct = (current_price / entry_price - 1) * 100
        broke_up = move_pct >= config.BREAKOUT_THRESHOLD_PCT
        broke_down = move_pct <= -config.BREAKOUT_THRESHOLD_PCT
        timed_out = bars_since_entry >= config.HORIZON_BARS

        if not (broke_up or broke_down or timed_out):
            return

        if broke_up and not broke_down:
            outcome = 1
        elif broke_down and not broke_up:
            outcome = 0
        else:
            outcome = 1 if move_pct >= 0 else 0

        result = self.broker.close_trade(symbol, current_price)

        row_index = self._open_row_index.pop(symbol, None)
        if row_index is not None:
            self.learner.record_outcome(row_index, exit_price=result["exit_price"], outcome=outcome)

        logger.info("[%s] Trade resolved: direction=%s move=%.3f%% outcome=%s pnl=$%.2f balance=$%.2f",
                     symbol, direction, move_pct, outcome, result["pnl_usd"], result["balance"])

    def _check_for_new_setup(self, symbol, bars):
        if self.broker.has_open_position(symbol):
            return

        close = bars["close"].values
        high = bars["high"].values
        low = bars["low"].values
        volume = bars["volume"].values

        setup = get_most_recent_base(close, high, low, volume, recency_bars=3)
        if setup is None:
            logger.info("[%s] No live setup this cycle.", symbol)
            return

        features = setup["features"]
        proba_up, pred_class = self.learner.predict(features)
        confidence = max(proba_up, 1 - proba_up)

        logger.info("[%s] Setup detected: leg_in_dir=%s proba_up=%.3f confidence=%.3f (threshold=%.2f)",
                     symbol, features["leg_in_direction"], proba_up, confidence, config.CONFIDENCE_THRESHOLD)

        if confidence < config.CONFIDENCE_THRESHOLD:
            logger.info("[%s] Confidence below threshold -- not trading this setup.", symbol)
            return

        direction = "buy" if pred_class == 1 else "sell"
        market_price = float(bars["close"].iloc[-1])
        fill_price = self.broker.open_trade(symbol, direction, market_price)

        ticket = self._next_ticket
        self._next_ticket += 1
        timestamp = datetime.now(timezone.utc).isoformat()

        self._open_row_index[symbol] = self.learner.log_open_trade(
            symbol=symbol, features_dict=features, timestamp=timestamp, ticket=ticket,
            direction=direction, entry_price=fill_price,
            proba_up=proba_up, pred_class=pred_class,
        )
        logger.info("[%s] Trade opened: %s @ %.6f (paper ticket=%s)", symbol, direction, fill_price, ticket)

    def run_cycle(self):
        if self.paused:
            return

        n_bars_needed = max(
            config.BARS_LOOKBACK,
            config.CONTEXT_LOOKBACK + config.LEG_IN_LOOKBACK + config.BASE_LEN_MAX + 10,
        )

        current_prices = {}

        for symbol in config.SYMBOLS:
            try:
                bars = get_recent_bars(symbol=symbol, n_bars=n_bars_needed)
            except StaleDataError as e:
                # EXPECTED for EURUSD=X over weekends / low-liquidity forex hours --
                # only worth investigating if it persists on BTC-USD, which trades 24/7
                logger.warning("[%s] %s", symbol, e)
                continue
            except Exception as e:
                logger.error("[%s] Failed to fetch market data: %s", symbol, e)
                continue

            current_prices[symbol] = float(bars["close"].iloc[-1])
            self._resolve_open_trade(symbol, bars)
            self._check_for_new_setup(symbol, bars)

        equity = self.broker.equity(current_prices)
        overall_acc = self.learner.running_accuracy()

        per_symbol_acc = {
            s: self.learner.running_accuracy(symbol=s) for s in config.SYMBOLS
        }
        acc_str = ", ".join(
            f"{s}={a:.3f}" if a is not None else f"{s}=n/a" for s, a in per_symbol_acc.items()
        )

        logger.info("Cycle complete. Equity=$%.2f%s | per-symbol acc: %s",
                     equity, f", overall_acc={overall_acc:.3f}" if overall_acc is not None else "", acc_str)


async def run_forever(engine: TradingEngine):
    logger.info("Starting paper trading loop on %s. Retraining %s.",
                config.SYMBOLS, "ENABLED" if config.ENABLE_RETRAINING else "DISABLED (predict-only)")
    while True:
        try:
            engine.run_cycle()
        except Exception as e:
            logger.exception("Error during loop cycle: %s", e)
        await asyncio.sleep(config.POLL_INTERVAL_SECONDS)


if __name__ == "__main__":
    engine = TradingEngine()
    asyncio.run(run_forever(engine))