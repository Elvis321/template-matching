import json
import logging
import pandas as pd
from pathlib import Path
import xgboost as xgb
from config import config

logger = logging.getLogger("online_learner")

MODEL_PARAMS = dict(
    n_estimators=200, max_depth=4, learning_rate=0.05,
    subsample=0.8, colsample_bytree=0.8, eval_metric="logloss", random_state=42,
)


class OnlineZoneLearner:
    def __init__(self, model_path, feature_cols_path, trade_log_path, retrain_every=25):
        self.model_path = Path(model_path)
        self.feature_cols_path = Path(feature_cols_path)
        self.trade_log_path = Path(trade_log_path)
        self.retrain_every = retrain_every

        with open(self.feature_cols_path) as f:
            self.feature_cols = json.load(f)

        self.model = xgb.XGBClassifier()
        self.model.load_model(str(self.model_path))
        logger.info("Loaded model from %s with %d features", self.model_path, len(self.feature_cols))

        self.trade_log_path.parent.mkdir(parents=True, exist_ok=True)
        if self.trade_log_path.exists():
            self.history_df = pd.read_csv(self.trade_log_path)
            logger.info("Loaded %d historical live trades from %s", len(self.history_df), self.trade_log_path)

            # --- migration: older trade logs predate multi-symbol support and
            # don't have a "symbol" column. Backfill it so downstream filtering
            # doesn't KeyError. We can't know which symbol these old rows were
            # actually trading, so tag them "UNKNOWN" -- they'll still count
            # toward overall accuracy but won't be attributed to any specific
            # symbol in the per-symbol breakdown.
            if "symbol" not in self.history_df.columns:
                logger.warning(
                    "trade_log.csv predates multi-symbol support -- backfilling "
                    "'symbol' column with 'UNKNOWN' for %d existing rows.",
                    len(self.history_df),
                )
                self.history_df["symbol"] = "UNKNOWN"
                self._save_log()
        else:
            cols = self.feature_cols + [
                "symbol", "timestamp", "ticket", "direction", "entry_price",
                "predicted_proba_up", "predicted_class", "status",
                "exit_price", "outcome", "correct",
            ]
            self.history_df = pd.DataFrame(columns=cols)
            self._save_log()

    def _save_log(self):
        self.history_df.to_csv(self.trade_log_path, index=False)

    def predict(self, features_dict):
        X = pd.DataFrame([features_dict])[self.feature_cols]
        proba_up = float(self.model.predict_proba(X)[0, 1])
        pred_class = 1 if proba_up > 0.5 else 0
        return proba_up, pred_class

    def log_open_trade(self, symbol, features_dict, timestamp, ticket, direction, entry_price,
                        proba_up, pred_class):
        row = dict(features_dict)
        row.update({
            "symbol": symbol, "timestamp": timestamp, "ticket": ticket, "direction": direction,
            "entry_price": entry_price, "predicted_proba_up": proba_up,
            "predicted_class": pred_class, "status": "open",
            "exit_price": None, "outcome": None, "correct": None,
        })
        self.history_df = pd.concat([self.history_df, pd.DataFrame([row])], ignore_index=True)
        self._save_log()
        return len(self.history_df) - 1

    def record_outcome(self, row_index, exit_price, outcome):
        self.history_df.loc[row_index, "status"] = "closed"
        self.history_df.loc[row_index, "exit_price"] = exit_price
        self.history_df.loc[row_index, "outcome"] = outcome
        predicted_class = self.history_df.loc[row_index, "predicted_class"]
        self.history_df.loc[row_index, "correct"] = int(predicted_class == outcome)
        self._save_log()

        n_closed = (self.history_df["status"] == "closed").sum()
        logger.info("Outcome recorded (row %d). Total closed trades: %d", row_index, n_closed)

        if config.ENABLE_RETRAINING and n_closed > 0 and n_closed % self.retrain_every == 0:
            self._retrain()

    def _retrain(self):
        closed = self.history_df[self.history_df["status"] == "closed"].copy()
        if len(closed) < 30:
            logger.info("Not enough closed trades yet to retrain (%d)", len(closed))
            return

        X = closed[self.feature_cols].astype(float)
        y = closed["outcome"].astype(int)

        new_model = xgb.XGBClassifier(**MODEL_PARAMS)
        new_model.fit(X, y)
        self.model = new_model
        self.model.save_model(str(self.model_path))
        logger.info("Retrained model on %d live-accumulated trades and saved to %s",
                     len(closed), self.model_path)

    def open_trades(self):
        return self.history_df[self.history_df["status"] == "open"]

    def running_accuracy(self, window=50, symbol=None):
        closed = self.history_df[self.history_df["status"] == "closed"]
        # defensive check in case an even older log format sneaks through some other path
        if symbol is not None and "symbol" in closed.columns:
            closed = closed[closed["symbol"] == symbol]
        if len(closed) == 0:
            return None
        recent = closed.tail(window)
        return float(recent["correct"].astype(float).mean())