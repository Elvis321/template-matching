"""
Same OnlineZoneLearner concept validated in the research notebook, adapted to:
- load a pretrained model from disk on startup instead of training from scratch
- persist every prediction + outcome to a CSV log (survives restarts)
- reload accumulated live history on startup so retraining picks up where it left off

Uses XGBoost's native save_model/load_model (JSON format) instead of joblib/pickle --
portable across XGBoost versions and platforms, avoiding "input stream corrupted"
errors when the training environment (Colab) and deployment environment (e.g. Windows)
have different XGBoost versions installed.
"""
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
        else:
            cols = self.feature_cols + [
                "timestamp", "ticket", "direction", "entry_price",
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

    def log_open_trade(self, features_dict, timestamp, ticket, direction, entry_price,
                        proba_up, pred_class):
        row = dict(features_dict)
        row.update({
            "timestamp": timestamp, "ticket": ticket, "direction": direction,
            "entry_price": entry_price, "predicted_proba_up": proba_up,
            "predicted_class": pred_class, "status": "open",
            "exit_price": None, "outcome": None, "correct": None,
        })
        self.history_df = pd.concat([self.history_df, pd.DataFrame([row])], ignore_index=True)
        self._save_log()
        return len(self.history_df) - 1  # row index for later update

    def record_outcome(self, row_index, exit_price, outcome):
        """outcome: 1 if price broke up, 0 if it broke down (same convention as training)."""
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

    def running_accuracy(self, window=50):
        closed = self.history_df[self.history_df["status"] == "closed"]
        if len(closed) == 0:
            return None
        recent = closed.tail(window)
        return float(recent["correct"].astype(float).mean())