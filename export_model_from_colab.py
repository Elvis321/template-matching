import joblib
import json
import xgboost as xgb

# retrain on the FULL dataset before shipping
final_model = xgb.XGBClassifier(
    n_estimators=200, max_depth=4, learning_rate=0.05,
    subsample=0.8, colsample_bytree=0.8, eval_metric="logloss", random_state=42,
)
final_model.fit(dataset_v2_sorted[feature_cols_v2], dataset_v2_sorted["outcome"])

joblib.dump(final_model, "model.joblib")
with open("feature_cols.json", "w") as f:
    json.dump(feature_cols_v2, f)

print("Exported model.joblib and feature_cols.json")

from google.colab import files
files.download("model.joblib")
files.download("feature_cols.json")