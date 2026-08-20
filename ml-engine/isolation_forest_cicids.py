import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.metrics import classification_report
import joblib, os

TRAIN_PATH = "data/processed/cic-ids-2017_train.parquet"
VAL_PATH = "data/processed/cic-ids-2017_val.parquet"
MODEL_OUT = "ml-engine/models/isolation_forest_cicids.joblib"

def main():
    train_df = pd.read_parquet(TRAIN_PATH)
    benign_only = train_df[train_df["Attack"] == 0].drop(columns=["Attack"])
    print(f"Training on {benign_only.shape[0]} benign-only rows")

    val_df = pd.read_parquet(VAL_PATH)
    y_val = val_df["Attack"]
    X_val = val_df.drop(columns=["Attack"])
    real_rate = y_val.mean()
    print(f"Real attack rate in val: {real_rate:.3f}")

    model = IsolationForest(n_estimators=200, contamination=real_rate, random_state=42, n_jobs=-1)
    model.fit(benign_only)

    preds = (model.predict(X_val) == -1).astype(int)
    print(classification_report(y_val, preds, target_names=["benign", "attack"]))

    os.makedirs(os.path.dirname(MODEL_OUT), exist_ok=True)
    joblib.dump(model, MODEL_OUT)
    print(f"Saved to {MODEL_OUT}")

if __name__ == "__main__":
    main()
