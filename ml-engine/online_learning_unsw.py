import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.metrics import f1_score, classification_report
import joblib
import os

TRAIN_PATH = "data/processed/unsw-nb15_train.parquet"
VAL_PATH = "data/processed/unsw-nb15_val.parquet"
MODEL_OUT = "ml-engine/models/xgboost_unsw_online.joblib"

def main():
    print("Loading data...")
    train_df = pd.read_parquet(TRAIN_PATH)
    val_df = pd.read_parquet(VAL_PATH)

    y_train = train_df["Attack"].values
    X_train = train_df.drop(columns=["Attack"])
    y_val = val_df["Attack"].values
    X_val = val_df.drop(columns=["Attack"])

    # Simulate "incidents arriving over time" by splitting training data into 5 sequential batches
    n_batches = 5
    batch_size = len(X_train) // n_batches
    print(f"Simulating {n_batches} incident batches of ~{batch_size} samples each\n")

    params = {
        "max_depth": 10,
        "learning_rate": 0.05,
        "subsample": 0.9,
        "colsample_bytree": 0.8,
        "min_child_weight": 3,
        "objective": "binary:logistic",
        "eval_metric": "logloss",
    }

    model = None
    history = []

    for b in range(n_batches):
        start = b * batch_size
        end = len(X_train) if b == n_batches - 1 else (b + 1) * batch_size
        X_batch = X_train.iloc[start:end]
        y_batch = y_train[start:end]

        dtrain = xgb.DMatrix(X_batch, label=y_batch)

        if model is None:
            print(f"Batch {b+1}/{n_batches}: initial training on {len(X_batch)} samples...")
            model = xgb.train(params, dtrain, num_boost_round=50)
        else:
            print(f"Batch {b+1}/{n_batches}: incrementally updating model with {len(X_batch)} new samples (warm-start)...")
            model = xgb.train(params, dtrain, num_boost_round=20, xgb_model=model)

        dval = xgb.DMatrix(X_val)
        val_preds = (model.predict(dval) >= 0.5).astype(int)
        f1 = f1_score(y_val, val_preds)
        history.append(f1)
        print(f"  -> F1 on held-out validation after batch {b+1}: {f1:.4f}\n")

    print("=== Online learning progression (F1 on same validation set after each batch) ===")
    for i, f1 in enumerate(history):
        print(f"  After batch {i+1}: F1={f1:.4f}")

    print(f"\nImprovement from batch 1 to final: {history[-1] - history[0]:+.4f}")

    print("\nFinal evaluation on validation set:")
    dval = xgb.DMatrix(X_val)
    final_preds = (model.predict(dval) >= 0.5).astype(int)
    print(classification_report(y_val, final_preds, target_names=["benign", "attack"]))

    os.makedirs(os.path.dirname(MODEL_OUT), exist_ok=True)
    joblib.dump(model, MODEL_OUT)
    print(f"Final online-updated model saved to {MODEL_OUT}")

if __name__ == "__main__":
    main()
