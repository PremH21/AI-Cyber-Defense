import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.metrics import f1_score, classification_report
import joblib
import os

TRAIN_PATH = "data/processed/cic-ids-2017_train.parquet"
VAL_PATH = "data/processed/cic-ids-2017_val.parquet"
MODEL_OUT = "ml-engine/models/xgboost_cicids_online_replay.joblib"

REPLAY_BUFFER_SIZE = 20000  # max samples kept in the replay buffer

def main():
    print("Loading data...")
    train_df = pd.read_parquet(TRAIN_PATH)
    val_df = pd.read_parquet(VAL_PATH)

    y_train = train_df["Attack"].values
    X_train = train_df.drop(columns=["Attack"])
    y_val = val_df["Attack"].values
    X_val = val_df.drop(columns=["Attack"])

    n_batches = 5
    batch_size = len(X_train) // n_batches
    print(f"Simulating {n_batches} incident batches of ~{batch_size} samples each")
    print(f"Experience replay buffer size: {REPLAY_BUFFER_SIZE}\n")

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
    rng = np.random.default_rng(42)

    replay_X = []
    replay_y = []

    for b in range(n_batches):
        start = b * batch_size
        end = len(X_train) if b == n_batches - 1 else (b + 1) * batch_size
        X_batch = X_train.iloc[start:end].reset_index(drop=True)
        y_batch = y_train[start:end]

        if replay_X:
            replay_X_concat = pd.concat(replay_X, ignore_index=True)
            replay_y_concat = np.concatenate(replay_y)
            X_combined = pd.concat([replay_X_concat, X_batch], ignore_index=True)
            y_combined = np.concatenate([replay_y_concat, y_batch])
            print(f"Batch {b+1}/{n_batches}: {len(X_batch)} new samples + {len(replay_X_concat)} replayed samples")
        else:
            X_combined = X_batch
            y_combined = y_batch
            print(f"Batch {b+1}/{n_batches}: initial training on {len(X_batch)} samples (no replay yet)")

        dtrain = xgb.DMatrix(X_combined, label=y_combined)

        if model is None:
            model = xgb.train(params, dtrain, num_boost_round=50)
        else:
            model = xgb.train(params, dtrain, num_boost_round=20, xgb_model=model)

        # Update replay buffer: randomly sample from this batch to keep for future rounds
        sample_n = min(REPLAY_BUFFER_SIZE // n_batches, len(X_batch))
        sample_idx = rng.choice(len(X_batch), size=sample_n, replace=False)
        replay_X.append(X_batch.iloc[sample_idx].reset_index(drop=True))
        replay_y.append(y_batch[sample_idx])

        # Cap total replay buffer size (keep most recent additions if over budget)
        total_replay = sum(len(r) for r in replay_X)
        while total_replay > REPLAY_BUFFER_SIZE and len(replay_X) > 1:
            removed = replay_X.pop(0)
            replay_y.pop(0)
            total_replay -= len(removed)

        dval = xgb.DMatrix(X_val)
        val_preds = (model.predict(dval) >= 0.5).astype(int)
        f1 = f1_score(y_val, val_preds)
        history.append(f1)
        print(f"  -> F1 on held-out validation after batch {b+1}: {f1:.4f}\n")

    print("=== Online learning WITH experience replay: F1 progression ===")
    for i, f1 in enumerate(history):
        print(f"  After batch {i+1}: F1={f1:.4f}")

    print(f"\nChange from peak to final: {history[-1] - max(history):+.4f}")
    print(f"Change from batch 1 to final: {history[-1] - history[0]:+.4f}")

    print("\nFinal evaluation on validation set:")
    dval = xgb.DMatrix(X_val)
    final_preds = (model.predict(dval) >= 0.5).astype(int)
    print(classification_report(y_val, final_preds, target_names=["benign", "attack"]))

    os.makedirs(os.path.dirname(MODEL_OUT), exist_ok=True)
    joblib.dump(model, MODEL_OUT)
    print(f"Final model saved to {MODEL_OUT}")

if __name__ == "__main__":
    main()
