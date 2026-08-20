import os
import json
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import classification_report, f1_score
from imblearn.over_sampling import SMOTE
import joblib

BASE_DIR = os.path.expanduser("~/ai-cyber-defense")
DATA_DIR = os.path.join(BASE_DIR, "data", "processed")
MODEL_DIR = os.path.join(BASE_DIR, "ml-engine", "models")
os.makedirs(MODEL_DIR, exist_ok=True)


def load_split(name):
    df = pd.read_parquet(os.path.join(DATA_DIR, f"unsw-nb15_multiclass_{name}.parquet"))
    y = df["AttackCat"].values
    X = df.drop(columns=["AttackCat"]).values
    return X, y


def main():
    with open(os.path.join(DATA_DIR, "unsw-nb15_multiclass_labels.json")) as f:
        class_names = json.load(f)
    print(f"Classes ({len(class_names)}): {class_names}\n")

    X_train, y_train = load_split("train")
    X_val, y_val = load_split("val")
    X_test, y_test = load_split("test")

    print("Before targeted SMOTE:")
    counts_before = np.bincount(y_train)
    for i, c in enumerate(class_names):
        print(f"  {c}: {counts_before[i]}")

    # Only boost classes with fewer than 2000 samples, up to 2000.
    # Leave well-represented classes untouched.
    target_floor = 2000
    sampling_strategy = {
        i: target_floor
        for i, count in enumerate(counts_before)
        if count < target_floor
    }
    print(f"\nBoosting {len(sampling_strategy)} minority classes to {target_floor} samples each...")

    sm = SMOTE(sampling_strategy=sampling_strategy, random_state=42, k_neighbors=3)
    X_train_bal, y_train_bal = sm.fit_resample(X_train, y_train)

    print("\nAfter targeted SMOTE:")
    counts_after = np.bincount(y_train_bal)
    for i, c in enumerate(class_names):
        print(f"  {c}: {counts_after[i]}")

    print("\n=== Training XGBoost on rebalanced data ===")
    dtrain = xgb.DMatrix(X_train_bal, label=y_train_bal)
    dval = xgb.DMatrix(X_val, label=y_val)
    dtest = xgb.DMatrix(X_test, label=y_test)

    params = {
        "objective": "multi:softmax",
        "num_class": len(class_names),
        "max_depth": 8,
        "eta": 0.1,
        "eval_metric": "mlogloss",
    }
    xgb_model = xgb.train(
        params, dtrain, num_boost_round=200,
        evals=[(dval, "val")], early_stopping_rounds=15, verbose_eval=20,
    )

    print("\n=== FINAL TEST SET EVALUATION (targeted-SMOTE XGBoost) ===")
    test_pred = xgb_model.predict(dtest).astype(int)
    print(classification_report(y_test, test_pred, target_names=class_names, zero_division=0))
    macro_f1 = f1_score(y_test, test_pred, average="macro")
    print(f"Macro-F1: {macro_f1:.4f}")

    joblib.dump(xgb_model, os.path.join(MODEL_DIR, "xgb_multiclass_unsw_v2.joblib"))
    print("\nSaved to ml-engine/models/xgb_multiclass_unsw_v2.joblib")


if __name__ == "__main__":
    main()
