import os
import json
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import classification_report, f1_score
from sklearn.ensemble import RandomForestClassifier
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

    print("Loading train/val/test splits...")
    X_train, y_train = load_split("train")
    X_val, y_val = load_split("val")
    X_test, y_test = load_split("test")
    print(f"Train: {X_train.shape}, Val: {X_val.shape}, Test: {X_test.shape}\n")

    # ---- Random Forest (handles class imbalance via class_weight) ----
    print("=== Training Random Forest (multi-class) ===")
    rf = RandomForestClassifier(
        n_estimators=200,
        max_depth=20,
        class_weight="balanced",
        n_jobs=-1,
        random_state=42,
    )
    rf.fit(X_train, y_train)
    rf_val_pred = rf.predict(X_val)
    print("Random Forest — validation performance:")
    print(classification_report(y_val, rf_val_pred, target_names=class_names, zero_division=0))
    joblib.dump(rf, os.path.join(MODEL_DIR, "rf_multiclass_unsw.joblib"))

    # ---- XGBoost (multi:softmax with per-sample weighting) ----
    print("\n=== Training XGBoost (multi-class) ===")
    class_counts = np.bincount(y_train)
    class_weights = class_counts.sum() / (len(class_counts) * class_counts)
    sample_weights = class_weights[y_train]

    dtrain = xgb.DMatrix(X_train, label=y_train, weight=sample_weights)
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
    xgb_val_pred = xgb_model.predict(dval).astype(int)
    print("\nXGBoost — validation performance:")
    print(classification_report(y_val, xgb_val_pred, target_names=class_names, zero_division=0))
    joblib.dump(xgb_model, os.path.join(MODEL_DIR, "xgb_multiclass_unsw.joblib"))

    # ---- Final test-set evaluation (best of the two on val macro-F1) ----
    rf_val_f1 = f1_score(y_val, rf_val_pred, average="macro")
    xgb_val_f1 = f1_score(y_val, xgb_val_pred, average="macro")
    print(f"\nValidation macro-F1 — RF: {rf_val_f1:.4f} | XGBoost: {xgb_val_f1:.4f}")

    best_name = "Random Forest" if rf_val_f1 >= xgb_val_f1 else "XGBoost"
    print(f"Best model on validation: {best_name}\n")

    print("=== FINAL TEST SET EVALUATION (both models) ===")
    rf_test_pred = rf.predict(X_test)
    xgb_test_pred = xgb_model.predict(dtest).astype(int)

    print("\nRandom Forest — test performance:")
    print(classification_report(y_test, rf_test_pred, target_names=class_names, zero_division=0))
    print(f"Macro-F1: {f1_score(y_test, rf_test_pred, average='macro'):.4f}")

    print("\nXGBoost — test performance:")
    print(classification_report(y_test, xgb_test_pred, target_names=class_names, zero_division=0))
    print(f"Macro-F1: {f1_score(y_test, xgb_test_pred, average='macro'):.4f}")

    print("\nDone. Models saved to ml-engine/models/")


if __name__ == "__main__":
    main()
