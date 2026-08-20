import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.metrics import classification_report, f1_score
import joblib
import os

TRAIN_PATH = "data/processed/unsw-nb15_train.parquet"
TEST_PATH = "data/processed/unsw-nb15_test.parquet"
MODEL_OUT_DIR = "federated/models"

def train_local_model(X, y, seed):
    model = xgb.XGBClassifier(
        n_estimators=300,
        max_depth=10,
        learning_rate=0.05,
        subsample=0.9,
        colsample_bytree=0.8,
        min_child_weight=3,
        random_state=seed,
        n_jobs=-1,
        eval_metric="logloss",
    )
    model.fit(X, y)
    return model

def main():
    print("Loading data...")
    train_df = pd.read_parquet(TRAIN_PATH)
    test_df = pd.read_parquet(TEST_PATH)

    y_train_full = train_df["Attack"]
    X_train_full = train_df.drop(columns=["Attack"])
    y_test = test_df["Attack"]
    X_test = test_df.drop(columns=["Attack"])

    print(f"Full centralized training set: {X_train_full.shape[0]} rows")

    # --- Simulate 2 organizations by splitting the training data in half ---
    # (In real federated learning, each org's data never leaves their own machine.
    #  We simulate this by physically partitioning the data here and training two
    #  fully independent local models, then aggregating only their predictions.)
    rng = np.random.default_rng(42)
    idx = rng.permutation(len(X_train_full))
    half = len(idx) // 2
    idx_org1, idx_org2 = idx[:half], idx[half:]

    X_org1, y_org1 = X_train_full.iloc[idx_org1], y_train_full.iloc[idx_org1]
    X_org2, y_org2 = X_train_full.iloc[idx_org2], y_train_full.iloc[idx_org2]

    print(f"\nOrg 1 local data: {X_org1.shape[0]} rows (never sees Org 2's data)")
    print(f"Org 2 local data: {X_org2.shape[0]} rows (never sees Org 1's data)")

    print("\nTraining Org 1's local model...")
    model_org1 = train_local_model(X_org1, y_org1, seed=1)

    print("Training Org 2's local model...")
    model_org2 = train_local_model(X_org2, y_org2, seed=2)

    print("\nTraining centralized baseline (full data, no federation) for comparison...")
    model_central = train_local_model(X_train_full, y_train_full, seed=42)

    # --- Evaluate each org's local-only model on the shared test set ---
    pred_org1 = model_org1.predict(X_test)
    pred_org2 = model_org2.predict(X_test)
    pred_central = model_central.predict(X_test)

    print("\n=== Org 1 alone (50% of data, no collaboration) ===")
    print(classification_report(y_test, pred_org1, target_names=["benign", "attack"]))

    print("=== Org 2 alone (50% of data, no collaboration) ===")
    print(classification_report(y_test, pred_org2, target_names=["benign", "attack"]))

    # --- Federated aggregation: average the two orgs' predicted probabilities ---
    # This simulates the benefit of federated model aggregation (each org keeps
    # its raw data private, but the ensemble of their models captures patterns
    # neither org's local-only data alone could fully represent).
    proba_org1 = model_org1.predict_proba(X_test)[:, 1]
    proba_org2 = model_org2.predict_proba(X_test)[:, 1]
    fed_proba = (proba_org1 + proba_org2) / 2
    fed_pred = (fed_proba >= 0.5).astype(int)

    print("=== FEDERATED (Org 1 + Org 2 aggregated, no raw data shared) ===")
    print(classification_report(y_test, fed_pred, target_names=["benign", "attack"]))

    print("=== Centralized baseline (100% of data pooled together) ===")
    print(classification_report(y_test, pred_central, target_names=["benign", "attack"]))

    print("\n=== Summary (F1 on attack class) ===")
    print(f"Org 1 alone:        {f1_score(y_test, pred_org1):.4f}")
    print(f"Org 2 alone:        {f1_score(y_test, pred_org2):.4f}")
    print(f"Federated (2-node): {f1_score(y_test, fed_pred):.4f}")
    print(f"Centralized (full): {f1_score(y_test, pred_central):.4f}")

    os.makedirs(MODEL_OUT_DIR, exist_ok=True)
    joblib.dump(model_org1, f"{MODEL_OUT_DIR}/org1_model.joblib")
    joblib.dump(model_org2, f"{MODEL_OUT_DIR}/org2_model.joblib")
    print(f"\nLocal node models saved to {MODEL_OUT_DIR}/")

if __name__ == "__main__":
    main()
