import numpy as np
import pandas as pd
from sklearn.metrics import classification_report, f1_score

TEST_PATH = "data/processed/unsw-nb15_test.parquet"


def main():
    print("Loading true labels...", flush=True)
    test_df = pd.read_parquet(TEST_PATH)
    y_test = test_df["Attack"].values

    print("Loading saved predictions...", flush=True)
    iso_pred = np.load("ml-engine/models/iso_unsw_test_preds.npy")
    rf_pred = np.load("ml-engine/models/rf_unsw_test_preds.npy")
    xgb_pred = np.load("ml-engine/models/xgb_unsw_test_preds.npy")
    ae_pred = np.load("ml-engine/models/ae_unsw_test_preds.npy")

    weighted_score = (0.35 * rf_pred + 0.35 * xgb_pred + 0.15 * iso_pred + 0.15 * ae_pred)
    ensemble_pred = (weighted_score >= 0.5).astype(int)

    print("\n=== Individual model performance on test set ===")
    for name, pred in [("Isolation Forest", iso_pred), ("Autoencoder", ae_pred),
                        ("Random Forest", rf_pred), ("XGBoost", xgb_pred)]:
        print(f"{name}: F1={f1_score(y_test, pred):.4f}")

    print("\n=== ENSEMBLE performance on test set ===")
    print(classification_report(y_test, ensemble_pred, target_names=["benign", "attack"]))


if __name__ == "__main__":
    main()
