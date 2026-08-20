import pandas as pd
import numpy as np
import joblib
import gc
from sklearn.metrics import classification_report, f1_score

TEST_PATH = "data/processed/cic-ids-2017_test.parquet"

def main():
    print("Loading test data...", flush=True)
    test_df = pd.read_parquet(TEST_PATH)
    y_test = test_df["Attack"].values
    X_test = test_df.drop(columns=["Attack"])
    print("Test data loaded.", flush=True)

    # --- Isolation Forest ---
    print("Loading isolation forest...", flush=True)
    iso_forest = joblib.load("ml-engine/models/isolation_forest_cicids.joblib")
    print("Predicting isolation forest...", flush=True)
    iso_pred = (iso_forest.predict(X_test) == -1).astype(int)
    print("Isolation forest done.", flush=True)
    del iso_forest
    gc.collect()

    # --- Random Forest ---
    print("Loading random forest...", flush=True)
    rf = joblib.load("ml-engine/models/random_forest_cicids.joblib")
    print("Predicting random forest...", flush=True)
    rf_pred = rf.predict(X_test)
    print("Random forest done.", flush=True)
    del rf
    gc.collect()

    # --- XGBoost ---
    print("Loading xgboost...", flush=True)
    xgb = joblib.load("ml-engine/models/xgboost_cicids.joblib")
    print("Predicting xgboost...", flush=True)
    xgb_pred = xgb.predict(X_test)
    print("XGBoost done.", flush=True)
    del xgb
    gc.collect()

    # --- Autoencoder (import torch only now, after sklearn work is done) ---
    print("Importing torch...", flush=True)
    import torch
    import torch.nn as nn
    device = torch.device("cpu")

    class Autoencoder(nn.Module):
        def __init__(self, n_features):
            super().__init__()
            self.net = nn.Sequential(
                nn.Linear(n_features, 64), nn.ReLU(),
                nn.Linear(64, 32), nn.ReLU(),
                nn.Linear(32, 16), nn.ReLU(),
                nn.Linear(16, 32), nn.ReLU(),
                nn.Linear(32, 64), nn.ReLU(),
                nn.Linear(64, n_features),
            )
        def forward(self, x):
            return self.net(x)

    print("Loading autoencoder...", flush=True)
    ae_meta = joblib.load("ml-engine/models/autoencoder_cicids_meta.joblib")
    ae = Autoencoder(ae_meta["n_features"]).to(device)
    ae.load_state_dict(torch.load("ml-engine/models/autoencoder_cicids.pt", map_location=device))
    ae.eval()
    ae_threshold = ae_meta["threshold"]
    print("Autoencoder loaded.", flush=True)

    print("Predicting autoencoder...", flush=True)
    X_test_t = torch.tensor(X_test.values.astype(np.float32)).to(device)
    with torch.no_grad():
        ae_err = torch.mean((ae(X_test_t) - X_test_t)**2, dim=1).cpu().numpy()
    ae_pred = (ae_err > ae_threshold).astype(int)
    print("Autoencoder done.", flush=True)

    # --- Combine ---
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
