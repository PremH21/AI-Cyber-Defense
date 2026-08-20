import os
os.environ["OMP_NUM_THREADS"] = "1"

import pandas as pd, numpy as np
from sklearn.metrics import classification_report, f1_score
import joblib

# --- Step 1: all scikit-learn / XGBoost work FIRST, before touching torch/MPS ---
iso_forest = joblib.load("ml-engine/models/isolation_forest_unsw.joblib")
rf = joblib.load("ml-engine/models/random_forest_unsw.joblib")
xgb = joblib.load("ml-engine/models/xgboost_unsw.joblib")

test_df = pd.read_parquet("data/processed/unsw-nb15_test.parquet")
y_test = test_df["Attack"].values
X_test = test_df.drop(columns=["Attack"])

iso_forest.set_params(n_jobs=1)
rf.set_params(n_jobs=1)
xgb.set_params(n_jobs=1)

iso_pred = (iso_forest.predict(X_test) == -1).astype(int)
rf_pred = rf.predict(X_test)
xgb_pred = xgb.predict(X_test)
print("sklearn predictions done")

# --- Step 2: torch/MPS work AFTER sklearn is fully finished ---
import torch, torch.nn as nn

class Autoencoder(nn.Module):
    def __init__(self, n_features):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_features, 32), nn.ReLU(),
            nn.Linear(32, 16), nn.ReLU(),
            nn.Linear(16, 8), nn.ReLU(),
            nn.Linear(8, 16), nn.ReLU(),
            nn.Linear(16, 32), nn.ReLU(),
            nn.Linear(32, n_features)
        )
    def forward(self, x): return self.net(x)

device = torch.device("cpu")  # force CPU here to avoid any MPS/fork interaction
ae_meta = joblib.load("ml-engine/models/lstm_ae_meta_unsw.joblib")
ae = Autoencoder(ae_meta["n_features"]).to(device)
ae.load_state_dict(torch.load("ml-engine/models/lstm_autoencoder_unsw.pt", map_location=device))
ae.eval()
ae_threshold = 0.04192

X_test_t = torch.tensor(X_test.values.astype(np.float32)).to(device)
with torch.no_grad():
    ae_err = torch.mean((ae(X_test_t) - X_test_t)**2, dim=1).numpy()
ae_pred = (ae_err > ae_threshold).astype(int)
print("autoencoder predictions done")

# --- Step 3: combine ---
weighted_score = (0.35 * rf_pred + 0.35 * xgb_pred + 0.15 * iso_pred + 0.15 * ae_pred)
ensemble_pred = (weighted_score >= 0.5).astype(int)

print("\n=== Individual model performance on test set ===")
for name, pred in [("Isolation Forest", iso_pred), ("Autoencoder", ae_pred),
                    ("Random Forest", rf_pred), ("XGBoost", xgb_pred)]:
    print(f"{name}: F1={f1_score(y_test, pred):.4f}")

print("\n=== ENSEMBLE performance on test set ===")
print(classification_report(y_test, ensemble_pred, target_names=["benign", "attack"]))
