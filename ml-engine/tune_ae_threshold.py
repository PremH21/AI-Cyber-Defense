import pandas as pd, numpy as np, torch, torch.nn as nn
from sklearn.metrics import classification_report, f1_score, roc_auc_score
import joblib

device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

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

meta = joblib.load("ml-engine/models/lstm_ae_meta_unsw.joblib")
model = Autoencoder(meta["n_features"]).to(device)
model.load_state_dict(torch.load("ml-engine/models/lstm_autoencoder_unsw.pt"))
model.eval()

# Tune threshold on VALIDATION set
val_df = pd.read_parquet("data/processed/unsw-nb15_val.parquet")
y_val = val_df["Attack"].values
X_val = torch.tensor(val_df.drop(columns=["Attack"]).values.astype(np.float32)).to(device)
with torch.no_grad():
    val_err = torch.mean((model(X_val) - X_val)**2, dim=1).cpu().numpy()

best_f1, best_thresh = 0, 0
for pct in range(50, 99):
    t = np.percentile(val_err, pct)
    f1 = f1_score(y_val, (val_err > t).astype(int))
    if f1 > best_f1:
        best_f1, best_thresh = f1, t
print(f"Threshold picked on VAL: {best_thresh:.5f} (val F1={best_f1:.4f})")

# Report final number on the untouched TEST set
test_df = pd.read_parquet("data/processed/unsw-nb15_test.parquet")
y_test = test_df["Attack"].values
X_test = torch.tensor(test_df.drop(columns=["Attack"]).values.astype(np.float32)).to(device)
with torch.no_grad():
    test_err = torch.mean((model(X_test) - X_test)**2, dim=1).cpu().numpy()

preds = (test_err > best_thresh).astype(int)
print("\n=== TRUE held-out TEST performance (report this number) ===")
print(classification_report(y_test, preds, target_names=["benign", "attack"]))
print(f"ROC-AUC on test: {roc_auc_score(y_test, test_err):.4f}")
