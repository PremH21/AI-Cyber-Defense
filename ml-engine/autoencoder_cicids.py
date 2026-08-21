import pandas as pd, numpy as np, torch, torch.nn as nn
from sklearn.metrics import classification_report, f1_score, roc_auc_score
import joblib, os

TRAIN_PATH = "data/processed/cic-ids-2017_train.parquet"
VAL_PATH = "data/processed/cic-ids-2017_val.parquet"
TEST_PATH = "data/processed/cic-ids-2017_test.parquet"
MODEL_OUT = "ml-engine/models/autoencoder_cicids.pt"
META_OUT = "ml-engine/models/autoencoder_meta_cicids.joblib"
device = torch.device("cpu")  # CPU for stability, matches prior fix for the ensemble segfault

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

def main():
    train_df = pd.read_parquet(TRAIN_PATH)
    benign = train_df[train_df["Attack"] == 0].drop(columns=["Attack"]).values.astype(np.float32)
    print(f"Training on {benign.shape[0]} benign-only rows")

    n_features = benign.shape[1]
    X_train_t = torch.tensor(benign).to(device)

    model = Autoencoder(n_features).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    loss_fn = nn.MSELoss()
    batch_size, n_epochs, n = 1024, 15, X_train_t.shape[0]

    for epoch in range(n_epochs):
        perm = torch.randperm(n)
        total_loss = 0.0
        for i in range(0, n, batch_size):
            idx = perm[i:i+batch_size]
            batch = X_train_t[idx]
            optimizer.zero_grad()
            recon = model(batch)
            loss = loss_fn(recon, batch)
            loss.backward(); optimizer.step()
            total_loss += loss.item() * batch.size(0)
        print(f"Epoch {epoch+1}/{n_epochs} MSE={total_loss/n:.5f}")

    with torch.no_grad():
        train_err = torch.mean((model(X_train_t) - X_train_t)**2, dim=1).numpy()
    threshold = float(np.percentile(train_err, 95))

    val_df = pd.read_parquet(VAL_PATH)
    y_val = val_df["Attack"].values
    X_val_t = torch.tensor(val_df.drop(columns=["Attack"]).values.astype(np.float32)).to(device)
    with torch.no_grad():
        val_err = torch.mean((model(X_val_t) - X_val_t)**2, dim=1).numpy()

    best_f1, best_thresh = 0, threshold
    for pct in range(50, 99):
        t = np.percentile(val_err, pct)
        f1 = f1_score(y_val, (val_err > t).astype(int))
        if f1 > best_f1:
            best_f1, best_thresh = f1, t

    test_df = pd.read_parquet(TEST_PATH)
    y_test = test_df["Attack"].values
    X_test_t = torch.tensor(test_df.drop(columns=["Attack"]).values.astype(np.float32)).to(device)
    with torch.no_grad():
        test_err = torch.mean((model(X_test_t) - X_test_t)**2, dim=1).numpy()

    preds = (test_err > best_thresh).astype(int)
    print(f"\n=== TEST performance (threshold tuned on val, tested on held-out test) ===")
    print(classification_report(y_test, preds, target_names=["benign", "attack"]))
    print(f"ROC-AUC: {roc_auc_score(y_test, test_err):.4f}")

    os.makedirs(os.path.dirname(MODEL_OUT), exist_ok=True)
    torch.save(model.state_dict(), MODEL_OUT)
    joblib.dump({"threshold": best_thresh, "n_features": n_features}, META_OUT)
    print(f"Saved to {MODEL_OUT}")

if __name__ == "__main__":
    main()
