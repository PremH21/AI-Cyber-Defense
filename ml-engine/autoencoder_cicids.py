import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import classification_report, roc_auc_score, f1_score
import joblib, os

TRAIN_PATH = "data/processed/cic-ids-2017_train.parquet"
VAL_PATH = "data/processed/cic-ids-2017_val.parquet"
MODEL_OUT = "ml-engine/models/autoencoder_cicids.pt"
META_OUT = "ml-engine/models/autoencoder_cicids_meta.joblib"

device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
print(f"Device: {device}")

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

def main():
    train_df = pd.read_parquet(TRAIN_PATH)
    benign_only = train_df[train_df["Attack"] == 0].drop(columns=["Attack"])
    n_features = benign_only.shape[1]
    print(f"Training on {benign_only.shape[0]} benign-only rows, {n_features} features")

    X_train_t = torch.tensor(benign_only.values.astype(np.float32)).to(device)

    model = Autoencoder(n_features).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    loss_fn = nn.MSELoss()

    n_epochs = 15
    batch_size = 2048
    for epoch in range(n_epochs):
        perm = torch.randperm(X_train_t.shape[0])
        total_loss = 0.0
        for i in range(0, len(perm), batch_size):
            idx = perm[i:i+batch_size]
            batch = X_train_t[idx]
            opt.zero_grad()
            recon = model(batch)
            loss = loss_fn(recon, batch)
            loss.backward()
            opt.step()
            total_loss += loss.item() * batch.size(0)
        if epoch % 3 == 0 or epoch == n_epochs - 1:
            print(f"Epoch {epoch}/{n_epochs} MSE={total_loss/X_train_t.shape[0]:.5f}")

    model.eval()
    val_df = pd.read_parquet(VAL_PATH)
    y_val = val_df["Attack"].values
    X_val_t = torch.tensor(val_df.drop(columns=["Attack"]).values.astype(np.float32)).to(device)

    with torch.no_grad():
        train_err = torch.mean((model(X_train_t) - X_train_t)**2, dim=1).cpu().numpy()
        val_err = torch.mean((model(X_val_t) - X_val_t)**2, dim=1).cpu().numpy()

    threshold = float(np.percentile(train_err, 95))
    preds = (val_err > threshold).astype(int)
    print(f"\nThreshold (95th pct of train error): {threshold:.5f}")
    print(classification_report(y_val, preds, target_names=["benign", "attack"]))
    print(f"ROC-AUC: {roc_auc_score(y_val, val_err):.4f}")

    os.makedirs(os.path.dirname(MODEL_OUT), exist_ok=True)
    torch.save(model.state_dict(), MODEL_OUT)
    joblib.dump({"threshold": threshold, "n_features": n_features}, META_OUT)
    print(f"Saved to {MODEL_OUT}")

if __name__ == "__main__":
    main()
