import pandas as pd, numpy as np, torch, torch.nn as nn
from sklearn.metrics import classification_report, roc_auc_score
import joblib, os

TRAIN_PATH = "data/processed/unsw-nb15_train.parquet"
VAL_PATH = "data/processed/unsw-nb15_val.parquet"
MODEL_OUT = "ml-engine/models/lstm_autoencoder_unsw.pt"
META_OUT = "ml-engine/models/lstm_ae_meta_unsw.joblib"
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
    def forward(self, x):
        return self.net(x)

def main():
    print(f"Device: {device}")
    train_df = pd.read_parquet(TRAIN_PATH)
    benign = train_df[train_df["Attack"] == 0].drop(columns=["Attack"]).values.astype(np.float32)
    val_df = pd.read_parquet(VAL_PATH)
    y_val = val_df["Attack"].values
    X_val = val_df.drop(columns=["Attack"]).values.astype(np.float32)

    n_features = benign.shape[1]
    X_train_t = torch.tensor(benign).to(device)
    X_val_t = torch.tensor(X_val).to(device)

    model = Autoencoder(n_features).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    loss_fn = nn.MSELoss()
    batch_size, n_epochs, n = 512, 30, X_train_t.shape[0]

    for epoch in range(n_epochs):
        model.train()
        perm = torch.randperm(n)
        total_loss = 0.0
        for i in range(0, n, batch_size):
            idx = perm[i:i+batch_size]
            batch = X_train_t[idx]
            optimizer.zero_grad()
            recon = model(batch)
            loss = loss_fn(recon, batch)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * batch.size(0)
        if epoch % 5 == 0:
            print(f"Epoch {epoch}/{n_epochs} MSE={total_loss/n:.5f}")

    model.eval()
    with torch.no_grad():
        train_err = torch.mean((model(X_train_t) - X_train_t)**2, dim=1).cpu().numpy()
        val_err = torch.mean((model(X_val_t) - X_val_t)**2, dim=1).cpu().numpy()

    threshold = float(np.percentile(train_err, 95))
    preds = (val_err > threshold).astype(int)
    print(classification_report(y_val, preds, target_names=["benign", "attack"]))
    print(f"ROC-AUC: {roc_auc_score(y_val, val_err):.4f}")

    os.makedirs(os.path.dirname(MODEL_OUT), exist_ok=True)
    torch.save(model.state_dict(), MODEL_OUT)
    joblib.dump({"threshold": threshold, "n_features": n_features}, META_OUT)
    print(f"Saved to {MODEL_OUT}")

if __name__ == "__main__":
    main()
