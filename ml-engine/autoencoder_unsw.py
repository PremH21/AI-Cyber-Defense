import os
import time
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import joblib

TRAIN_PATH = "data/processed/unsw-nb15_train.parquet"
VAL_PATH = "data/processed/unsw-nb15_val.parquet"
MODEL_OUT = "ml-engine/models/autoencoder_unsw.pt"
META_OUT = "ml-engine/models/autoencoder_unsw_meta.joblib"

EPOCHS = 15
BATCH_SIZE = 512
LR = 0.001


class Autoencoder(nn.Module):
    def __init__(self, n_features):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(n_features, 32), nn.ReLU(),
            nn.Linear(32, 16), nn.ReLU(),
            nn.Linear(16, 8), nn.ReLU(),
        )
        self.decoder = nn.Sequential(
            nn.Linear(8, 16), nn.ReLU(),
            nn.Linear(16, 32), nn.ReLU(),
            nn.Linear(32, n_features),
        )

    def forward(self, x):
        return self.decoder(self.encoder(x))


def main():
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Using device: {device}", flush=True)

    print("Loading training data (benign-only, as autoencoders should train on normal traffic)...", flush=True)
    train_df = pd.read_parquet(TRAIN_PATH)
    y_train = train_df["Attack"]
    X_train_full = train_df.drop(columns=["Attack"])
    # Train the autoencoder ONLY on benign rows so it learns to reconstruct "normal" well
    # and produces high reconstruction error on attacks (the whole point of AE-based anomaly detection)
    X_train_benign = X_train_full[y_train == 0].values.astype(np.float32)
    print(f"Training on {X_train_benign.shape[0]} benign rows, {X_train_benign.shape[1]} features", flush=True)

    print("Loading validation data...", flush=True)
    val_df = pd.read_parquet(VAL_PATH)
    y_val = val_df["Attack"].values
    X_val = val_df.drop(columns=["Attack"]).values.astype(np.float32)

    n_features = X_train_benign.shape[1]
    model = Autoencoder(n_features).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=LR)
    loss_fn = nn.MSELoss()

    X_train_t = torch.tensor(X_train_benign).to(device)
    n = len(X_train_t)

    print("\nTraining autoencoder...", flush=True)
    for epoch in range(EPOCHS):
        model.train()
        perm = torch.randperm(n)
        total_loss = 0.0
        start = time.time()
        for i in range(0, n, BATCH_SIZE):
            idx = perm[i:i + BATCH_SIZE]
            batch = X_train_t[idx]
            opt.zero_grad()
            recon = model(batch)
            loss = loss_fn(recon, batch)
            loss.backward()
            opt.step()
            total_loss += loss.item() * len(idx)
        avg_loss = total_loss / n
        print(f"  Epoch {epoch+1}/{EPOCHS} — avg recon loss: {avg_loss:.6f} ({time.time()-start:.1f}s)", flush=True)

    print("\nComputing reconstruction errors on validation set to pick a threshold...", flush=True)
    model.eval()
    X_val_t = torch.tensor(X_val).to(device)
    with torch.no_grad():
        recon_val = model(X_val_t)
        errs_val = torch.mean((recon_val - X_val_t) ** 2, dim=1).cpu().numpy()

    # Threshold: use the benign rows' error distribution — pick the 95th percentile of benign error
    # as the cutoff (i.e. flag anything reconstructing worse than 95% of normal traffic)
    benign_errs = errs_val[y_val == 0]
    threshold = float(np.percentile(benign_errs, 95))
    print(f"Threshold (95th percentile of benign reconstruction error): {threshold:.6f}", flush=True)

    preds_val = (errs_val > threshold).astype(int)
    from sklearn.metrics import classification_report
    print("\nAutoencoder performance on validation set (using this threshold):")
    print(classification_report(y_val, preds_val, target_names=["benign", "attack"]))

    os.makedirs(os.path.dirname(MODEL_OUT), exist_ok=True)
    torch.save(model.state_dict(), MODEL_OUT)
    joblib.dump({"n_features": n_features, "threshold": threshold}, META_OUT)
    print(f"\nModel saved to {MODEL_OUT}")
    print(f"Meta (n_features, threshold) saved to {META_OUT}")


if __name__ == "__main__":
    main()
