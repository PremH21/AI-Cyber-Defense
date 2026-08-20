import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import joblib

TEST_PATH = "data/processed/unsw-nb15_test.parquet"
OUT_PATH = "ml-engine/models/ae_unsw_test_preds.npy"


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
    print("Loading test data...", flush=True)
    test_df = pd.read_parquet(TEST_PATH)
    X_test = test_df.drop(columns=["Attack"]).values.astype(np.float32)

    meta = joblib.load("ml-engine/models/autoencoder_unsw_meta.joblib")
    device = torch.device("cpu")
    ae = Autoencoder(meta["n_features"]).to(device)
    ae.load_state_dict(torch.load("ml-engine/models/autoencoder_unsw.pt", map_location=device))
    ae.eval()

    threshold = meta["threshold"]
    print(f"Running inference on {len(X_test)} rows...", flush=True)
    with torch.no_grad():
        batch = torch.tensor(X_test)
        recon = ae(batch)
        err = torch.mean((recon - batch) ** 2, dim=1).numpy()

    pred = (err > threshold).astype(int)
    np.save(OUT_PATH, pred)
    print(f"Saved autoencoder predictions to {OUT_PATH}", flush=True)


if __name__ == "__main__":
    main()
