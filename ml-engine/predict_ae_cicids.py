import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import joblib

TEST_PATH = "data/processed/cic-ids-2017_test.parquet"
OUT_PATH = "ml-engine/models/ae_cicids_test_preds.npy"

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
    print("Loading test data...", flush=True)
    test_df = pd.read_parquet(TEST_PATH)
    X_test = test_df.drop(columns=["Attack"]).values.astype(np.float32)

    device = torch.device("cpu")
    meta = joblib.load("ml-engine/models/autoencoder_cicids_meta.joblib")
    ae = Autoencoder(meta["n_features"]).to(device)
    ae.load_state_dict(torch.load("ml-engine/models/autoencoder_cicids.pt", map_location=device))
    ae.eval()
    threshold = meta["threshold"]

    print(f"Running inference on {len(X_test)} rows in batches...", flush=True)
    batch_size = 5000
    errs = []
    with torch.no_grad():
        for i in range(0, len(X_test), batch_size):
            batch = torch.tensor(X_test[i:i+batch_size])
            recon = ae(batch)
            err = torch.mean((recon - batch)**2, dim=1).numpy()
            errs.append(err)
            print(f"  {i}/{len(X_test)}", flush=True)

    ae_err = np.concatenate(errs)
    ae_pred = (ae_err > threshold).astype(int)
    np.save(OUT_PATH, ae_pred)
    print(f"Saved autoencoder predictions to {OUT_PATH}", flush=True)

if __name__ == "__main__":
    main()
