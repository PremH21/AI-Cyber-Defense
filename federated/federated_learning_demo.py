import os
import copy
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import f1_score, classification_report
from sklearn.preprocessing import StandardScaler
import glob

BASE_DIR = os.path.expanduser("~/ai-cyber-defense")
CIC_DIR = os.path.join(BASE_DIR, "data", "cic-ids-2017")
OUT_DIR = os.path.join(BASE_DIR, "federated", "models")
os.makedirs(OUT_DIR, exist_ok=True)

torch.manual_seed(42)
np.random.seed(42)

device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
print(f"Using device: {device}")

# Node partitioning: each "organization" gets genuinely different traffic,
# just like real federated learning across independent enterprises.
NODE_FILES = {
    "node_monday_org": "Monday-WorkingHours.pcap_ISCX.csv",
    "node_tuesday_org": "Tuesday-WorkingHours.pcap_ISCX.csv",
    "node_wednesday_org": "Wednesday-workingHours.pcap_ISCX.csv",
}


class BinaryClassifier(nn.Module):
    def __init__(self, n_features):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_features, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
        )

    def forward(self, x):
        return self.net(x).squeeze(-1)


def load_node_data(filename):
    path = os.path.join(CIC_DIR, filename)
    df = pd.read_csv(path, low_memory=False)
    df.columns = [c.strip() for c in df.columns]
    df = df.replace([np.inf, -np.inf], np.nan).dropna().drop_duplicates()

    label_col = "Label" if "Label" in df.columns else df.columns[-1]
    y = (df[label_col].astype(str).str.strip().str.upper() != "BENIGN").astype(int).values

    feature_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    X = df[feature_cols].values.astype(np.float32)
    return X, y, feature_cols


def train_local(model, X_train, y_train, epochs=5, lr=0.01, batch_size=2048):
    model.train()
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.BCEWithLogitsLoss()

    X_t = torch.tensor(X_train, dtype=torch.float32).to(device)
    y_t = torch.tensor(y_train, dtype=torch.float32).to(device)
    n = len(X_t)

    for epoch in range(epochs):
        perm = torch.randperm(n)
        total_loss = 0.0
        for i in range(0, n, batch_size):
            idx = perm[i:i+batch_size]
            xb, yb = X_t[idx], y_t[idx]
            opt.zero_grad()
            out = model(xb)
            loss = loss_fn(out, yb)
            loss.backward()
            opt.step()
            total_loss += loss.item() * len(idx)
        print(f"    epoch {epoch+1}/{epochs} — loss: {total_loss/n:.4f}")
    return model


def evaluate(model, X_test, y_test):
    model.eval()
    X_t = torch.tensor(X_test, dtype=torch.float32).to(device)
    with torch.no_grad():
        logits = model(X_t)
        preds = (torch.sigmoid(logits) > 0.5).cpu().numpy().astype(int)
    return f1_score(y_test, preds), preds


def fed_avg(state_dicts):
    avg_state = copy.deepcopy(state_dicts[0])
    for key in avg_state:
        avg_state[key] = torch.stack([sd[key].float() for sd in state_dicts], dim=0).mean(dim=0)
    return avg_state


def main():
    print("=== Federated Learning Demo (Innovation 5) ===")
    print("3 simulated organization nodes, each with genuinely different local traffic\n")

    node_data = {}
    all_feature_cols = None
    for node_name, filename in NODE_FILES.items():
        print(f"Loading {node_name} ({filename})...")
        X, y, feature_cols = load_node_data(filename)
        if all_feature_cols is None:
            all_feature_cols = feature_cols
        else:
            # align to common feature set across all nodes
            common = [c for c in all_feature_cols if c in feature_cols]
            all_feature_cols = common
        node_data[node_name] = (X, y, feature_cols)
        print(f"  {X.shape[0]} rows, {sum(y)} attacks ({100*sum(y)/len(y):.1f}%)\n")

    n_features = len(all_feature_cols)
    print(f"Common feature set across all nodes: {n_features} features\n")

    # Split each node's data into local train/test, and re-select common columns
    node_splits = {}
    for node_name, (X, y, feature_cols) in node_data.items():
        col_idx = [feature_cols.index(c) for c in all_feature_cols]
        X = X[:, col_idx]

        scaler = StandardScaler()
        n = len(X)
        split = int(n * 0.8)
        perm = np.random.permutation(n)
        train_idx, test_idx = perm[:split], perm[split:]

        X_train = scaler.fit_transform(X[train_idx])
        X_test = scaler.transform(X[test_idx])
        y_train, y_test = y[train_idx], y[test_idx]

        node_splits[node_name] = (X_train, y_train, X_test, y_test)

    # ---- STEP 1: Train LOCAL-ONLY models per node ----
    print("=== STEP 1: Training LOCAL-ONLY models (no federation) ===\n")
    local_models = {}
    local_f1_own_test = {}
    for node_name, (X_train, y_train, X_test, y_test) in node_splits.items():
        print(f"Training local model for {node_name}...")
        model = BinaryClassifier(n_features).to(device)
        model = train_local(model, X_train, y_train, epochs=5)
        f1, _ = evaluate(model, X_test, y_test)
        local_models[node_name] = model
        local_f1_own_test[node_name] = f1
        print(f"  {node_name} local model F1 on OWN test set: {f1:.4f}\n")

    # Cross-evaluate: how well does each node's LOCAL model generalize to OTHER nodes' data?
    print("=== Cross-node generalization of LOCAL-ONLY models (the federation problem) ===")
    cross_f1 = {}
    for train_node, model in local_models.items():
        for eval_node, (_, _, X_test, y_test) in node_splits.items():
            f1, _ = evaluate(model, X_test, y_test)
            cross_f1[(train_node, eval_node)] = f1
            marker = " (own data)" if train_node == eval_node else ""
            print(f"  Model trained on {train_node:20s} -> tested on {eval_node:20s}: F1={f1:.4f}{marker}")

    # ---- STEP 2: Federated Averaging (FedAvg) ----
    print("\n=== STEP 2: FEDERATED AVERAGING — aggregating local model weights ===")
    state_dicts = [m.state_dict() for m in local_models.values()]
    fed_state = fed_avg(state_dicts)

    fed_model = BinaryClassifier(n_features).to(device)
    fed_model.load_state_dict(fed_state)

    print("Evaluating FEDERATED model on each node's local test set:")
    fed_f1_per_node = {}
    for node_name, (_, _, X_test, y_test) in node_splits.items():
        f1, _ = evaluate(fed_model, X_test, y_test)
        fed_f1_per_node[node_name] = f1
        print(f"  Federated model -> {node_name:20s}: F1={f1:.4f}")

    # ---- SUMMARY: does federation actually help? ----
    print("\n=== SUMMARY: Federated vs Local-only (the actual test of Innovation 5) ===")
    print(f"{'Node':22s} {'Local-only F1':>15s} {'Federated F1':>15s} {'Delta':>10s}")
    for node_name in node_splits:
        local = local_f1_own_test[node_name]
        fed = fed_f1_per_node[node_name]
        delta = fed - local
        sign = "+" if delta >= 0 else ""
        print(f"{node_name:22s} {local:15.4f} {fed:15.4f} {sign}{delta:9.4f}")

    avg_local = np.mean(list(local_f1_own_test.values()))
    avg_fed = np.mean(list(fed_f1_per_node.values()))
    print(f"\nAverage local-only F1 (each node on own data): {avg_local:.4f}")
    print(f"Average federated F1 (shared model on each node's data): {avg_fed:.4f}")
    print(f"Net effect of federation: {avg_fed - avg_local:+.4f}")

    torch.save(fed_model.state_dict(), os.path.join(OUT_DIR, "federated_model.pt"))
    print(f"\nSaved federated model to {os.path.join(OUT_DIR, 'federated_model.pt')}")


if __name__ == "__main__":
    main()
