import os
import copy
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import f1_score
from sklearn.preprocessing import StandardScaler

BASE_DIR = os.path.expanduser("~/ai-cyber-defense")
CIC_DIR = os.path.join(BASE_DIR, "data", "cic-ids-2017")
OUT_DIR = os.path.join(BASE_DIR, "federated", "models")
os.makedirs(OUT_DIR, exist_ok=True)

torch.manual_seed(42)
np.random.seed(42)
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
print(f"Using device: {device}")

NODE_FILES = {
    "node_monday_org": "Monday-WorkingHours.pcap_ISCX.csv",       # all-benign, extreme non-IID stress case
    "node_tuesday_org": "Tuesday-WorkingHours.pcap_ISCX.csv",      # 2.2% attacks
    "node_wednesday_org": "Wednesday-workingHours.pcap_ISCX.csv",  # 31.7% attacks
}

N_ROUNDS = 15
LOCAL_EPOCHS_PER_ROUND = 1


class BinaryClassifier(nn.Module):
    def __init__(self, n_features):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_features, 64), nn.ReLU(),
            nn.Linear(64, 32), nn.ReLU(),
            nn.Linear(32, 1),
        )

    def forward(self, x):
        return self.net(x).squeeze(-1)


def load_node_data(filename):
    df = pd.read_csv(os.path.join(CIC_DIR, filename), low_memory=False)
    df.columns = [c.strip() for c in df.columns]
    df = df.replace([np.inf, -np.inf], np.nan).dropna().drop_duplicates()
    label_col = "Label" if "Label" in df.columns else df.columns[-1]
    y = (df[label_col].astype(str).str.strip().str.upper() != "BENIGN").astype(int).values
    feature_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    X = df[feature_cols].values.astype(np.float32)
    return X, y, feature_cols


def local_epoch(model, X_train, y_train, lr=0.005, batch_size=2048):
    model.train()
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.BCEWithLogitsLoss()
    X_t = torch.tensor(X_train, dtype=torch.float32).to(device)
    y_t = torch.tensor(y_train, dtype=torch.float32).to(device)
    n = len(X_t)
    perm = torch.randperm(n)
    total_loss = 0.0
    for i in range(0, n, batch_size):
        idx = perm[i:i+batch_size]
        xb, yb = X_t[idx], y_t[idx]
        opt.zero_grad()
        loss = loss_fn(model(xb), yb)
        loss.backward()
        opt.step()
        total_loss += loss.item() * len(idx)
    return total_loss / n


def evaluate(model, X_test, y_test):
    model.eval()
    X_t = torch.tensor(X_test, dtype=torch.float32).to(device)
    with torch.no_grad():
        preds = (torch.sigmoid(model(X_t)) > 0.5).cpu().numpy().astype(int)
    return f1_score(y_test, preds, zero_division=0), preds


def weighted_fed_avg(state_dicts, weights):
    weights = np.array(weights) / sum(weights)
    avg_state = copy.deepcopy(state_dicts[0])
    for key in avg_state:
        stacked = torch.stack([sd[key].float() * w for sd, w in zip(state_dicts, weights)], dim=0)
        avg_state[key] = stacked.sum(dim=0)
    return avg_state


def main():
    print("=== Federated Learning Demo v2 (Innovation 5) — proper iterative FedAvg ===")
    print(f"{N_ROUNDS} communication rounds, {LOCAL_EPOCHS_PER_ROUND} local epoch(s) per round\n")

    node_data = {}
    all_feature_cols = None
    for node_name, filename in NODE_FILES.items():
        print(f"Loading {node_name} ({filename})...")
        X, y, feature_cols = load_node_data(filename)
        all_feature_cols = feature_cols if all_feature_cols is None else \
            [c for c in all_feature_cols if c in feature_cols]
        node_data[node_name] = (X, y, feature_cols)
        print(f"  {X.shape[0]} rows, {sum(y)} attacks ({100*sum(y)/len(y):.2f}%)\n")

    n_features = len(all_feature_cols)
    print(f"Common feature set: {n_features} features\n")

    node_splits = {}
    node_weights = {}
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
        node_splits[node_name] = (X_train, y[train_idx], X_test, y[test_idx])
        node_weights[node_name] = split  # weight by local training set size

    # ---- LOCAL-ONLY baselines (each node trains independently, same total epochs) ----
    print("=== Baseline: LOCAL-ONLY models (no federation), for comparison ===\n")
    local_f1 = {}
    for node_name, (X_train, y_train, X_test, y_test) in node_splits.items():
        model = BinaryClassifier(n_features).to(device)
        for _ in range(N_ROUNDS * LOCAL_EPOCHS_PER_ROUND):
            local_epoch(model, X_train, y_train)
        f1, _ = evaluate(model, X_test, y_test)
        local_f1[node_name] = f1
        print(f"  {node_name}: local-only F1 on own test set = {f1:.4f}")

    # ---- TRUE ITERATIVE FEDAVG ----
    print(f"\n=== Federated training: {N_ROUNDS} rounds of local-train -> weighted-average -> broadcast ===")
    global_model = BinaryClassifier(n_features).to(device)
    global_state = global_model.state_dict()

    for round_num in range(N_ROUNDS):
        local_states = []
        for node_name, (X_train, y_train, _, _) in node_splits.items():
            local_model = BinaryClassifier(n_features).to(device)
            local_model.load_state_dict(global_state)  # start from current global weights
            for _ in range(LOCAL_EPOCHS_PER_ROUND):
                local_epoch(local_model, X_train, y_train)
            local_states.append(local_model.state_dict())

        weights = [node_weights[n] for n in node_splits]
        global_state = weighted_fed_avg(local_states, weights)

        if (round_num + 1) % 5 == 0 or round_num == 0:
            global_model.load_state_dict(global_state)
            avg_f1 = np.mean([evaluate(global_model, ns[2], ns[3])[0] for ns in node_splits.values()])
            print(f"  Round {round_num+1}/{N_ROUNDS} — avg F1 across nodes: {avg_f1:.4f}")

    global_model.load_state_dict(global_state)

    print("\n=== FINAL: Federated model evaluated on each node's local test set ===")
    fed_f1 = {}
    for node_name, (_, _, X_test, y_test) in node_splits.items():
        f1, _ = evaluate(global_model, X_test, y_test)
        fed_f1[node_name] = f1
        print(f"  {node_name}: federated F1 = {f1:.4f}")

    print("\n=== SUMMARY: Federated vs Local-only ===")
    print(f"{'Node':22s} {'Local-only F1':>15s} {'Federated F1':>15s} {'Delta':>10s}")
    for node_name in node_splits:
        delta = fed_f1[node_name] - local_f1[node_name]
        sign = "+" if delta >= 0 else ""
        print(f"{node_name:22s} {local_f1[node_name]:15.4f} {fed_f1[node_name]:15.4f} {sign}{delta:9.4f}")

    print(f"\nAverage local-only F1: {np.mean(list(local_f1.values())):.4f}")
    print(f"Average federated F1:  {np.mean(list(fed_f1.values())):.4f}")
    print("\nNote: node_monday_org has 0% attack traffic — a genuine extreme non-IID case.")
    print("Its local F1 is undefined/0 by construction (no positive class to detect locally).")
    print("What matters is whether federation lets Tuesday/Wednesday nodes retain good")
    print("performance despite Monday's degenerate data being included in aggregation.")

    torch.save(global_model.state_dict(), os.path.join(OUT_DIR, "federated_model_v2.pt"))
    print(f"\nSaved to {os.path.join(OUT_DIR, 'federated_model_v2.pt')}")


if __name__ == "__main__":
    main()
