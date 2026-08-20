import os
import copy
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import f1_score

BASE_DIR = os.path.expanduser("~/ai-cyber-defense")
CIC_DIR = os.path.join(BASE_DIR, "data", "cic-ids-2017")
OUT_DIR = os.path.join(BASE_DIR, "federated", "models")
os.makedirs(OUT_DIR, exist_ok=True)

torch.manual_seed(42)
np.random.seed(42)
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
print(f"Using device: {device}")

MAIN_NODES = {
    "node_tuesday_org": "Tuesday-WorkingHours.pcap_ISCX.csv",
    "node_wednesday_org": "Wednesday-workingHours.pcap_ISCX.csv",
    "node_friday_ddos_org": "Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv",
}
DEGENERATE_NODE = ("node_monday_org", "Monday-WorkingHours.pcap_ISCX.csv")

N_ROUNDS = 15
LOCAL_EPOCHS_PER_ROUND = 1
MIN_POSITIVE_CLASS_FRACTION = 0.001  # client validation threshold: needs >0.1% positive examples


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
    return f1_score(y_test, preds, zero_division=0)


def weighted_fed_avg(state_dicts, weights):
    weights = np.array(weights) / sum(weights)
    avg_state = copy.deepcopy(state_dicts[0])
    for key in avg_state:
        stacked = torch.stack([sd[key].float() * w for sd, w in zip(state_dicts, weights)], dim=0)
        avg_state[key] = stacked.sum(dim=0)
    return avg_state


def federated_normalize_stats(node_raw_Xs, node_sizes):
    total = sum(node_sizes)
    global_mean = sum(X.mean(axis=0) * n for X, n in zip(node_raw_Xs, node_sizes)) / total
    global_var = sum(((X.var(axis=0) + (X.mean(axis=0) - global_mean) ** 2) * n)
                      for X, n in zip(node_raw_Xs, node_sizes)) / total
    global_std = np.sqrt(global_var)
    global_std[global_std == 0] = 1.0
    return global_mean, global_std


def split_node(X, y, feature_cols, common_cols, global_mean, global_std):
    col_idx = [feature_cols.index(c) for c in common_cols]
    X = X[:, col_idx]
    X = (X - global_mean) / global_std
    n = len(X)
    split = int(n * 0.8)
    perm = np.random.permutation(n)
    train_idx, test_idx = perm[:split], perm[split:]
    return X[train_idx], y[train_idx], X[test_idx], y[test_idx], split


def run_fedavg(node_splits, node_weights, n_features, label=""):
    global_model = BinaryClassifier(n_features).to(device)
    global_state = global_model.state_dict()
    for round_num in range(N_ROUNDS):
        local_states = []
        for node_name, (X_train, y_train, _, _) in node_splits.items():
            local_model = BinaryClassifier(n_features).to(device)
            local_model.load_state_dict(global_state)
            for _ in range(LOCAL_EPOCHS_PER_ROUND):
                local_epoch(local_model, X_train, y_train)
            local_states.append(local_model.state_dict())
        weights = [node_weights[n] for n in node_splits]
        global_state = weighted_fed_avg(local_states, weights)
        if (round_num + 1) % 5 == 0 or round_num == 0:
            global_model.load_state_dict(global_state)
            avg_f1 = np.mean([evaluate(global_model, ns[2], ns[3]) for ns in node_splits.values()])
            print(f"  [{label}] Round {round_num+1}/{N_ROUNDS} — avg F1 across nodes: {avg_f1:.4f}")
    global_model.load_state_dict(global_state)
    return global_model


def main():
    print("=== Loading all 4 nodes for CLIENT-VALIDATED federated learning ===\n")

    node_raw = {}
    all_feature_cols = None
    for node_name, filename in list(MAIN_NODES.items()) + [DEGENERATE_NODE]:
        print(f"Loading {node_name} ({filename})...")
        X, y, feature_cols = load_node_data(filename)
        all_feature_cols = feature_cols if all_feature_cols is None else \
            [c for c in all_feature_cols if c in feature_cols]
        node_raw[node_name] = (X, y, feature_cols)
        pos_frac = sum(y) / len(y)
        print(f"  {X.shape[0]} rows, {sum(y)} attacks ({100*pos_frac:.3f}%)\n")

    n_features = len(all_feature_cols)

    # ---- CLIENT VALIDATION STEP: exclude clients that can't support meaningful training ----
    print("=== CLIENT VALIDATION (aggregator-side check before accepting a client) ===")
    valid_nodes = {}
    rejected_nodes = {}
    for node_name, (X, y, feature_cols) in node_raw.items():
        pos_frac = sum(y) / len(y)
        if pos_frac < MIN_POSITIVE_CLASS_FRACTION:
            rejected_nodes[node_name] = pos_frac
            print(f"  REJECTED: {node_name} — positive class fraction {100*pos_frac:.4f}% "
                  f"< threshold {100*MIN_POSITIVE_CLASS_FRACTION:.4f}% (cannot contribute meaningful attack signal)")
        else:
            valid_nodes[node_name] = (X, y, feature_cols)
            print(f"  ACCEPTED: {node_name} — positive class fraction {100*pos_frac:.4f}%")
    print()

    # Shared normalization computed from ALL nodes (even rejected ones can safely
    # contribute to normalization stats — that's not sensitive to class imbalance)
    aligned_Xs, sizes = [], []
    for node_name, (X, y, feature_cols) in node_raw.items():
        col_idx = [feature_cols.index(c) for c in all_feature_cols]
        aligned_Xs.append(X[:, col_idx])
        sizes.append(len(X))
    global_mean, global_std = federated_normalize_stats(aligned_Xs, sizes)

    node_splits = {}
    node_weights = {}
    for node_name, (X, y, feature_cols) in valid_nodes.items():
        X_train, y_train, X_test, y_test, split = split_node(
            X, y, feature_cols, all_feature_cols, global_mean, global_std)
        node_splits[node_name] = (X_train, y_train, X_test, y_test)
        node_weights[node_name] = split

    print(f"=== Federated training with {len(valid_nodes)} VALIDATED clients "
          f"({len(rejected_nodes)} rejected) ===")
    fed_model = run_fedavg(node_splits, node_weights, n_features, label="validated")

    print("\n=== FINAL: Federated model (client-validated) on each accepted node's test set ===")
    fed_f1 = {}
    for node_name, (_, _, X_test, y_test) in node_splits.items():
        f1 = evaluate(fed_model, X_test, y_test)
        fed_f1[node_name] = f1
        print(f"  {node_name}: F1 = {f1:.4f}")

    print("\n=== COMPARISON ACROSS ALL APPROACHES ===")
    reference = {
        "No degenerate node (v4)": {"node_tuesday_org": 0.8337, "node_wednesday_org": 0.9196, "node_friday_ddos_org": 0.7231},
        "+Monday, uncapped (v5)": {"node_tuesday_org": 0.0000, "node_wednesday_org": 0.9000, "node_friday_ddos_org": 0.7241},
        "+Monday, weight-capped (v5)": {"node_tuesday_org": 0.0000, "node_wednesday_org": 0.8950, "node_friday_ddos_org": 0.7269},
        "+Monday, client-VALIDATED (v6)": fed_f1,
    }
    print(f"{'Approach':32s} {'Tuesday':>10s} {'Wednesday':>10s} {'Friday-DDoS':>12s} {'Average':>10s}")
    for approach, results in reference.items():
        avg = np.mean(list(results.values()))
        print(f"{approach:32s} {results['node_tuesday_org']:10.4f} {results['node_wednesday_org']:10.4f} "
              f"{results['node_friday_ddos_org']:12.4f} {avg:10.4f}")

    torch.save(fed_model.state_dict(), os.path.join(OUT_DIR, "federated_model_v6_validated.pt"))
    print(f"\nSaved to {os.path.join(OUT_DIR, 'federated_model_v6_validated.pt')}")


if __name__ == "__main__":
    main()
