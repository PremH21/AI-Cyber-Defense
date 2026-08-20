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
MAX_CLIENT_WEIGHT_SHARE = 0.5


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


def cap_weights(raw_weights, max_share):
    total = sum(raw_weights)
    return [min(w, max_share * total) for w in raw_weights]


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
    print("=== Loading all 4 nodes (3 main + 1 degenerate) for stress test ===\n")

    node_raw = {}
    all_feature_cols = None
    for node_name, filename in list(MAIN_NODES.items()) + [DEGENERATE_NODE]:
        print(f"Loading {node_name} ({filename})...")
        X, y, feature_cols = load_node_data(filename)
        all_feature_cols = feature_cols if all_feature_cols is None else \
            [c for c in all_feature_cols if c in feature_cols]
        node_raw[node_name] = (X, y, feature_cols)
        print(f"  {X.shape[0]} rows, {sum(y)} attacks ({100*sum(y)/len(y):.2f}%)\n")

    n_features = len(all_feature_cols)

    # Shared normalization computed from ALL 4 nodes (including degenerate) —
    # this itself matters: does one degenerate node skew the shared stats too?
    aligned_Xs, sizes = [], []
    for node_name, (X, y, feature_cols) in node_raw.items():
        col_idx = [feature_cols.index(c) for c in all_feature_cols]
        aligned_Xs.append(X[:, col_idx])
        sizes.append(len(X))

    print("Computing SHARED federated normalization statistics across ALL 4 nodes...")
    global_mean, global_std = federated_normalize_stats(aligned_Xs, sizes)
    print("Done.\n")

    all_splits = {}
    all_weights = {}
    for node_name, (X, y, feature_cols) in node_raw.items():
        X_train, y_train, X_test, y_test, split = split_node(
            X, y, feature_cols, all_feature_cols, global_mean, global_std)
        all_splits[node_name] = (X_train, y_train, X_test, y_test)
        all_weights[node_name] = split

    main_names = list(MAIN_NODES.keys())
    deg_name = DEGENERATE_NODE[0]

    print("=== B1: UNCAPPED — all 4 nodes weighted by size (Monday is largest, dominates) ===")
    fed_uncapped = run_fedavg(all_splits, all_weights, n_features, label="uncapped")
    print("\nResult on the 3 real-attack nodes after including degenerate Monday:")
    uncapped_results = {}
    for name in main_names:
        f1 = evaluate(fed_uncapped, all_splits[name][2], all_splits[name][3])
        uncapped_results[name] = f1
        print(f"  {name}: F1 = {f1:.4f}")

    print("\n=== B2: CAPPED — no single node exceeds "
          f"{int(MAX_CLIENT_WEIGHT_SHARE*100)}% of aggregation weight (mitigation) ===")
    capped_vals = cap_weights(list(all_weights.values()), MAX_CLIENT_WEIGHT_SHARE)
    capped_weights = dict(zip(all_weights.keys(), capped_vals))
    fed_capped = run_fedavg(all_splits, capped_weights, n_features, label="capped")
    print("\nResult on the 3 real-attack nodes with capped Monday influence:")
    capped_results = {}
    for name in main_names:
        f1 = evaluate(fed_capped, all_splits[name][2], all_splits[name][3])
        capped_results[name] = f1
        print(f"  {name}: F1 = {f1:.4f}")

    print("\n=== SUMMARY: effect of degenerate node + mitigation ===")
    print(f"{'Node':22s} {'No-Monday (v4)':>16s} {'+Monday uncapped':>18s} {'+Monday capped':>16s}")
    reference_v4 = {"node_tuesday_org": 0.8337, "node_wednesday_org": 0.9196, "node_friday_ddos_org": 0.7231}
    for name in main_names:
        print(f"{name:22s} {reference_v4[name]:16.4f} {uncapped_results[name]:18.4f} {capped_results[name]:16.4f}")

    avg_v4 = np.mean(list(reference_v4.values()))
    avg_uncapped = np.mean(list(uncapped_results.values()))
    avg_capped = np.mean(list(capped_results.values()))
    print(f"\nAverage F1 — no degenerate node: {avg_v4:.4f}")
    print(f"Average F1 — degenerate node, uncapped weight: {avg_uncapped:.4f}")
    print(f"Average F1 — degenerate node, capped weight:   {avg_capped:.4f}")
    print(f"\nDegradation from degenerate node (uncapped): {avg_uncapped - avg_v4:+.4f}")
    print(f"Recovery from capping mitigation: {avg_capped - avg_uncapped:+.4f}")

    torch.save(fed_capped.state_dict(), os.path.join(OUT_DIR, "federated_model_v5_capped.pt"))
    print(f"\nSaved capped model to {os.path.join(OUT_DIR, 'federated_model_v5_capped.pt')}")


if __name__ == "__main__":
    main()
