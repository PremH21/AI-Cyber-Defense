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

# All 3 nodes now have genuine attack traffic — a fair, working FedAvg demo.
MAIN_NODES = {
    "node_tuesday_org": "Tuesday-WorkingHours.pcap_ISCX.csv",              # 2.2% attacks (Patator)
    "node_wednesday_org": "Wednesday-workingHours.pcap_ISCX.csv",          # 31.7% attacks (DoS)
    "node_friday_ddos_org": "Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv",  # DDoS-heavy
}
# Kept separate as a deliberate stress-test / failure-mode demonstration.
DEGENERATE_NODE = ("node_monday_org", "Monday-WorkingHours.pcap_ISCX.csv")  # 0% attacks

N_ROUNDS = 15
LOCAL_EPOCHS_PER_ROUND = 1
MAX_CLIENT_WEIGHT_SHARE = 0.5  # mitigation: cap any single node's influence in aggregation


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
    """Prevents any single client from dominating aggregation (a real FL mitigation technique)."""
    total = sum(raw_weights)
    capped = [min(w, max_share * total) for w in raw_weights]
    return capped


def prepare_node(filename):
    X, y, feature_cols = load_node_data(filename)
    return X, y, feature_cols


def split_node(X, y, feature_cols, common_cols):
    col_idx = [feature_cols.index(c) for c in common_cols]
    X = X[:, col_idx]
    scaler = StandardScaler()
    n = len(X)
    split = int(n * 0.8)
    perm = np.random.permutation(n)
    train_idx, test_idx = perm[:split], perm[split:]
    X_train = scaler.fit_transform(X[train_idx])
    X_test = scaler.transform(X[test_idx])
    return X_train, y[train_idx], X_test, y[test_idx], split


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
    print("=== PART A: Federated Learning with 3 well-behaved nodes ===\n")

    node_raw = {}
    all_feature_cols = None
    for node_name, filename in MAIN_NODES.items():
        print(f"Loading {node_name} ({filename})...")
        X, y, feature_cols = prepare_node(filename)
        all_feature_cols = feature_cols if all_feature_cols is None else \
            [c for c in all_feature_cols if c in feature_cols]
        node_raw[node_name] = (X, y, feature_cols)
        print(f"  {X.shape[0]} rows, {sum(y)} attacks ({100*sum(y)/len(y):.2f}%)\n")

    n_features = len(all_feature_cols)
    node_splits = {}
    node_weights = {}
    for node_name, (X, y, feature_cols) in node_raw.items():
        X_train, y_train, X_test, y_test, split = split_node(X, y, feature_cols, all_feature_cols)
        node_splits[node_name] = (X_train, y_train, X_test, y_test)
        node_weights[node_name] = split

    print("=== Baseline: LOCAL-ONLY models ===")
    local_f1 = {}
    for node_name, (X_train, y_train, X_test, y_test) in node_splits.items():
        model = BinaryClassifier(n_features).to(device)
        for _ in range(N_ROUNDS * LOCAL_EPOCHS_PER_ROUND):
            local_epoch(model, X_train, y_train)
        f1 = evaluate(model, X_test, y_test)
        local_f1[node_name] = f1
        print(f"  {node_name}: local-only F1 = {f1:.4f}")

    print("\n=== Federated training (3 well-behaved nodes) ===")
    fed_model = run_fedavg(node_splits, node_weights, n_features, label="main")

    print("\n=== FINAL: Federated model on each node's local test set ===")
    fed_f1 = {}
    for node_name, (_, _, X_test, y_test) in node_splits.items():
        f1 = evaluate(fed_model, X_test, y_test)
        fed_f1[node_name] = f1
        print(f"  {node_name}: federated F1 = {f1:.4f}")

    print("\n=== SUMMARY (Part A): Federated vs Local-only, well-behaved nodes ===")
    print(f"{'Node':22s} {'Local-only F1':>15s} {'Federated F1':>15s} {'Delta':>10s}")
    for node_name in node_splits:
        delta = fed_f1[node_name] - local_f1[node_name]
        sign = "+" if delta >= 0 else ""
        print(f"{node_name:22s} {local_f1[node_name]:15.4f} {fed_f1[node_name]:15.4f} {sign}{delta:9.4f}")
    print(f"\nAverage local-only F1: {np.mean(list(local_f1.values())):.4f}")
    print(f"Average federated F1:  {np.mean(list(fed_f1.values())):.4f}")

    torch.save(fed_model.state_dict(), os.path.join(OUT_DIR, "federated_model_v3_main.pt"))

    # ---- PART B: deliberate stress test with the degenerate all-benign node ----
    print("\n\n=== PART B: Stress test — adding a degenerate all-benign node (Monday) ===")
    print("Demonstrates a known FL failure mode (client data imbalance dominating aggregation)")
    print("and a real mitigation (capping any single client's aggregation weight).\n")

    deg_name, deg_file = DEGENERATE_NODE
    print(f"Loading {deg_name} ({deg_file})...")
    X_deg, y_deg, feat_deg = prepare_node(deg_file)
    common_with_deg = [c for c in all_feature_cols if c in feat_deg]
    X_deg_train, y_deg_train, X_deg_test, y_deg_test, deg_split = split_node(X_deg, y_deg, feat_deg, common_with_deg)
    print(f"  {X_deg.shape[0]} rows, {sum(y_deg)} attacks (0.00%)\n")

    stress_splits = dict(node_splits)
    stress_splits[deg_name] = (X_deg_train, y_deg_train, X_deg_test, y_deg_test)
    stress_weights_uncapped = dict(node_weights)
    stress_weights_uncapped[deg_name] = deg_split

    print("--- B1: UNCAPPED weighting (Monday dominates due to size) ---")
    fed_model_uncapped = run_fedavg(stress_splits, stress_weights_uncapped, n_features, label="uncapped")
    print("Result on Tuesday/Wednesday/Friday nodes after including degenerate Monday node:")
    for node_name in node_splits:
        f1 = evaluate(fed_model_uncapped, stress_splits[node_name][2], stress_splits[node_name][3])
        print(f"  {node_name}: F1 = {f1:.4f}")

    print("\n--- B2: CAPPED weighting (mitigation: no client exceeds "
          f"{int(MAX_CLIENT_WEIGHT_SHARE*100)}% of aggregation weight) ---")
    capped_weights_vals = cap_weights(list(stress_weights_uncapped.values()), MAX_CLIENT_WEIGHT_SHARE)
    stress_weights_capped = dict(zip(stress_weights_uncapped.keys(), capped_weights_vals))
    fed_model_capped = run_fedavg(stress_splits, stress_weights_capped, n_features, label="capped")
    print("Result on Tuesday/Wednesday/Friday nodes with capped Monday influence:")
    for node_name in node_splits:
        f1 = evaluate(fed_model_capped, stress_splits[node_name][2], stress_splits[node_name][3])
        print(f"  {node_name}: F1 = {f1:.4f}")

    print("\nDone.")


if __name__ == "__main__":
    main()
