import pandas as pd
import numpy as np
import joblib
import random
import json
import os

TEST_PATH = "data/processed/unsw-nb15_test.parquet"
XGB_PATH = "ml-engine/models/xgboost_unsw.joblib"
POLICY_OUT = "response-engine/policy_table.json"

ACTIONS = ["alert_only", "block_ip", "quarantine", "rollback"]

def severity_bin(prob):
    if prob < 0.20:
        return 0
    elif prob < 0.40:
        return 1
    elif prob < 0.60:
        return 2
    elif prob < 0.80:
        return 3
    else:
        return 4

def compute_reward(action, true_label, severity_prob):
    action_strength = {"alert_only": 0, "block_ip": 1, "quarantine": 2, "rollback": 3}[action]

    if true_label == 0:
        return 1.0 if action == "alert_only" else -0.3 * action_strength

    if severity_prob >= 0.80:
        ideal = 3
    elif severity_prob >= 0.60:
        ideal = 2
    elif severity_prob >= 0.40:
        ideal = 1
    else:
        ideal = 0

    return 1.0 - 0.5 * abs(action_strength - ideal)

def main():
    print("Loading test data and XGBoost model...")
    test_df = pd.read_parquet(TEST_PATH)
    y_true = test_df["Attack"].values
    X_test = test_df.drop(columns=["Attack"])

    xgb = joblib.load(XGB_PATH)
    severity_probs = xgb.predict_proba(X_test)[:, 1]

    n_states = 5
    n_actions = len(ACTIONS)
    Q = np.zeros((n_states, n_actions))

    alpha = 0.1
    epsilon = 0.2
    n_episodes = 5

    print(f"Training Q-learning policy over {n_episodes} episodes on {len(y_true)} samples...")
    rng = np.random.default_rng(42)

    for ep in range(n_episodes):
        idx = rng.permutation(len(y_true))
        total_reward = 0.0
        for i in idx:
            state = severity_bin(severity_probs[i])

            if random.random() < epsilon:
                action_idx = random.randrange(n_actions)
            else:
                action_idx = int(np.argmax(Q[state]))

            action = ACTIONS[action_idx]
            reward = compute_reward(action, y_true[i], severity_probs[i])
            total_reward += reward

            Q[state, action_idx] += alpha * (reward - Q[state, action_idx])

        print(f"  Episode {ep+1}/{n_episodes}: avg reward = {total_reward/len(y_true):.4f}")

    print("\nLearned policy (best action per severity bin):")
    bin_labels = ["0-20%", "20-40%", "40-60%", "60-80%", "80-100%"]
    policy = {}
    for s in range(n_states):
        best_action = ACTIONS[int(np.argmax(Q[s]))]
        policy[bin_labels[s]] = best_action
        print(f"  Severity {bin_labels[s]}: -> {best_action}  (Q-values: {dict(zip(ACTIONS, np.round(Q[s], 3)))})")

    os.makedirs(os.path.dirname(POLICY_OUT), exist_ok=True)
    with open(POLICY_OUT, "w") as f:
        json.dump({"policy": policy, "q_table": Q.tolist(), "actions": ACTIONS, "bins": bin_labels}, f, indent=2)
    print(f"\nPolicy saved to {POLICY_OUT}")

    print("\n=== Evaluating learned policy on full test set ===")
    action_counts = {a: 0 for a in ACTIONS}
    total_reward = 0.0
    for i in range(len(y_true)):
        state = severity_bin(severity_probs[i])
        action_idx = int(np.argmax(Q[state]))
        action = ACTIONS[action_idx]
        action_counts[action] += 1
        total_reward += compute_reward(action, y_true[i], severity_probs[i])

    print(f"Average reward per detection: {total_reward/len(y_true):.4f}")
    print("Action distribution:")
    for a, c in action_counts.items():
        print(f"  {a}: {c} ({100*c/len(y_true):.1f}%)")

def choose_action(severity_prob, policy_path=POLICY_OUT):
    with open(policy_path) as f:
        data = json.load(f)
    bin_idx = severity_bin(severity_prob)
    bin_label = data["bins"][bin_idx]
    return data["policy"][bin_label]

if __name__ == "__main__":
    main()
