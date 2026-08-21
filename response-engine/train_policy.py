import json, random
import pandas as pd
import joblib

ACTIONS = ["alert_only", "block_ip", "quarantine", "rollback", "escalate_human"]
ALPHA, GAMMA = 0.1, 0.9
EPSILON_START, EPSILON_END = 0.4, 0.05
N_EPISODES = 100000

def reward_fn(action, is_real_attack, severity):
    sw = severity / 10.0
    if action in ("block_ip", "quarantine", "rollback"):
        return 10 * sw if is_real_attack else -5
    elif action == "alert_only":
        return -4 * sw if is_real_attack else 2   # correct low-cost handling now rewarded properly
    elif action == "escalate_human":
        return 0.5 if is_real_attack else -1        # analyst time has a real cost; only pays off on genuine attacks
    return 0

def asset_bucket(severity):
    r = random.random()
    if severity >= 7: return "high" if r < 0.6 else "medium"
    elif severity >= 4: return "medium" if r < 0.6 else random.choice(["low", "high"])
    else: return "low" if r < 0.6 else "medium"

def main():
    print("Loading real model predictions + true labels...")
    model = joblib.load("ml-engine/models/xgboost_unsw.joblib")
    test_df = pd.read_parquet("data/processed/unsw-nb15_test.parquet")
    y_true = test_df["Attack"].values
    X = test_df.drop(columns=["Attack"])
    proba = model.predict_proba(X)[:, 1]

    q_table, visit_counts = {}, {}
    def get_q(state):
        if state not in q_table:
            q_table[state] = {a: 0.0 for a in ACTIONS}
            visit_counts[state] = {a: 0 for a in ACTIONS}
        return q_table[state]

    print(f"Training over {N_EPISODES} episodes with epsilon decay...")
    n = len(y_true)
    for ep in range(N_EPISODES):
        epsilon = EPSILON_START - (EPSILON_START - EPSILON_END) * (ep / N_EPISODES)
        idx = random.randrange(n)
        is_attack = bool(y_true[idx])
        confidence = proba[idx]
        severity = min(10, max(1, int(confidence * 10) + (3 if is_attack else 0)))
        asset = asset_bucket(severity)
        state = f"{severity}|{asset}"

        q = get_q(state)
        action = random.choice(ACTIONS) if random.random() < epsilon else max(q, key=q.get)
        visit_counts[state][action] += 1
        r = reward_fn(action, is_attack, severity)
        old_q = q[action]
        q[action] = old_q + ALPHA * (r + GAMMA * max(q.values()) - old_q)

        if ep % 20000 == 0:
            print(f"  Episode {ep}/{N_EPISODES}  epsilon={epsilon:.3f}")

    print(f"\nTraining complete. {len(q_table)} states.")
    print("\nFull learned policy:")
    for state in sorted(q_table.keys()):
        best = max(q_table[state], key=q_table[state].get)
        print(f"  {state:12s} -> {best:15s} (visited {visit_counts[state][best]} times)")

    import os
    os.makedirs("response-engine/models", exist_ok=True)
    with open("response-engine/models/q_table.json", "w") as f:
        json.dump(q_table, f, indent=2)
    print("\nSaved to response-engine/models/q_table.json")

if __name__ == "__main__":
    main()
