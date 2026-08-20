import os
import json
import numpy as np
import pandas as pd
import random

BASE_DIR = os.path.expanduser("~/ai-cyber-defense")
DATA_DIR = os.path.join(BASE_DIR, "data", "processed")
MODEL_DIR = os.path.join(BASE_DIR, "response-engine", "models")
os.makedirs(MODEL_DIR, exist_ok=True)

ACTIONS = ["block", "quarantine", "virtual_patch", "alert_only", "rollback"]
SEVERITIES = list(range(1, 11))
ASSET_LEVELS = ["low", "medium", "high"]
THREAT_TYPES = ["benign", "dos_ddos", "recon_scan", "web_attack", "credential_attack",
                "malware_backdoor", "infiltration", "botnet"]

random.seed(42)
np.random.seed(42)


def map_label_to_threat_type(label: str) -> str:
    label = label.lower()
    if "benign" in label or "normal" in label:
        return "benign"
    if "dos" in label or "ddos" in label:
        return "dos_ddos"
    if "portscan" in label or "recon" in label or "fuzzer" in label or "analysis" in label:
        return "recon_scan"
    if "web attack" in label or "sql" in label or "xss" in label:
        return "web_attack"
    if "patator" in label or "brute" in label:
        return "credential_attack"
    if "backdoor" in label or "shellcode" in label or "exploit" in label or "generic" in label:
        return "malware_backdoor"
    if "infiltration" in label:
        return "infiltration"
    if "bot" in label or "worm" in label:
        return "botnet"
    return "malware_backdoor"  # conservative default for anything unmapped


def severity_for_threat(threat_type: str) -> int:
    base = {
        "benign": 0,
        "recon_scan": 3,
        "web_attack": 6,
        "credential_attack": 6,
        "dos_ddos": 7,
        "botnet": 8,
        "malware_backdoor": 8,
        "infiltration": 9,
    }[threat_type]
    # add small noise to simulate confidence-driven severity variance
    return int(np.clip(base + np.random.randint(-1, 2), 0, 10))


def reward_function(threat_type, severity, asset_level, action):
    """
    Reward design:
    - Correctly matching action strength to threat severity/asset value = high reward
    - Over-reacting to benign/low-severity = penalty (operational cost, false positive)
    - Under-reacting to high-severity/high-asset = large penalty (real damage)
    """
    is_benign = threat_type == "benign"
    asset_weight = {"low": 1.0, "medium": 1.5, "high": 2.2}[asset_level]

    action_strength = {
        "alert_only": 1,
        "virtual_patch": 2,
        "quarantine": 3,
        "block": 4,
        "rollback": 5,
    }[action]

    if is_benign:
        # any action beyond alert_only on benign traffic is a false-positive cost
        return -2.0 * (action_strength - 1) * asset_weight if action != "alert_only" else 1.0

    ideal_strength = 1 + round((severity / 10) * 4)  # maps severity 0-10 to strength 1-5
    mismatch = action_strength - ideal_strength

    if mismatch == 0:
        return 5.0 * asset_weight
    elif mismatch > 0:
        # over-reaction: mild penalty (operational disruption), scaled by asset value
        return -0.5 * mismatch * asset_weight
    else:
        # under-reaction to a real threat: severe penalty, worse on high-value assets
        return -2.0 * abs(mismatch) * asset_weight * (severity / 5)


def state_key(threat_type, severity, asset_level):
    return (threat_type, severity, asset_level)


def train_q_learning(n_episodes=200000, alpha=0.1, gamma=0.9, epsilon_start=1.0, epsilon_end=0.05):
    q_table = {}

    def get_q(state):
        if state not in q_table:
            q_table[state] = {a: 0.0 for a in ACTIONS}
        return q_table[state]

    reward_history = []
    epsilon_decay = (epsilon_start - epsilon_end) / n_episodes

    for ep in range(n_episodes):
        epsilon = max(epsilon_end, epsilon_start - ep * epsilon_decay)

        threat_type = random.choice(THREAT_TYPES)
        severity = severity_for_threat(threat_type)
        asset_level = random.choice(ASSET_LEVELS)
        state = state_key(threat_type, severity, asset_level)

        q_values = get_q(state)
        if random.random() < epsilon:
            action = random.choice(ACTIONS)
        else:
            action = max(q_values, key=q_values.get)

        reward = reward_function(threat_type, severity, asset_level, action)

        # single-step episodic task: next-state value = 0 (terminal)
        q_values[action] += alpha * (reward - q_values[action])
        reward_history.append(reward)

        if (ep + 1) % 20000 == 0:
            recent_avg = np.mean(reward_history[-20000:])
            print(f"Episode {ep+1}/{n_episodes} — epsilon={epsilon:.3f} — avg reward (last 20k): {recent_avg:.3f}")

    return q_table, reward_history


def evaluate_policy(q_table, n_eval=5000):
    correct_strength_match = 0
    total_reward = 0.0
    false_positive_actions = 0
    dangerous_underreactions = 0

    for _ in range(n_eval):
        threat_type = random.choice(THREAT_TYPES)
        severity = severity_for_threat(threat_type)
        asset_level = random.choice(ASSET_LEVELS)
        state = state_key(threat_type, severity, asset_level)

        q_values = q_table.get(state, {a: 0.0 for a in ACTIONS})
        action = max(q_values, key=q_values.get)
        reward = reward_function(threat_type, severity, asset_level, action)
        total_reward += reward

        if threat_type == "benign" and action != "alert_only":
            false_positive_actions += 1
        if threat_type != "benign" and severity >= 7 and action in ("alert_only", "virtual_patch"):
            dangerous_underreactions += 1

    print(f"\n=== Policy evaluation over {n_eval} simulated incidents ===")
    print(f"Average reward per incident: {total_reward/n_eval:.3f}")
    print(f"False-positive over-reactions on benign traffic: {false_positive_actions} ({100*false_positive_actions/n_eval:.2f}%)")
    print(f"Dangerous under-reactions on severity>=7 threats: {dangerous_underreactions} ({100*dangerous_underreactions/n_eval:.2f}%)")


def print_sample_policy(q_table):
    print("\n=== Sample learned policy (threat_type, severity, asset) -> best action ===")
    for threat_type in THREAT_TYPES:
        for asset_level in ASSET_LEVELS:
            severity = severity_for_threat(threat_type)
            state = state_key(threat_type, severity, asset_level)
            q_values = q_table.get(state, {a: 0.0 for a in ACTIONS})
            best_action = max(q_values, key=q_values.get)
            print(f"  {threat_type:20s} sev={severity:2d} asset={asset_level:6s} -> {best_action}")


def main():
    print("Training tabular Q-learning RL response agent...")
    print(f"States: {len(THREAT_TYPES)} threat types x 11 severities x {len(ASSET_LEVELS)} asset levels")
    print(f"Actions: {ACTIONS}\n")

    q_table, reward_history = train_q_learning()

    print_sample_policy(q_table)
    evaluate_policy(q_table)

    serializable_q = {f"{k[0]}|{k[1]}|{k[2]}": v for k, v in q_table.items()}
    with open(os.path.join(MODEL_DIR, "q_table.json"), "w") as f:
        json.dump(serializable_q, f, indent=2)
    print(f"\nSaved Q-table to {os.path.join(MODEL_DIR, 'q_table.json')} ({len(q_table)} states learned)")


if __name__ == "__main__":
    main()
