"""
Demo helper: seeds the watched folder with sample files, then after a delay,
simulates a ransomware attack by deleting/corrupting most of them —
so you can watch Shadow Copy Guardian detect and auto-restore in real time.

Run this in a SEPARATE terminal while shadow_copy_guardian.py is running.
"""

import os
import time
import random
import string

WATCHED_DIR = "response-engine/demo_data/watched_folder"


def seed_files(n=10):
    os.makedirs(WATCHED_DIR, exist_ok=True)
    for i in range(n):
        content = "".join(random.choices(string.ascii_letters, k=200))
        with open(os.path.join(WATCHED_DIR, f"important_doc_{i}.txt"), "w") as f:
            f.write(content)
    print(f"Seeded {n} files into {WATCHED_DIR}")


def simulate_attack():
    files = os.listdir(WATCHED_DIR)
    print(f"\nSimulating ransomware attack: encrypting/deleting {len(files)} files...")
    for fname in files:
        fpath = os.path.join(WATCHED_DIR, fname)
        if random.random() < 0.5:
            os.remove(fpath)
        else:
            with open(fpath, "w") as f:
                f.write("ENCRYPTED_BY_SIMULATED_RANSOMWARE_" + "".join(random.choices(string.ascii_letters, k=50)))
    print("Attack simulation complete. Watch the Guardian terminal for detection + auto-restore.")


if __name__ == "__main__":
    print("=== Step 1: seeding clean files ===")
    seed_files(10)
    print("\nWaiting 12 seconds so the Guardian can take a clean baseline snapshot...")
    time.sleep(12)
    print("\n=== Step 2: simulating attack ===")
    simulate_attack()
