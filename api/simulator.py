"""
Traffic Simulator — continuously feeds real UNSW-NB15 test rows into the
live API, simulating a stream of incoming network events. Mixes benign
and attack rows so the demo shows the system's actual behavior across
different severities, not just a scripted attack sequence.

Run with the API already running in another terminal:
  python api/simulator.py
"""
import time
import random
import requests
import pandas as pd

API_URL = "http://localhost:8000/ingest"
TEST_PATH = "data/processed/unsw-nb15_test.parquet"
INTERVAL_SECONDS = 2  # gap between simulated events
ASSET_TIERS = ["low", "medium", "high"]

def main():
    df = pd.read_parquet(TEST_PATH)
    n = len(df)
    print(f"Loaded {n} real test-set rows to simulate as live traffic.")
    print(f"Sending one event every {INTERVAL_SECONDS}s to {API_URL}\n")

    try:
        while True:
            row_index = random.randrange(n)
            asset = random.choice(ASSET_TIERS)
            payload = {"row_index": row_index, "asset_criticality": asset}

            try:
                resp = requests.post(API_URL, json=payload, timeout=5)
                result = resp.json()
                label = result.get("predicted_label", "?")
                true = result.get("true_label", "?")
                sev = result.get("severity", "?")
                action = result.get("action_taken", "?")
                latency = result.get("detection_latency_ms", "?")
                match = "OK" if label == true else "MISS"
                print(f"[row {row_index:6d}] true={true:7s} pred={label:7s} [{match}]  "
                      f"severity={sev}  asset={asset:6s}  action={action:15s}  {latency}ms")
            except requests.exceptions.ConnectionError:
                print("ERROR: API not reachable. Is uvicorn running on port 8000?")
                break
            except Exception as e:
                print(f"ERROR: {e}")

            time.sleep(INTERVAL_SECONDS)
    except KeyboardInterrupt:
        print("\nSimulator stopped.")

if __name__ == "__main__":
    main()
