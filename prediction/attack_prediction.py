import pandas as pd
import numpy as np
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import classification_report
import joblib
import os

DATA_PATH = "data/processed/unsw-nb15_train.parquet"
MODEL_OUT = "prediction/models/attack_prediction_gb.joblib"
WINDOW_SIZE = 200  # rows per time window (simulated temporal bucket)

def build_temporal_features(df):
    df = df.reset_index(drop=True)
    df["window"] = df.index // WINDOW_SIZE

    window_attacks = df.groupby("window")["Attack"].sum().rename("attack_count")
    window_df = window_attacks.reset_index()

    window_df["lag1"] = window_df["attack_count"].shift(1).fillna(0)
    window_df["lag2"] = window_df["attack_count"].shift(2).fillna(0)
    window_df["lag3"] = window_df["attack_count"].shift(3).fillna(0)
    window_df["rolling_mean_3"] = window_df["attack_count"].rolling(3).mean().fillna(0)

    window_df["next_window_spike"] = (
        window_df["attack_count"].shift(-1) > window_df["attack_count"].median()
    ).astype(int)

    window_df = window_df.dropna(subset=["next_window_spike"])
    return window_df

def main():
    print("Loading data...")
    df = pd.read_parquet(DATA_PATH)
    print(f"Loaded {df.shape[0]} rows")

    print("Building temporal window features...")
    window_df = build_temporal_features(df)
    print(f"Built {window_df.shape[0]} time windows (window size={WINDOW_SIZE} rows)")

    feature_cols = ["lag1", "lag2", "lag3", "rolling_mean_3"]
    X = window_df[feature_cols]
    y = window_df["next_window_spike"]

    split_idx = int(len(X) * 0.8)
    X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
    y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]

    print(f"\nTrain windows: {len(X_train)}, Test windows: {len(X_test)}")
    print("Training GradientBoostingClassifier...")
    model = GradientBoostingClassifier(n_estimators=150, max_depth=3, random_state=42)
    model.fit(X_train, y_train)

    preds = model.predict(X_test)
    print("\nEvaluation (predicting next-window attack spike):")
    print(classification_report(y_test, preds, target_names=["no_spike", "spike"], zero_division=0))

    os.makedirs(os.path.dirname(MODEL_OUT), exist_ok=True)
    joblib.dump(model, MODEL_OUT)
    print(f"\nModel saved to {MODEL_OUT}")

if __name__ == "__main__":
    main()
