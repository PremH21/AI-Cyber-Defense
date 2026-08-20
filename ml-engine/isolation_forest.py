import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.metrics import classification_report
import joblib
import os

DATA_PATH = "data/processed/unsw-nb15_train.parquet"
MODEL_OUT = "ml-engine/models/isolation_forest_unsw.joblib"

def main():
    print("Loading data...")
    df = pd.read_parquet(DATA_PATH)

    y = df["Attack"]
    X = df.drop(columns=["Attack"])

    print(f"Loaded {X.shape[0]} rows, {X.shape[1]} features")

    contamination = y.mean()
    print(f"Using contamination={contamination:.3f}")

    print("Fitting IsolationForest...")
    model = IsolationForest(
        n_estimators=100,
        contamination=contamination,
        random_state=42,
        n_jobs=-1,
    )
    model.fit(X)

    scores = model.decision_function(X)
    preds = model.predict(X)

    print("\nFirst 10 rows - anomaly score / prediction / true label:")
    for i in range(10):
        pred_label = "ANOMALY" if preds[i] == -1 else "normal"
        print(f"  row {i}: score={scores[i]:.4f}  pred={pred_label}  true_attack={y.iloc[i]}")

    preds_binary = (preds == -1).astype(int)
    print("\nSanity-check against true labels:")
    print(classification_report(y, preds_binary, target_names=["benign", "attack"]))

    os.makedirs(os.path.dirname(MODEL_OUT), exist_ok=True)
    joblib.dump(model, MODEL_OUT)
    print(f"\nModel saved to {MODEL_OUT}")

if __name__ == "__main__":
    main()
