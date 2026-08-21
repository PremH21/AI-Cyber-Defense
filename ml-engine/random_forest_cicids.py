import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
import joblib, os

TRAIN_PATH = "data/processed/cic-ids-2017_train.parquet"
VAL_PATH = "data/processed/cic-ids-2017_val.parquet"
MODEL_OUT = "ml-engine/models/random_forest_cicids.joblib"

def main():
    print("Loading training data (this is the 2.9M-row dataset, will take a moment)...")
    train_df = pd.read_parquet(TRAIN_PATH)
    y_train = train_df["Attack"]
    X_train = train_df.drop(columns=["Attack"])
    print(f"Train: {X_train.shape[0]} rows, {X_train.shape[1]} features")

    val_df = pd.read_parquet(VAL_PATH)
    y_val = val_df["Attack"]
    X_val = val_df.drop(columns=["Attack"])

    print("Training RandomForestClassifier (may take several minutes on this dataset size)...")
    model = RandomForestClassifier(n_estimators=200, random_state=42, n_jobs=-1)
    model.fit(X_train, y_train)

    preds = model.predict(X_val)
    print(classification_report(y_val, preds, target_names=["benign", "attack"]))

    os.makedirs(os.path.dirname(MODEL_OUT), exist_ok=True)
    joblib.dump(model, MODEL_OUT)
    print(f"Saved to {MODEL_OUT}")

if __name__ == "__main__":
    main()
