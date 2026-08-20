import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
import joblib
import os

TRAIN_PATH = "data/processed/unsw-nb15_train.parquet"
VAL_PATH = "data/processed/unsw-nb15_val.parquet"
MODEL_OUT = "ml-engine/models/random_forest_unsw.joblib"

def main():
    print("Loading training data...")
    train_df = pd.read_parquet(TRAIN_PATH)
    y_train = train_df["Attack"]
    X_train = train_df.drop(columns=["Attack"])
    print(f"Train: {X_train.shape[0]} rows, {X_train.shape[1]} features")

    print("Loading validation data...")
    val_df = pd.read_parquet(VAL_PATH)
    y_val = val_df["Attack"]
    X_val = val_df.drop(columns=["Attack"])
    print(f"Val: {X_val.shape[0]} rows")

    print("\nTraining RandomForestClassifier...")
    model = RandomForestClassifier(
        n_estimators=200,
        max_depth=None,
        random_state=42,
        n_jobs=-1,
    )
    model.fit(X_train, y_train)

    print("\nPredicting on validation set...")
    preds = model.predict(X_val)

    print("\nFirst 10 validation rows - predicted / true label:")
    for i in range(10):
        print(f"  row {i}: pred={preds[i]}  true_attack={y_val.iloc[i]}")

    print("\nEvaluation on validation set:")
    print(classification_report(y_val, preds, target_names=["benign", "attack"]))

    # Which features mattered most - useful later for your SHAP dashboard and report
    importances = pd.Series(model.feature_importances_, index=X_train.columns)
    top10 = importances.sort_values(ascending=False).head(10)
    print("Top 10 most important features:")
    print(top10)

    os.makedirs(os.path.dirname(MODEL_OUT), exist_ok=True)
    joblib.dump(model, MODEL_OUT)
    print(f"\nModel saved to {MODEL_OUT}")

if __name__ == "__main__":
    main()
