import pandas as pd
from sklearn.metrics import classification_report
import xgboost as xgb
import joblib
import os

TRAIN_PATH = "data/processed/cic-ids-2017_train.parquet"
VAL_PATH = "data/processed/cic-ids-2017_val.parquet"
MODEL_OUT = "ml-engine/models/xgboost_cicids.joblib"

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

    print("\nTraining XGBClassifier on CIC-IDS-2017 (this may take a few minutes, ~2.9M rows)...")
    model = xgb.XGBClassifier(
        n_estimators=300,
        max_depth=10,
        learning_rate=0.05,
        subsample=0.9,
        colsample_bytree=0.8,
        min_child_weight=3,
        random_state=42,
        n_jobs=-1,
        eval_metric="logloss",
    )
    model.fit(X_train, y_train)

    print("\nPredicting on validation set...")
    preds = model.predict(X_val)

    print("\nEvaluation on validation set:")
    print(classification_report(y_val, preds, target_names=["benign", "attack"]))

    importances = pd.Series(model.feature_importances_, index=X_train.columns)
    top10 = importances.sort_values(ascending=False).head(10)
    print("Top 10 most important features:")
    print(top10)

    os.makedirs(os.path.dirname(MODEL_OUT), exist_ok=True)
    joblib.dump(model, MODEL_OUT)
    print(f"\nModel saved to {MODEL_OUT}")

if __name__ == "__main__":
    main()
