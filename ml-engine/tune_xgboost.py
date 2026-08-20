import pandas as pd
from xgboost import XGBClassifier
from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import classification_report
import joblib, os

train_df = pd.read_parquet("data/processed/unsw-nb15_train.parquet")
y_train = train_df["Attack"]
X_train = train_df.drop(columns=["Attack"])

val_df = pd.read_parquet("data/processed/unsw-nb15_val.parquet")
y_val = val_df["Attack"]
X_val = val_df.drop(columns=["Attack"])

param_dist = {
    "n_estimators": [100, 200, 300, 400],
    "max_depth": [4, 6, 8, 10],
    "learning_rate": [0.01, 0.05, 0.1, 0.2],
    "subsample": [0.7, 0.8, 0.9, 1.0],
    "colsample_bytree": [0.7, 0.8, 0.9, 1.0],
    "min_child_weight": [1, 3, 5],
}

search = RandomizedSearchCV(
    XGBClassifier(random_state=42, n_jobs=-1, eval_metric="logloss"),
    param_distributions=param_dist,
    n_iter=25,
    scoring="f1",
    cv=3,
    random_state=42,
    n_jobs=-1,
    verbose=2,
)
print("Searching (this will take a few minutes)...")
search.fit(X_train, y_train)

print(f"\nBest params: {search.best_params_}")
print(f"Best CV F1: {search.best_score_:.4f}")

best_model = search.best_estimator_
preds = best_model.predict(X_val)
print("\nValidation performance with tuned model:")
print(classification_report(y_val, preds, target_names=["benign", "attack"]))

joblib.dump(best_model, "ml-engine/models/xgboost_unsw.joblib")
print("Saved tuned model to ml-engine/models/xgboost_unsw.joblib")
