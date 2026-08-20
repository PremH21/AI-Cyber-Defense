import pandas as pd
import numpy as np
import joblib

TEST_PATH = "data/processed/cic-ids-2017_test.parquet"

def main():
    print("Loading test data...", flush=True)
    test_df = pd.read_parquet(TEST_PATH)
    X_test = test_df.drop(columns=["Attack"])

    print("Loading and predicting isolation forest...", flush=True)
    iso = joblib.load("ml-engine/models/isolation_forest_cicids.joblib")
    iso_pred = (iso.predict(X_test) == -1).astype(int)
    np.save("ml-engine/models/iso_cicids_test_preds.npy", iso_pred)
    print("Saved isolation forest predictions.", flush=True)

    print("Loading and predicting random forest...", flush=True)
    rf = joblib.load("ml-engine/models/random_forest_cicids.joblib")
    rf_pred = rf.predict(X_test)
    np.save("ml-engine/models/rf_cicids_test_preds.npy", rf_pred)
    print("Saved random forest predictions.", flush=True)

    print("Loading and predicting xgboost...", flush=True)
    xgb = joblib.load("ml-engine/models/xgboost_cicids.joblib")
    xgb_pred = xgb.predict(X_test)
    np.save("ml-engine/models/xgb_cicids_test_preds.npy", xgb_pred)
    print("Saved xgboost predictions.", flush=True)

    print("All sklearn predictions saved.", flush=True)

if __name__ == "__main__":
    main()
