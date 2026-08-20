import os
import glob
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from imblearn.over_sampling import SMOTE

BASE_DIR = os.path.expanduser("~/ai-cyber-defense")
CIC_DIR = os.path.join(BASE_DIR, "data", "cic-ids-2017")
UNSW_DIR = os.path.join(BASE_DIR, "data", "unsw-nb15")
OUT_DIR = os.path.join(BASE_DIR, "data", "processed")
os.makedirs(OUT_DIR, exist_ok=True)


def clean_common(df):
    df.columns = [c.strip() for c in df.columns]
    df = df.replace([np.inf, -np.inf], np.nan)
    df = df.dropna()
    df = df.drop_duplicates()
    return df


def process_cic_ids(sample_frac=1.0):
    print("\n=== Loading CIC-IDS-2017 ===")
    csv_files = glob.glob(os.path.join(CIC_DIR, "*.csv"))
    if not csv_files:
        print("No CIC-IDS-2017 CSV files found - skipping.")
        return

    frames = []
    for f in csv_files:
        print(f"  Reading {os.path.basename(f)} ...")
        chunk = pd.read_csv(f, low_memory=False)
        frames.append(chunk)
    df = pd.concat(frames, ignore_index=True)
    print(f"  Combined shape before cleaning: {df.shape}")

    df = clean_common(df)

    label_col = "Label" if "Label" in df.columns else df.columns[-1]
    df["Attack"] = (df[label_col].astype(str).str.strip().str.upper() != "BENIGN").astype(int)
    df = df.drop(columns=[label_col])

    feature_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    feature_cols = [c for c in feature_cols if c != "Attack"]
    df = df[feature_cols + ["Attack"]]

    print(f"  Cleaned shape: {df.shape}  (features: {len(feature_cols)})")
    print("  Attack distribution:")
    print(df["Attack"].value_counts())

    if sample_frac < 1.0:
        df = df.sample(frac=sample_frac, random_state=42)

    split_and_save(df, feature_cols, "cic-ids-2017")


def process_unsw():
    print("\n=== Loading UNSW-NB15 ===")
    train_path = os.path.join(UNSW_DIR, "UNSW_NB15_training-set.parquet")
    test_path = os.path.join(UNSW_DIR, "UNSW_NB15_testing-set.parquet")
    if not (os.path.exists(train_path) and os.path.exists(test_path)):
        print("UNSW-NB15 parquet files not found - skipping.")
        return

    df_train = pd.read_parquet(train_path)
    df_test = pd.read_parquet(test_path)
    df = pd.concat([df_train, df_test], ignore_index=True)
    print(f"  Combined shape before cleaning: {df.shape}")

    df = clean_common(df)

    label_col = "label" if "label" in df.columns else "Label"
    df["Attack"] = df[label_col].astype(int)
    drop_cols = [c for c in ["label", "Label", "attack_cat", "id"] if c in df.columns]
    df = df.drop(columns=drop_cols)

    cat_cols = [c for c in df.select_dtypes(exclude=[np.number]).columns if c != "Attack"]
    for c in cat_cols:
        df[c] = LabelEncoder().fit_transform(df[c].astype(str))

    feature_cols = [c for c in df.columns if c != "Attack"]
    print(f"  Cleaned shape: {df.shape}  (features: {len(feature_cols)})")
    print("  Attack distribution:")
    print(df["Attack"].value_counts())

    split_and_save(df, feature_cols, "unsw-nb15")


def split_and_save(df, feature_cols, name):
    X = df[feature_cols].values
    y = df["Attack"].values

    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=0.30, random_state=42, stratify=y
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.50, random_state=42, stratify=y_temp
    )

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_val = scaler.transform(X_val)
    X_test = scaler.transform(X_test)

    print(f"  Before SMOTE: {sum(y_train == 0)} benign, {sum(y_train == 1)} attacks")
    sm = SMOTE(random_state=42)
    X_train, y_train = sm.fit_resample(X_train, y_train)
    print(f"  After SMOTE:  {sum(y_train == 0)} benign, {sum(y_train == 1)} attacks")

    def to_df(X, y):
        out = pd.DataFrame(X, columns=feature_cols)
        out["Attack"] = y
        return out

    train_df, val_df, test_df = to_df(X_train, y_train), to_df(X_val, y_val), to_df(X_test, y_test)

    train_path = os.path.join(OUT_DIR, f"{name}_train.parquet")
    val_path = os.path.join(OUT_DIR, f"{name}_val.parquet")
    test_path = os.path.join(OUT_DIR, f"{name}_test.parquet")

    train_df.to_parquet(train_path, index=False)
    val_df.to_parquet(val_path, index=False)
    test_df.to_parquet(test_path, index=False)

    print(f"  Saved {train_path}  ({train_df.shape})")
    print(f"  Saved {val_path}  ({val_df.shape})")
    print(f"  Saved {test_path}  ({test_df.shape})")


if __name__ == "__main__":
    print("PREPROCESSING START")
    process_cic_ids()
    process_unsw()
    print("\nPREPROCESSING COMPLETE")
