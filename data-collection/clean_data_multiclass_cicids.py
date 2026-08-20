import os
import glob
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
import json

BASE_DIR = os.path.expanduser("~/ai-cyber-defense")
CIC_DIR = os.path.join(BASE_DIR, "data", "cic-ids-2017")
OUT_DIR = os.path.join(BASE_DIR, "data", "processed")
os.makedirs(OUT_DIR, exist_ok=True)


def clean_common(df):
    df.columns = [c.strip() for c in df.columns]
    df = df.replace([np.inf, -np.inf], np.nan)
    df = df.dropna()
    df = df.drop_duplicates()
    return df


def main():
    print("=== Loading CIC-IDS-2017 for multi-class classification ===")
    csv_files = glob.glob(os.path.join(CIC_DIR, "*.csv"))
    frames = []
    for f in csv_files:
        print(f"  Reading {os.path.basename(f)} ...")
        frames.append(pd.read_csv(f, low_memory=False))
    df = pd.concat(frames, ignore_index=True)
    print(f"Combined shape before cleaning: {df.shape}")

    df = clean_common(df)

    label_col = "Label" if "Label" in df.columns else df.columns[-1]
    df[label_col] = df[label_col].astype(str).str.strip()

    feature_cols = [c for c in df.select_dtypes(include=[np.number]).columns]

    label_encoder = LabelEncoder()
    df["AttackCat"] = label_encoder.fit_transform(df[label_col])
    class_names = list(label_encoder.classes_)
    print(f"\nClasses ({len(class_names)}): {class_names}")
    print(df[label_col].value_counts())

    X = df[feature_cols].values
    y = df["AttackCat"].values

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

    def to_df(X, y):
        out = pd.DataFrame(X, columns=feature_cols)
        out["AttackCat"] = y
        return out

    train_df, val_df, test_df = to_df(X_train, y_train), to_df(X_val, y_val), to_df(X_test, y_test)

    train_df.to_parquet(os.path.join(OUT_DIR, "cicids_multiclass_train.parquet"), index=False)
    val_df.to_parquet(os.path.join(OUT_DIR, "cicids_multiclass_val.parquet"), index=False)
    test_df.to_parquet(os.path.join(OUT_DIR, "cicids_multiclass_test.parquet"), index=False)

    with open(os.path.join(OUT_DIR, "cicids_multiclass_labels.json"), "w") as f:
        json.dump(class_names, f)

    print(f"\nSaved train/val/test parquet files + label mapping")
    print(f"Train: {train_df.shape}, Val: {val_df.shape}, Test: {test_df.shape}")


if __name__ == "__main__":
    main()
