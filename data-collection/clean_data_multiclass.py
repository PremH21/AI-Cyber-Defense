import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler

BASE_DIR = os.path.expanduser("~/ai-cyber-defense")
UNSW_DIR = os.path.join(BASE_DIR, "data", "unsw-nb15")
OUT_DIR = os.path.join(BASE_DIR, "data", "processed")
os.makedirs(OUT_DIR, exist_ok=True)

def clean_common(df):
    df.columns = [c.strip() for c in df.columns]
    df = df.replace([np.inf, -np.inf], np.nan)
    df = df.dropna()
    df = df.drop_duplicates()
    return df

def main():
    print("=== Loading UNSW-NB15 for multi-class classification ===")
    train_path = os.path.join(UNSW_DIR, "UNSW_NB15_training-set.parquet")
    test_path = os.path.join(UNSW_DIR, "UNSW_NB15_testing-set.parquet")

    df_train = pd.read_parquet(train_path)
    df_test = pd.read_parquet(test_path)
    df = pd.concat([df_train, df_test], ignore_index=True)
    print(f"Combined shape before cleaning: {df.shape}")

    df = clean_common(df)

    # Normalize category labels
    df["attack_cat"] = df["attack_cat"].astype(str).str.strip()
    df.loc[df["attack_cat"].str.lower() == "normal", "attack_cat"] = "Normal"

    drop_cols = [c for c in ["label", "Label", "id"] if c in df.columns]
    df = df.drop(columns=drop_cols)

    cat_cols = [c for c in df.select_dtypes(exclude=[np.number]).columns if c != "attack_cat"]
    for c in cat_cols:
        df[c] = LabelEncoder().fit_transform(df[c].astype(str))

    label_encoder = LabelEncoder()
    df["AttackCat"] = label_encoder.fit_transform(df["attack_cat"])
    class_names = list(label_encoder.classes_)
    print(f"\nClasses ({len(class_names)}): {class_names}")
    print(df["attack_cat"].value_counts())

    feature_cols = [c for c in df.columns if c not in ["attack_cat", "AttackCat"]]

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

    train_df.to_parquet(os.path.join(OUT_DIR, "unsw-nb15_multiclass_train.parquet"), index=False)
    val_df.to_parquet(os.path.join(OUT_DIR, "unsw-nb15_multiclass_val.parquet"), index=False)
    test_df.to_parquet(os.path.join(OUT_DIR, "unsw-nb15_multiclass_test.parquet"), index=False)

    import json
    with open(os.path.join(OUT_DIR, "unsw-nb15_multiclass_labels.json"), "w") as f:
        json.dump(class_names, f)

    print(f"\nSaved train/val/test parquet files + label mapping")
    print(f"Train: {train_df.shape}, Val: {val_df.shape}, Test: {test_df.shape}")

if __name__ == "__main__":
    main()
