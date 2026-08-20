import numpy as np
import pandas as pd
import joblib
import shap
import json

TEST_PATH = "data/processed/cic-ids-2017_test.parquet"
MODEL_PATH = "ml-engine/models/xgboost_cicids.joblib"
OUT_PATH = "ml-engine/models/shap_explanations_cicids.json"

N_SAMPLES_TO_EXPLAIN = 20


def main():
    print("Loading model and test data...", flush=True)
    model = joblib.load(MODEL_PATH)
    test_df = pd.read_parquet(TEST_PATH)
    y_test = test_df["Attack"].values
    X_test = test_df.drop(columns=["Attack"])
    feature_names = X_test.columns.tolist()

    print("Building SHAP TreeExplainer...", flush=True)
    explainer = shap.TreeExplainer(model)

    attack_idx = np.where(y_test == 1)[0][:N_SAMPLES_TO_EXPLAIN // 2]
    benign_idx = np.where(y_test == 0)[0][:N_SAMPLES_TO_EXPLAIN // 2]
    sample_idx = np.concatenate([attack_idx, benign_idx])
    X_sample = X_test.iloc[sample_idx]

    print(f"Computing SHAP values for {len(sample_idx)} sample incidents...", flush=True)
    shap_values = explainer.shap_values(X_sample)

    explanations = []
    for row_pos, orig_idx in enumerate(sample_idx):
        row_shap = shap_values[row_pos]
        top5_idx = np.argsort(np.abs(row_shap))[::-1][:5]
        top5 = [
            {
                "feature": feature_names[i],
                "value": float(X_sample.iloc[row_pos, i]),
                "shap_contribution": float(row_shap[i]),
                "direction": "pushes toward ATTACK" if row_shap[i] > 0 else "pushes toward BENIGN",
            }
            for i in top5_idx
        ]
        explanations.append({
            "test_row_index": int(orig_idx),
            "true_label": "attack" if y_test[orig_idx] == 1 else "benign",
            "base_value": float(explainer.expected_value),
            "top_5_contributing_features": top5,
        })

    with open(OUT_PATH, "w") as f:
        json.dump(explanations, f, indent=2)
    print(f"\nSaved {len(explanations)} SHAP explanations to {OUT_PATH}", flush=True)

    print("\n=== Example explanation (first attack case) ===")
    example = next(e for e in explanations if e["true_label"] == "attack")
    print(f"Test row {example['test_row_index']} — true label: {example['true_label']}")
    print(f"Base value (model's average output): {example['base_value']:.4f}")
    print("Top 5 features driving this decision:")
    for feat in example["top_5_contributing_features"]:
        print(f"  {feat['feature']:20s} value={feat['value']:.3f}  "
              f"contribution={feat['shap_contribution']:+.4f}  ({feat['direction']})")


if __name__ == "__main__":
    main()
