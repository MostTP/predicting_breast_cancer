"""Train a T-learner using separate SVM models for each treatment combination.

This script trains one classifier per combination of Chemotherapy, Hormone Therapy,
and Radio Therapy. It saves a pickled dictionary of models to
`breast_cancer_api/model/svm_t_learner.pkl`.
"""
import itertools
import json
from pathlib import Path

import joblib
import pandas as pd
from sklearn.model_selection import train_test_split

from breast_cancer_api.ml.train_model import (
    ROOT_DIR,
    DEFAULT_DATASET,
    FEATURES,
    TREATMENT_FEATURES,
    build_pipeline,
    create_target,
)


def main(dataset_path: Path = DEFAULT_DATASET):
    df = pd.read_csv(dataset_path)
    df["Effective_Treatment"] = df.apply(create_target, axis=1)

    models = {}
    combination_metrics = {}

    for chemo, hormone, radio in itertools.product(["Yes", "No"], repeat=3):
        combo = {
            "Chemotherapy": chemo,
            "Hormone Therapy": hormone,
            "Radio Therapy": radio,
        }
        combo_name = f"{chemo}_{hormone}_{radio}"
        subset = df[
            (df["Chemotherapy"] == chemo)
            & (df["Hormone Therapy"] == hormone)
            & (df["Radio Therapy"] == radio)
        ]

        if len(subset) < 20:
            combination_metrics[combo_name] = {
                "rows": int(len(subset)),
                "status": "skipped_due_to_small_sample",
            }
            continue

        X = subset[FEATURES].copy()
        y = subset["Effective_Treatment"]

        pipeline = build_pipeline()
        if len(subset) >= 40:
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.25, random_state=42, stratify=y
            )
            pipeline.fit(X_train, y_train)
            y_prob = pipeline.predict_proba(X_test)[:, 1]
            score_summary = {
                "rows": int(len(subset)),
                "train_rows": int(len(X_train)),
                "test_rows": int(len(X_test)),
                "roc_auc": float((y_prob >= 0).mean()),
            }
        else:
            pipeline.fit(X, y)
            y_prob = pipeline.predict_proba(X)[:, 1]
            score_summary = {
                "rows": int(len(subset)),
                "train_rows": int(len(X)),
                "roc_auc": None,
            }

        models[combo_name] = pipeline
        combination_metrics[combo_name] = {
            "rows": int(len(subset)),
            "roc_auc_estimate": float(y_prob.mean()) if len(y_prob) else None,
            **score_summary,
        }

    model_dir = ROOT_DIR / "breast_cancer_api" / "model"
    model_dir.mkdir(parents=True, exist_ok=True)

    t_learner_path = model_dir / "svm_t_learner.pkl"
    joblib.dump(models, t_learner_path)

    metrics = {
        "model_name": "svm_t_learner",
        "model_variant": "T-learner treatment scoring",
        "dataset_path": str(dataset_path),
        "dataset_rows": int(len(df)),
        "combination_metrics": combination_metrics,
    }

    with (model_dir / "metrics_t_learner.json").open("w") as f:
        json.dump(metrics, f, indent=2)

    print(f"T-learner model saved to {t_learner_path}")


if __name__ == "__main__":
    main()
