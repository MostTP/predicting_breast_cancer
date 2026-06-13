"""Train an S-learner (single model with treatment indicators) using the existing SVM pipeline.

This script reuses `build_pipeline` and dataset utilities from `train_model.py`.
Saves a calibrated SVM pipeline to `breast_cancer_api/model/svm_s_learner.pkl`.
"""
from pathlib import Path
import json

import joblib
import pandas as pd
from sklearn.model_selection import train_test_split

from breast_cancer_api.ml.train_model import (
    ROOT_DIR,
    DEFAULT_DATASET,
    FEATURES,
    build_pipeline,
    create_target,
)


def main(dataset_path: Path = DEFAULT_DATASET):
    df = pd.read_csv(dataset_path)

    # create target using existing function
    y = df.apply(create_target, axis=1)

    X = df[FEATURES].copy()

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    pipeline = build_pipeline()
    pipeline.fit(X_train, y_train)

    model_dir = ROOT_DIR / "breast_cancer_api" / "model"
    model_dir.mkdir(parents=True, exist_ok=True)
    out_path = model_dir / "svm_s_learner.pkl"
    joblib.dump(pipeline, out_path)

    # quick evaluation on test set
    probs = pipeline.predict_proba(X_test)[:, 1]

    metrics = {
        "test_rows": int(len(X_test)),
        "example_mean_predicted_probability": float(probs.mean()),
    }

    with (model_dir / "metrics_s_learner.json").open("w") as f:
        json.dump(metrics, f, indent=2)

    print(f"S-learner model saved to {out_path}")


if __name__ == "__main__":
    main()
