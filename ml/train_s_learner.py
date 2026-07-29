"""
Train the primary S-learner (single SVM with treatment flags as features).
Calibrates probabilities and saves to the exact paths the API expects.
"""

import json
import logging
import pickle
from pathlib import Path
from typing import Optional, Tuple

import pandas as pd
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC
from sklearn.calibration import CalibratedClassifierCV
from sklearn.model_selection import StratifiedKFold, RandomizedSearchCV
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, average_precision_score,
)

from breast_cancer_api.utils.preprocessing import build_preprocessing_pipeline
from .data_utils import CANONICAL_NUMERICAL, CANONICAL_CATEGORICAL

logger = logging.getLogger(__name__)

OUTPUT_DIR = Path("breast_cancer_api/model")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

MODEL_PATH = OUTPUT_DIR / "svm_breast_cancer_recommender.pkl"
S_LEARNER_PATH = OUTPUT_DIR / "svm_s_learner.pkl"
METADATA_PATH = OUTPUT_DIR / "model_metadat.json"
METRICS_PATH = OUTPUT_DIR / "metrics.json"
METRICS_S_PATH = OUTPUT_DIR / "metrics_s_learner.json"
SCHEMA_PATH = OUTPUT_DIR / "feature_schema.json"

RANDOM_STATE = 42
CV_FOLDS = 5

# =====================================================================
# CHANGED: prefix with classifier__estimator__ because CalibratedClassifierCV
# wraps the SVC. Also removed probability param (not needed).
# =====================================================================
PARAM_GRID = {
    "classifier__estimator__kernel": ["rbf", "poly", "sigmoid"],
    "classifier__estimator__C": [0.1, 1.0, 10.0, 100.0],
    "classifier__estimator__gamma": ["scale", "auto", 0.001, 0.01, 0.1],
    "classifier__estimator__degree": [2, 3, 4],
    "classifier__estimator__class_weight": ["balanced", None],
}


def _get_feature_lists(df: pd.DataFrame):
    num = [c for c in CANONICAL_NUMERICAL if c in df.columns]
    cat = [c for c in CANONICAL_CATEGORICAL if c in df.columns]
    return num, cat


def _compute_metrics(y_true, y_prob) -> dict:
    y_pred = (y_prob >= 0.5).astype(int)
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_true, y_prob)),
        "pr_auc": float(average_precision_score(y_true, y_prob)),
    }


def train(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: Optional[pd.DataFrame] = None,
):
    logger.info("=" * 60)
    logger.info("TRAINING S-LEARNER")
    logger.info("=" * 60)

    X_train = train_df.drop(columns=["Effective_Treatment"])
    y_train = train_df["Effective_Treatment"]
    X_val = val_df.drop(columns=["Effective_Treatment"])
    y_val = val_df["Effective_Treatment"]

    num_features, cat_features = _get_feature_lists(X_train)
    logger.info(f"Features: {len(num_features)} numerical, {len(cat_features)} categorical")

    schema = {
        "numerical_features": num_features,
        "categorical_features": cat_features,
        "all_features": list(X_train.columns),
    }
    with open(SCHEMA_PATH, "w") as f:
        json.dump(schema, f, indent=2)

    preprocessing = build_preprocessing_pipeline(num_features, cat_features, k_best="all")

    # =====================================================================
    # CHANGED: Wrap SVC in CalibratedClassifierCV(ensemble=False).
    # No more probability=True. Calibration happens automatically during fit.
    # cv=3 keeps runtime sane (3 calibration folds vs 5).
    # =====================================================================
    base_svc = SVC(random_state=RANDOM_STATE)
    calibrated_svc = CalibratedClassifierCV(
        base_svc, method="sigmoid", ensemble=False, cv=3
    )

    pipe = Pipeline([
        ("preprocessing", preprocessing),
        ("classifier", calibrated_svc),
    ])

    cv = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_STATE)
    search = RandomizedSearchCV(
        pipe, PARAM_GRID, n_iter=20, cv=cv,
        scoring="roc_auc", n_jobs=-1, verbose=1,
        random_state=RANDOM_STATE, return_train_score=True,
    )
    search.fit(X_train, y_train)
    logger.info(f"Best CV ROC-AUC: {search.best_score_:.4f}")
    logger.info(f"Best params: {search.best_params_}")

    # =====================================================================
    # REMOVED: Manual calibration on validation set.
    # CalibratedClassifierCV already calibrated during RandomizedSearchCV.fit().
    # search.best_estimator_ is the final, calibrated model.
    # =====================================================================
    final_model = search.best_estimator_

    # Evaluate
    metrics = {}
    for name, X, y in [("train", X_train, y_train), ("val", X_val, y_val)]:
        prob = final_model.predict_proba(X)[:, 1]
        metrics[name] = _compute_metrics(y, prob)
        logger.info(f"{name} metrics: {metrics[name]}")

    if test_df is not None:
        X_test = test_df.drop(columns=["Effective_Treatment"])
        y_test = test_df["Effective_Treatment"]
        prob = final_model.predict_proba(X_test)[:, 1]
        metrics["test"] = _compute_metrics(y_test, prob)
        logger.info(f"test metrics: {metrics['test']}")

    with open(METRICS_PATH, "w") as f:
        json.dump(metrics, f, indent=2)
    with open(METRICS_S_PATH, "w") as f:
        json.dump(metrics, f, indent=2)

    metadata = {
        "model_type": "SVM S-Learner (Calibrated)",
        "best_params": search.best_params_,
        "features": schema,
        "training_samples": len(X_train),
        "target": "Effective_Treatment (OS>=60mo, Living, No Relapse)",
    }
    with open(METADATA_PATH, "w") as f:
        json.dump(metadata, f, indent=2, default=str)

    with open(MODEL_PATH, "wb") as f:
        pickle.dump(final_model, f)
    with open(S_LEARNER_PATH, "wb") as f:
        pickle.dump(final_model, f)

    logger.info(f"Saved model to {MODEL_PATH}")
    logger.info(f"Saved metadata to {METADATA_PATH}")
    logger.info("S-Learner training complete.")
    return final_model, metrics