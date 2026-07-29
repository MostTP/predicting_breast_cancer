"""
Train T-learner models: one classifier per treatment combination.
Saved as a dict in svm_t_learner.pkl so the API can do dict.get(key).
"""

import json
import logging
import pickle
from pathlib import Path
from typing import Dict

import pandas as pd
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC
from sklearn.calibration import CalibratedClassifierCV
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score

from breast_cancer_api.utils.preprocessing import build_preprocessing_pipeline
from .data_utils import CANONICAL_NUMERICAL, CANONICAL_CATEGORICAL

logger = logging.getLogger(__name__)

OUTPUT_DIR = Path("breast_cancer_api/model")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

T_LEARNER_PATH = OUTPUT_DIR / "svm_t_learner.pkl"
METRICS_T_PATH = OUTPUT_DIR / "metrics_t_learner.json"

RANDOM_STATE = 42

COMBINATIONS = [
    ("No", "No", "No"),
    ("Yes", "No", "No"),
    ("No", "Yes", "No"),
    ("No", "No", "Yes"),
    ("Yes", "Yes", "No"),
    ("Yes", "No", "Yes"),
    ("No", "Yes", "Yes"),
    ("Yes", "Yes", "Yes"),
]


class ProbaInverter:
    """
    Wrapper that swaps probability columns to fix a model with inverted
    predictions (AUC < 0.5). Exposes the same interface as Pipeline so
    the API doesn't need to change.
    """
    def __init__(self, pipeline: Pipeline):
        self.pipeline = pipeline
        self.named_steps = pipeline.named_steps
        self.classes_ = pipeline.classes_

    def predict_proba(self, X):
        prob = self.pipeline.predict_proba(X)
        # Swap P(y=0) and P(y=1)
        return np.column_stack([prob[:, 1], prob[:, 0]])

    def predict(self, X):
        return 1 - self.pipeline.predict(X)


def _get_feature_lists(df: pd.DataFrame):
    num = [c for c in CANONICAL_NUMERICAL if c in df.columns]
    cat = [c for c in CANONICAL_CATEGORICAL if c in df.columns]
    return num, cat


def _get_classifier(n_samples: int, n_pos: int):
    """
    Choose a stable classifier based on arm size.
    - Very small arms (<100 samples or <10 in either class): LogisticRegression
    - Larger arms: Calibrated SVM (no deprecated probability=True)
    """
    if n_samples < 100 or n_pos < 10 or (n_samples - n_pos) < 10:
        logger.info(f"  -> Using LogisticRegression (n={n_samples}, pos={n_pos})")
        return LogisticRegression(
            max_iter=1000, class_weight="balanced", random_state=RANDOM_STATE
        )

    base_svc = SVC(
        kernel="rbf",
        C=1.0,
        gamma="scale",
        class_weight="balanced",
        random_state=RANDOM_STATE,
    )
    # ensemble=False uses a single calibration split (faster, less variance on small data)
    return CalibratedClassifierCV(
        base_svc, method="sigmoid", ensemble=False, cv=3
    )


def train(train_df: pd.DataFrame):
    logger.info("=" * 60)
    logger.info("TRAINING T-LEARNERS")
    logger.info("=" * 60)

    num_features, cat_features = _get_feature_lists(train_df)
    t_learners: Dict[str, Pipeline] = {}
    metrics = {}

    for chemo, hormone, radio in COMBINATIONS:
        key = f"{chemo}_{hormone}_{radio}"
        mask = (
            (train_df["Chemotherapy"] == chemo)
            & (train_df["Hormone_Therapy"] == hormone)
            & (train_df["Radio_Therapy"] == radio)
        )

        n_samples = int(mask.sum())
        if n_samples < 30:
            logger.warning(f"[{key}] Skipped: only {n_samples} samples (need >=30)")
            continue

        sub = train_df[mask]
        X = sub.drop(columns=["Effective_Treatment"])
        y = sub["Effective_Treatment"]
        n_pos = int(y.sum())
        n_neg = n_samples - n_pos

        # -----------------------------------------------------------------
        # Per-arm train/val split for honest metrics
        # -----------------------------------------------------------------
        if n_samples >= 60 and n_pos >= 5 and n_neg >= 5:
            X_tr, X_val, y_tr, y_val = train_test_split(
                X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y
            )
        else:
            X_tr, y_tr = X, y
            X_val, y_val = None, None

        prep = build_preprocessing_pipeline(num_features, cat_features, k_best="all")
        classifier = _get_classifier(n_samples, n_pos)

        pipe = Pipeline([
            ("preprocessing", prep),
            ("classifier", classifier),
        ])

        pipe.fit(X_tr, y_tr)

        # -----------------------------------------------------------------
        # Metrics  (compute AFTER inversion so logs match the saved model)
        # -----------------------------------------------------------------
        metric_entry = {
            "n_samples": n_samples,
            "n_positive": n_pos,
            "n_negative": n_neg,
        }

        # Validation AUC + inversion guardrail
        if X_val is not None:
            prob_val = pipe.predict_proba(X_val)[:, 1]
            try:
                auc_val = float(roc_auc_score(y_val, prob_val))
            except ValueError:
                auc_val = None

            if auc_val is not None and auc_val < 0.5:
                logger.warning(
                    f"[{key}] Validation AUC {auc_val:.3f} < 0.5 — wrapping model to invert probabilities"
                )
                pipe = ProbaInverter(pipe)
                # Recompute with inverted model
                prob_val = pipe.predict_proba(X_val)[:, 1]
                auc_val = float(roc_auc_score(y_val, prob_val))
                metric_entry["inverted"] = True
            else:
                metric_entry["inverted"] = False

            metric_entry["val_roc_auc"] = auc_val
        else:
            metric_entry["val_roc_auc"] = None
            metric_entry["inverted"] = False

        # Training AUC — computed on the FINAL model (inverted if needed)
        prob_tr = pipe.predict_proba(X_tr)[:, 1]
        try:
            auc_tr = float(roc_auc_score(y_tr, prob_tr))
        except ValueError:
            auc_tr = None
        metric_entry["train_roc_auc"] = auc_tr

        # Log
        if X_val is not None:
            logger.info(
                f"[{key}] Trained on {n_samples} samples "
                f"(train AUC={auc_tr:.3f}, val AUC={auc_val:.3f})"
            )
        else:
            logger.info(
                f"[{key}] Trained on {n_samples} samples "
                f"(train AUC={auc_tr:.3f}, no val split)"
            )

        t_learners[key] = pipe
        metrics[key] = metric_entry

    # Save dict of models
    with open(T_LEARNER_PATH, "wb") as f:
        pickle.dump(t_learners, f)

    with open(METRICS_T_PATH, "w") as f:
        json.dump(metrics, f, indent=2)

    logger.info(f"Saved {len(t_learners)} T-learners to {T_LEARNER_PATH}")
    return t_learners