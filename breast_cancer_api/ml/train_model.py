import argparse
import itertools
import json
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV, calibration_curve
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    brier_score_loss,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.svm import SVC


ROOT_DIR = Path(__file__).resolve().parents[2]
API_MODEL_DIR = ROOT_DIR / "breast_cancer_api" / "model"
DEFAULT_DATASET = ROOT_DIR / "Breast_Cancer_METABRIC.csv"
DEFAULT_THRESHOLD = 0.3

FEATURES = [
    "Age at Diagnosis",
    "Tumor Size",
    "Tumor Stage",
    "Neoplasm Histologic Grade",
    "Lymph nodes examined positive",
    "ER Status",
    "PR Status",
    "HER2 Status",
    "Inferred Menopausal State",
    "Chemotherapy",
    "Hormone Therapy",
    "Radio Therapy",
]

NUMERIC_FEATURES = [
    "Age at Diagnosis",
    "Tumor Size",
    "Tumor Stage",
    "Neoplasm Histologic Grade",
    "Lymph nodes examined positive",
]

CATEGORICAL_FEATURES = [
    "ER Status",
    "PR Status",
    "HER2 Status",
    "Inferred Menopausal State",
    "Chemotherapy",
    "Hormone Therapy",
    "Radio Therapy",
]

TREATMENT_FEATURES = [
    "Chemotherapy",
    "Hormone Therapy",
    "Radio Therapy",
]

SUBGROUP_FEATURES = [
    "ER Status",
    "PR Status",
    "HER2 Status",
    "Tumor Stage",
    "Inferred Menopausal State",
]


def create_target(row):
    survival = row["Overall Survival (Months)"]
    alive = str(row["Patient's Vital Status"]).lower()
    relapse = str(row["Relapse Free Status"]).lower()

    if survival >= 60 and "living" in alive and "not" in relapse:
        return 1

    return 0


def build_pipeline():
    numeric_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])

    categorical_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(handle_unknown="ignore")),
    ])

    preprocessor = ColumnTransformer([
        ("num", numeric_pipeline, NUMERIC_FEATURES),
        ("cat", categorical_pipeline, CATEGORICAL_FEATURES),
    ])

    base_svm = SVC(kernel="rbf", class_weight="balanced")
    classifier = CalibratedClassifierCV(estimator=base_svm, cv=5)

    return Pipeline([
        ("preprocessor", preprocessor),
        ("classifier", classifier),
    ])


def evaluate_predictions(y_true, y_prob, threshold):
    y_pred = (y_prob >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()

    return {
        "threshold": threshold,
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "roc_auc": float(roc_auc_score(y_true, y_prob)),
        "pr_auc": float(average_precision_score(y_true, y_prob)),
        "brier_score": float(brier_score_loss(y_true, y_prob)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "confusion_matrix": {
            "true_negative": int(tn),
            "false_positive": int(fp),
            "false_negative": int(fn),
            "true_positive": int(tp),
        },
        "classification_report": classification_report(
            y_true,
            y_pred,
            output_dict=True,
            zero_division=0,
        ),
    }


def calibration_metrics(y_true, y_prob):
    prob_true, prob_pred = calibration_curve(
        y_true,
        y_prob,
        n_bins=10,
        strategy="quantile",
    )

    return [
        {
            "mean_predicted_probability": float(pred),
            "observed_positive_rate": float(true),
        }
        for true, pred in zip(prob_true, prob_pred)
    ]


def subgroup_metrics(frame, y_true, y_prob, threshold):
    output = {}
    eval_frame = frame.reset_index(drop=True).copy()
    eval_frame["_y_true"] = np.asarray(y_true)
    eval_frame["_y_prob"] = np.asarray(y_prob)

    for feature in SUBGROUP_FEATURES:
        groups = {}
        for value, group in eval_frame.groupby(feature, dropna=False):
            if len(group) < 20 or group["_y_true"].nunique() < 2:
                continue
            groups[str(value)] = evaluate_predictions(
                group["_y_true"],
                group["_y_prob"],
                threshold,
            )
            groups[str(value)]["support"] = int(len(group))
        output[feature] = groups

    return output


def recommend_treatment(patient, model):
    rows = []
    for chemo, hormone, radio in itertools.product(["Yes", "No"], repeat=3):
        patient_copy = patient.copy()
        patient_copy["Chemotherapy"] = chemo
        patient_copy["Hormone Therapy"] = hormone
        patient_copy["Radio Therapy"] = radio
        probability = model.predict_proba(pd.DataFrame([patient_copy]))[0, 1]

        rows.append({
            "Chemotherapy": chemo,
            "Hormone Therapy": hormone,
            "Radio Therapy": radio,
            "Recommended_Probability": float(probability),
        })

    return (
        pd.DataFrame(rows)
        .sort_values("Recommended_Probability", ascending=False)
        .reset_index(drop=True)
    )


def evaluate_recommendations(x_test, y_test, model):
    candidate_rows = []

    for patient_index, row in x_test.reset_index(drop=True).iterrows():
        base = row.drop(TREATMENT_FEATURES).to_dict()
        for chemo, hormone, radio in itertools.product(["Yes", "No"], repeat=3):
            candidate = base.copy()
            candidate["_patient_index"] = patient_index
            candidate["Chemotherapy"] = chemo
            candidate["Hormone Therapy"] = hormone
            candidate["Radio Therapy"] = radio
            candidate_rows.append(candidate)

    candidates = pd.DataFrame(candidate_rows)
    candidates["Recommended_Probability"] = model.predict_proba(
        candidates[FEATURES]
    )[:, 1]

    best_rows = (
        candidates.sort_values(
            ["_patient_index", "Recommended_Probability"],
            ascending=[True, False],
        )
        .groupby("_patient_index", as_index=False)
        .head(1)
        .sort_values("_patient_index")
        .reset_index(drop=True)
    )

    results = x_test.copy().reset_index(drop=True)
    results["Actual_Outcome"] = y_test.reset_index(drop=True)
    results["Recommended_Chemo"] = best_rows["Chemotherapy"].to_numpy()
    results["Recommended_Hormone"] = best_rows["Hormone Therapy"].to_numpy()
    results["Recommended_Radio"] = best_rows["Radio Therapy"].to_numpy()
    results["Recommended_Probability"] = best_rows[
        "Recommended_Probability"
    ].to_numpy()
    results["Actual_Treatment"] = (
        x_test["Chemotherapy"].reset_index(drop=True)
        + "_"
        + x_test["Hormone Therapy"].reset_index(drop=True)
        + "_"
        + x_test["Radio Therapy"].reset_index(drop=True)
    )
    results["Recommended_Treatment"] = (
        results["Recommended_Chemo"]
        + "_"
        + results["Recommended_Hormone"]
        + "_"
        + results["Recommended_Radio"]
    )

    summary = (
        results.groupby("Recommended_Treatment")["Recommended_Probability"]
        .agg(["count", "mean", "min", "max"])
        .sort_values("mean", ascending=False)
    )

    return {
        "agreement_with_recorded_treatment": float(
            (results["Actual_Treatment"] == results["Recommended_Treatment"]).mean()
        ),
        "recommended_probability": {
            "mean": float(results["Recommended_Probability"].mean()),
            "median": float(results["Recommended_Probability"].median()),
            "min": float(results["Recommended_Probability"].min()),
            "max": float(results["Recommended_Probability"].max()),
        },
        "recommended_treatment_counts": {
            key: int(value)
            for key, value in results["Recommended_Treatment"].value_counts().items()
        },
    }, results, summary


def save_json(path, payload):
    path.write_text(json.dumps(payload, indent=2, sort_keys=True))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--threshold", type=float, default=DEFAULT_THRESHOLD)
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--output-dir", type=Path, default=API_MODEL_DIR)
    parser.add_argument("--n-jobs", type=int, default=1)
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(args.dataset)
    df["Effective_Treatment"] = df.apply(create_target, axis=1)

    x = df[FEATURES]
    y = df["Effective_Treatment"]

    x_train, x_test, y_train, y_test = train_test_split(
        x,
        y,
        test_size=0.2,
        random_state=args.random_state,
        stratify=y,
    )

    pipeline = build_pipeline()
    param_grid = {
        "classifier__estimator__C": [1, 10, 100],
        "classifier__estimator__gamma": [0.001, 0.01, 0.1],
    }

    grid = GridSearchCV(
        pipeline,
        param_grid,
        scoring="roc_auc",
        cv=5,
        n_jobs=args.n_jobs,
        return_train_score=True,
    )
    grid.fit(x_train, y_train)

    best_model = grid.best_estimator_
    y_prob = best_model.predict_proba(x_test)[:, 1]

    metrics = {
        "trained_at": datetime.now(timezone.utc).isoformat(),
        "dataset_path": str(args.dataset),
        "dataset_rows": int(len(df)),
        "target_distribution": {
            str(key): int(value)
            for key, value in y.value_counts().sort_index().items()
        },
        "split": {
            "train_rows": int(len(x_train)),
            "test_rows": int(len(x_test)),
            "test_size": 0.2,
            "random_state": args.random_state,
        },
        "grid_search": {
            "scoring": "roc_auc",
            "cv": 5,
            "best_params": grid.best_params_,
            "best_cv_roc_auc": float(grid.best_score_),
        },
        "test_metrics": evaluate_predictions(y_test, y_prob, args.threshold),
        "calibration_curve": calibration_metrics(y_test, y_prob),
        "subgroup_metrics": subgroup_metrics(x_test, y_test, y_prob, args.threshold),
    }

    recommendation_metrics, rec_results, treatment_summary = evaluate_recommendations(
        x_test,
        y_test,
        best_model,
    )
    metrics["recommendation_metrics"] = recommendation_metrics

    metadata = {
        "model_name": "svm_breast_cancer_recommender",
        "model_family": "Support Vector Machine classifier",
        "target_name": "Effective_Treatment",
        "target_definition": (
            "1 when survival is at least 60 months, the patient is living, "
            "and relapse status indicates no relapse; 0 otherwise."
        ),
        "threshold": args.threshold,
        "features": FEATURES,
        "numeric_features": NUMERIC_FEATURES,
        "categorical_features": CATEGORICAL_FEATURES,
        "treatment_features": TREATMENT_FEATURES,
        "causal_status": "Not a causal treatment-effect model",
        "probability_status": "CalibratedClassifierCV used; external calibration validation still required",
        "trained_at": metrics["trained_at"],
        "metrics_file": "metrics.json",
    }

    feature_schema = {
        "features": FEATURES,
        "numeric_features": NUMERIC_FEATURES,
        "categorical_features": CATEGORICAL_FEATURES,
        "target": "Effective_Treatment",
    }

    model_path = args.output_dir / "svm_breast_cancer_recommender.pkl"
    metrics_path = args.output_dir / "metrics.json"
    metadata_path = args.output_dir / "model_metadata.json"
    schema_path = args.output_dir / "feature_schema.json"

    joblib.dump(best_model, model_path)
    save_json(metrics_path, metrics)
    save_json(metadata_path, metadata)
    save_json(schema_path, feature_schema)

    rec_results.to_csv(ROOT_DIR / "svm_treatment_recommendations.csv", index=False)
    treatment_summary.to_csv(ROOT_DIR / "treatment_summary.csv")

    print(json.dumps({
        "model_path": str(model_path),
        "metrics_path": str(metrics_path),
        "metadata_path": str(metadata_path),
        "feature_schema_path": str(schema_path),
        "best_params": grid.best_params_,
        "best_cv_roc_auc": metrics["grid_search"]["best_cv_roc_auc"],
        "test_metrics": metrics["test_metrics"],
        "recommendation_metrics": recommendation_metrics,
    }, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
