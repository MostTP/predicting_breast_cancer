import json
from pathlib import Path


MODEL_METADATA = {
    "model_name": "svm_breast_cancer_recommender",
    "model_family": "Support Vector Machine classifier",
    "model_variant": "S-learner treatment scoring",
    "target_name": "Effective_Treatment",
    "target_definition": (
        "Proxy outcome created in the training notebook: 1 when survival is at "
        "least 60 months, the patient is living, and relapse status indicates no relapse; "
        "0 otherwise."
    ),
    "goal_statement": (
        "Predict the most effective breast cancer treatment combination by "
        "scoring candidate regimens with a Support Vector Machine trained on outcomes."
    ),
    "prediction_meaning": (
        "Estimated probability of the proxy outcome for the supplied patient and "
        "treatment flags."
    ),
    "causal_status": "Not a causal treatment-effect model",
    "probability_status": "Requires calibration validation before clinical interpretation",
    "known_missing_clinical_factors": [
        "surgery type",
        "HER2-targeted therapy",
        "immunotherapy",
        "specific endocrine therapy regimen",
        "specific chemotherapy regimen",
        "genomic assay scores",
        "Ki-67",
        "metastatic or recurrence setting",
        "comorbidities",
        "performance status",
        "contraindications"
    ],
    "minimum_validation_needed": [
        "ROC-AUC and PR-AUC",
        "calibration curve and Brier score",
        "confusion matrix at the chosen threshold",
        "external validation dataset",
        "subgroup performance by receptor status, stage, age, and menopausal state",
        "clinical review of treatment-plausibility rules"
    ]
}


def get_model_metadata():
    metadata_path = Path(__file__).resolve().parents[1] / "model" / "model_metadata.json"
    if not metadata_path.exists():
        return MODEL_METADATA

    try:
        with metadata_path.open() as metadata_file:
            return {**MODEL_METADATA, **json.load(metadata_file)}
    except Exception:
        return MODEL_METADATA


def get_model_metrics():
    metrics_path = Path(__file__).resolve().parents[1] / "model" / "metrics.json"
    if not metrics_path.exists():
        return None

    try:
        with metrics_path.open() as metrics_file:
            return json.load(metrics_file)
    except Exception:
        return None
