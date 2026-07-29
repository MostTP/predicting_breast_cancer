"""Central configuration for the breast cancer treatment SVM project."""

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Any


# -----------------------------------------------------------------------------
# Paths
# -----------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).parent.resolve()
DATA_DIR = PROJECT_ROOT / "breast_cancer_treatment" / "data"
MODELS_DIR = PROJECT_ROOT / "models"
DISSERTATION_DIR = PROJECT_ROOT / "dissertation"
FIGURES_DIR = DISSERTATION_DIR / "figures"

for d in [MODELS_DIR, FIGURES_DIR]:
    d.mkdir(parents=True, exist_ok=True)


# -----------------------------------------------------------------------------
# Data Files
# -----------------------------------------------------------------------------
METABRIC_PATH = DATA_DIR / "metabric" / "METABRIC_RNA_Mutation.csv"
TCGA_PATH = DATA_DIR / "tcga" / "tcga_brca_clinical.csv"
EXTERNAL_PATH = DATA_DIR / "external" / "yau_vijver.csv"


# -----------------------------------------------------------------------------
# Feature Definitions
# -----------------------------------------------------------------------------
CLINICAL_FEATURES: List[str] = [
    "Age_at_diagnosis",
    "Tumor_Size",
    "Tumor_Stage",
    "Neoplasm_Histologic_Grade",
    "Lymph_nodes_examined_positive",
    "ER_Status",
    "PR_Status",
    "HER2_Status",
    "Inferred_Menopausal_State",
]

OPTIONAL_MOLECULAR_FEATURES: List[str] = [
    "Gene_Expression_Feature_1",
    "Gene_Expression_Feature_2",
    "Copy_Number_Alteration_1",
]

TREATMENT_FEATURES: List[str] = [
    "Chemotherapy",
    "Hormone_Therapy",
    "Radio_Therapy",
]

OUTCOME_FEATURES: List[str] = [
    "Overall_Survival_Status",
    "Overall_Survival_Months",
    "Relapse_Free_Status",
    "Relapse_Free_Months",
]

ALL_INPUT_FEATURES: List[str] = CLINICAL_FEATURES + OPTIONAL_MOLECULAR_FEATURES + TREATMENT_FEATURES

NUMERICAL_FEATURES: List[str] = [
    "Age_at_diagnosis",
    "Tumor_Size",
    "Neoplasm_Histologic_Grade",
    "Lymph_nodes_examined_positive",
    "Gene_Expression_Feature_1",
    "Gene_Expression_Feature_2",
    "Copy_Number_Alteration_1",
]

CATEGORICAL_FEATURES: List[str] = [
    "Tumor_Stage",
    "ER_Status",
    "PR_Status",
    "HER2_Status",
    "Inferred_Menopausal_State",
    "Chemotherapy",
    "Hormone_Therapy",
    "Radio_Therapy",
]


# -----------------------------------------------------------------------------
# Target Engineering
# -----------------------------------------------------------------------------
MIN_SURVIVAL_MONTHS: int = 60


# -----------------------------------------------------------------------------
# Model Configuration
# -----------------------------------------------------------------------------
RANDOM_STATE: int = 42
N_JOBS: int = -1
CV_FOLDS: int = 5
SCORING: str = "roc_auc"

SVM_PARAM_GRID: Dict[str, List[Any]] = field(default_factory=lambda: {
    "classifier__kernel": ["rbf", "poly", "sigmoid"],
    "classifier__C": [0.1, 1.0, 10.0, 100.0],
    "classifier__gamma": ["scale", "auto", 0.001, 0.01, 0.1],
    "classifier__degree": [2, 3, 4],
    "classifier__class_weight": ["balanced", None],
})

LR_PARAM_GRID: Dict[str, List[Any]] = field(default_factory=lambda: {
    "classifier__C": [0.01, 0.1, 1.0, 10.0, 100.0],
    "classifier__penalty": ["l1", "l2"],
    "classifier__solver": ["liblinear", "saga"],
    "classifier__class_weight": ["balanced", None],
})

RF_PARAM_GRID: Dict[str, List[Any]] = field(default_factory=lambda: {
    "classifier__n_estimators": [100, 200, 500],
    "classifier__max_depth": [5, 10, 20, None],
    "classifier__min_samples_split": [2, 5, 10],
    "classifier__class_weight": ["balanced", "balanced_subsample", None],
})

XGB_PARAM_GRID: Dict[str, List[Any]] = field(default_factory=lambda: {
    "classifier__n_estimators": [100, 200, 500],
    "classifier__max_depth": [3, 5, 7, 10],
    "classifier__learning_rate": [0.01, 0.1, 0.3],
    "classifier__subsample": [0.8, 1.0],
    "classifier__colsample_bytree": [0.8, 1.0],
})


# -----------------------------------------------------------------------------
# Treatment Combinations
# -----------------------------------------------------------------------------
TREATMENT_COMBINATIONS: List[List[str]] = [
    [],                                          # No Treatment
    ["Chemotherapy"],                            # Chemotherapy
    ["Hormone_Therapy"],                         # Hormone Therapy
    ["Radio_Therapy"],                           # Radiotherapy
    ["Chemotherapy", "Hormone_Therapy"],         # Chemo + Hormone
    ["Chemotherapy", "Radio_Therapy"],           # Chemo + Radio
    ["Hormone_Therapy", "Radio_Therapy"],        # Hormone + Radio
    ["Chemotherapy", "Hormone_Therapy", "Radio_Therapy"],  # All three
]

TREATMENT_LABELS: List[str] = [
    "No Treatment",
    "Chemotherapy",
    "Hormone Therapy",
    "Radiotherapy",
    "Chemotherapy + Hormone Therapy",
    "Chemotherapy + Radiotherapy",
    "Hormone Therapy + Radiotherapy",
    "Chemotherapy + Hormone Therapy + Radiotherapy",
]