"""
Shared data loading, cleaning, harmonization, and target engineering.
Handles METABRIC, TCGA, and External cohorts with different schemas.
"""

import logging
from pathlib import Path
from typing import List, Optional, Tuple

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

logger = logging.getLogger(__name__)

# Canonical column names expected by the API / model
CANONICAL_NUMERICAL = [
    "Age_at_diagnosis",
    "Tumor_Size",
    "Neoplasm_Histologic_Grade",
    "Lymph_nodes_examined_positive",
    "Gene_Expression_Feature_1",
    "Gene_Expression_Feature_2",
    "Copy_Number_Alteration_1",
]

CANONICAL_CATEGORICAL = [
    "Tumor_Stage",
    "ER_Status",
    "PR_Status",
    "HER2_Status",
    "Inferred_Menopausal_State",
    "Chemotherapy",
    "Hormone_Therapy",
    "Radio_Therapy",
]

OUTCOME_COLS = [
    "Overall_Survival_Status",
    "Overall_Survival_Months",
    "Relapse_Free_Status",
    "Relapse_Free_Status_Months",
]

ALL_CANONICAL_INPUTS = CANONICAL_NUMERICAL + CANONICAL_CATEGORICAL


# ------------------------------------------------------------------
# Cleaning
# ------------------------------------------------------------------

def _normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = [
        c.strip()
        .replace(" ", "_")
        .replace("-", "_")
        .replace("(", "")
        .replace(")", "")
        for c in df.columns
    ]
    return df


def _standardize_receptors(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    for col in ["ER_Status", "PR_Status", "HER2_Status"]:
        if col not in df.columns:
            continue
        s = df[col].fillna("Unknown").astype(str).str.strip().str.title()
        pos = {"Positive", "Pos", "+", "1", "Yes"}
        neg = {"Negative", "Neg", "-", "0", "No"}
        df[col] = s.apply(lambda x: "Positive" if x in pos else ("Negative" if x in neg else x))
    return df


def _standardize_treatments(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    for col in ["Chemotherapy", "Hormone_Therapy", "Radio_Therapy"]:
        if col not in df.columns:
            continue
        s = df[col].fillna("No").astype(str).str.strip().str.title()
        yes = {"Yes", "Y", "1", "True"}
        df[col] = s.apply(lambda x: "Yes" if x in yes else "No")
    return df


def _standardize_menopausal(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    if "Inferred_Menopausal_State" in df.columns:
        s = df["Inferred_Menopausal_State"].fillna("Unknown").astype(str).str.strip().str.lower()
        df["Inferred_Menopausal_State"] = s.apply(
            lambda x: "Pre" if "pre" in x else ("Post" if "post" in x else x.title())
        )
    return df

def _standardize_stage(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    if "Tumor_Stage" in df.columns:
        stage_map = {"I": "I", "II": "II", "III": "III", "IV": "IV",
                     "1": "I", "2": "II", "3": "III", "4": "IV"}
        s = df["Tumor_Stage"].astype(str).str.strip().str.upper()
        df["Tumor_Stage"] = s.map(stage_map).fillna(df["Tumor_Stage"])
    return df


def clean_metabric(df: pd.DataFrame) -> pd.DataFrame:
    """Clean METABRIC-specific quirks."""
    df = _normalize_columns(df)
    df = _standardize_receptors(df)
    df = _standardize_treatments(df)
    df = _standardize_menopausal(df)
    df = _standardize_stage(df)

    for col in ["Age_at_diagnosis", "Tumor_Size", "Neoplasm_Histologic_Grade",
                "Lymph_nodes_examined_positive", "Overall_Survival_Months"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    for col in ["Overall_Survival_Status", "Relapse_Free_Status"]:
        if col in df.columns:
            df[col] = df[col].astype(str).str.strip().str.lower()

    before = len(df)
    df = df.drop_duplicates()
    if len(df) < before:
        logger.info(f"METABRIC: dropped {before - len(df)} duplicates")

    return df


def clean_external(df: pd.DataFrame, cohort_name: str = "External") -> pd.DataFrame:
    """Clean TCGA or Yau/Vijver data. Handles alternate column names."""
    df = _normalize_columns(df)

    aliases = {
        "Age": "Age_at_diagnosis",
        "Age_At_Diagnosis": "Age_at_diagnosis",
        "Pathologic_Stage": "Tumor_Stage",
        "Stage": "Tumor_Stage",
        "Clinical_Stage": "Tumor_Stage",
        "Histologic_Grade": "Neoplasm_Histologic_Grade",
        "Grade": "Neoplasm_Histologic_Grade",
        "Lymph_Nodes_Positive": "Lymph_nodes_examined_positive",
        "Node_Status": "Lymph_nodes_examined_positive",
        "Er_Status": "ER_Status",
        "Pr_Status": "PR_Status",
        "Her2_Status": "HER2_Status",
        "Menopausal_Status": "Inferred_Menopausal_State",
        "Menopausal_State": "Inferred_Menopausal_State",
    }
    rename_map = {k: v for k, v in aliases.items() if k in df.columns and v not in df.columns}
    if rename_map:
        df = df.rename(columns=rename_map)
        logger.info(f"{cohort_name}: renamed columns {list(rename_map.keys())}")

    df = _standardize_receptors(df)
    df = _standardize_treatments(df)
    df = _standardize_menopausal(df)
    df = _standardize_stage(df)

    for col in ["Age_at_diagnosis", "Tumor_Size", "Neoplasm_Histologic_Grade",
                "Lymph_nodes_examined_positive", "Overall_Survival_Months"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    for col in ["Overall_Survival_Status", "Relapse_Free_Status"]:
        if col in df.columns:
            df[col] = df[col].astype(str).str.strip().str.lower()

    before = len(df)
    df = df.drop_duplicates()
    if len(df) < before:
        logger.info(f"{cohort_name}: dropped {before - len(df)} duplicates")

    return df


# ------------------------------------------------------------------
# Target Engineering
# ------------------------------------------------------------------

def engineer_target(df: pd.DataFrame, min_months: int = 60, require_relapse: bool = True) -> pd.Series:
    """
    Effective_Treatment = 1 IF:
      Overall_Survival_Months >= 60
      AND Overall_Survival_Status indicates Living
      AND (Relapse_Free_Status indicates No Relapse, if available)
    
    Args:
        df: DataFrame with outcome columns.
        min_months: Threshold for long survival.
        require_relapse: If False, skips relapse check (for TCGA/external cohorts
                        that don't have clean relapse data).
    """
    os_months = pd.to_numeric(df.get("Overall_Survival_Months"), errors="coerce")
    os_status = df.get("Overall_Survival_Status", pd.Series(np.nan, index=df.index)).astype(str).str.strip().str.lower()
    rf_status = df.get("Relapse_Free_Status", pd.Series(np.nan, index=df.index)).astype(str).str.strip().str.lower()

    is_living = os_status.str.contains(r"living|alive", na=False) | (os_status == "0")
    long_survival = os_months >= min_months

    # Check if relapse data is actually available and not all missing
    has_relapse = (
        "Relapse_Free_Status" in df.columns 
        and df["Relapse_Free_Status"].notna().any()
    )

    if require_relapse and has_relapse:
        rf_status = df["Relapse_Free_Status"].astype(str).str.strip().str.lower()
        no_relapse = ~rf_status.str.contains(r"relapse|recurrence|yes|1", na=False) | (rf_status == "0")
        effective = is_living & no_relapse & long_survival
        logger.info("Using full target (survival + relapse)")
    else:
        effective = is_living & long_survival
        if not has_relapse:
            logger.info("Relapse data unavailable — using survival-only proxy target")
        else:
            logger.info("Using survival-only target (require_relapse=False)")

    effective = effective.astype(int)
    n_pos = int(effective.sum())
    logger.info(f"Target: {n_pos}/{len(effective)} effective ({effective.mean()*100:.1f}%)")
    return effective


# ------------------------------------------------------------------
# Harmonization
# ------------------------------------------------------------------
def harmonize_for_model(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    for col in ALL_CANONICAL_INPUTS:
        if col not in df.columns:
            df[col] = np.nan

    # Drop outcome columns ONLY (not the target)
    for col in OUTCOME_COLS:
        if col in df.columns:
            df = df.drop(columns=[col])

    # Convert categoricals to string (fix mixed float/str for sklearn)
    for col in CANONICAL_CATEGORICAL:
        if col in df.columns:
            df[col] = df[col].astype(str).replace("nan", "Missing").replace("None", "Missing")

    # Keep canonical inputs + target
    keep = [c for c in ALL_CANONICAL_INPUTS if c in df.columns]
    if "Effective_Treatment" in df.columns:
        keep.append("Effective_Treatment")

    return df[keep]


# ------------------------------------------------------------------
# Main Loader
# ------------------------------------------------------------------

def load_all_data(
    metabric_path: Path,
    tcga_path: Optional[Path] = None,
    external_path: Optional[Path] = None,
    random_state: int = 42,
) -> Tuple[pd.DataFrame, pd.DataFrame, Optional[pd.DataFrame]]:
    """
    Load and prepare datasets.

    Returns:
        train_df: METABRIC train + TCGA (if provided) — used for fitting
        val_df:   METABRIC validation — used for calibration & early stopping
        test_df:  External test set (if provided) — used for final evaluation only
    """
    logger.info("Loading METABRIC...")
    df_meta = clean_metabric(pd.read_csv(metabric_path))
    df_meta["Effective_Treatment"] = engineer_target(df_meta, require_relapse=True)

    # METABRIC train/val split (stratified)
    train_meta, val_meta = train_test_split(
        df_meta,
        test_size=0.15,
        random_state=random_state,
        stratify=df_meta["Effective_Treatment"],
    )
    logger.info(f"METABRIC split: train={len(train_meta)}, val={len(val_meta)}")

    train_frames = [train_meta]

    if tcga_path and tcga_path.exists():
        logger.info("Loading TCGA...")
        df_tcga = clean_external(pd.read_csv(tcga_path), "TCGA")
        # TCGA uses survival-only target (no relapse data)
        df_tcga["Effective_Treatment"] = engineer_target(df_tcga, require_relapse=False)
        train_frames.append(df_tcga)
    else:
        logger.info("TCGA not provided or not found — skipping.")

    train_df = pd.concat(train_frames, ignore_index=True)
    train_df = harmonize_for_model(train_df)

    val_df = harmonize_for_model(val_meta)

    test_df = None
    if external_path and external_path.exists():
        logger.info("Loading External test set...")
        df_ext = clean_external(pd.read_csv(external_path), "External")
        df_ext["Effective_Treatment"] = engineer_target(df_ext, require_relapse=False)
        test_df = harmonize_for_model(df_ext)
    else:
        logger.info("External test set not provided — final evaluation will use METABRIC val only.")

    return train_df, val_df, test_df