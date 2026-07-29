#!/usr/bin/env python3
"""
Merge GDC TCGA-BRCA TSV files into canonical schema.
Handles missing columns gracefully.
"""

import pandas as pd
import numpy as np
from pathlib import Path

RAW_DIR = Path("data/tcga_raw")
OUTPUT = Path("data/tcga/tcga_prepared.csv")

clinical = pd.read_csv(RAW_DIR / "clinical.tsv", sep="\t", low_memory=False)
clinical = clinical.drop_duplicates(subset=["cases.case_id"], keep="first")

def get_col(df, *possible_names):
    """Return first matching column or None."""
    for name in possible_names:
        if name in df.columns:
            return df[name]
    return None

# --- Core fields ---
clinical["Patient_ID"] = clinical["cases.case_id"]
clinical["Age_at_diagnosis"] = pd.to_numeric(get_col(clinical, "demographic.age_at_index", "diagnoses.age_at_diagnosis"), errors="coerce")

# Stage
stage_raw = get_col(clinical, "diagnoses.ajcc_pathologic_stage", "diagnoses.ajcc_clinical_stage")
if stage_raw is not None:
    clinical["Tumor_Stage"] = stage_raw.astype(str).str.strip()
    stage_map = {
        "stage ia": "I", "stage ib": "I", "stage i": "I",
        "stage iia": "II", "stage iib": "II", "stage ii": "II",
        "stage iiia": "III", "stage iiib": "III", "stage iiic": "III", "stage iii": "III",
        "stage iv": "IV",
    }
    clinical["Tumor_Stage"] = clinical["Tumor_Stage"].str.lower().map(stage_map)

# Grade
grade_raw = get_col(clinical, "diagnoses.tumor_grade")
if grade_raw is not None:
    clinical["Neoplasm_Histologic_Grade"] = pd.to_numeric(grade_raw, errors="coerce")

# Nodes
nodes_raw = get_col(clinical, "diagnoses.ajcc_pathologic_n", "pathology_details.lymph_nodes_positive")
if nodes_raw is not None:
    clinical["Lymph_nodes_examined_positive"] = nodes_raw.astype(str).str.extract(r"(\d+)").astype(float)

# Receptors — TCGA clinical TSV often lacks these; check all possible names
er_raw = get_col(clinical, "diagnoses.breast_carcinoma_estrogen_receptor_status", 
                 "diagnoses.er_status_by_ihc", "diagnoses.er_percent_positive_range")
if er_raw is not None:
    clinical["ER_Status"] = er_raw.astype(str).str.strip().str.title()

pr_raw = get_col(clinical, "diagnoses.progesterone_receptor_status",
                 "diagnoses.pr_status_by_ihc", "diagnoses.pr_percent_positive_range")
if pr_raw is not None:
    clinical["PR_Status"] = pr_raw.astype(str).str.strip().str.title()

her2_raw = get_col(clinical, "diagnoses.her2_immunohistochemistry_level",
                   "diagnoses.her2_status_by_ihc", "diagnoses.her2_erbb2_percent_positive_range")
if her2_raw is not None:
    clinical["HER2_Status"] = her2_raw.astype(str).str.strip().str.title()

# Survival
vital = get_col(clinical, "demographic.vital_status")
if vital is not None:
    clinical["Overall_Survival_Status"] = vital.astype(str).str.strip().str.lower().map({
        "alive": "Living", "dead": "Deceased"
    })

days_follow = get_col(clinical, "diagnoses.days_to_last_follow_up", "follow_ups.days_to_follow_up")
if days_follow is not None:
    clinical["Overall_Survival_Months"] = pd.to_numeric(days_follow, errors="coerce") / 30.44

days_death = get_col(clinical, "demographic.days_to_death")
if days_death is not None:
    death_months = pd.to_numeric(days_death, errors="coerce") / 30.44
    clinical.loc[death_months.notna(), "Overall_Survival_Months"] = death_months[death_months.notna()]

# Tumor size from pathology
pathology = None
if (RAW_DIR / "pathology_detail.tsv").exists():
    pathology = pd.read_csv(RAW_DIR / "pathology_detail.tsv", sep="\t", low_memory=False)
    pathology = pathology.drop_duplicates(subset=["cases.case_id"], keep="first")
    size_raw = get_col(pathology, "pathology_details.tumor_largest_dimension_diameter", 
                       "pathology_details.greatest_tumor_dimension", "pathology_details.tumor_size")
    if size_raw is not None:
        size_map = pathology.set_index("cases.case_id")[size_raw.name]
        clinical["Tumor_Size"] = clinical["cases.case_id"].map(size_map)

# Menopausal
clinical["Inferred_Menopausal_State"] = clinical["Age_at_diagnosis"].apply(
    lambda x: "Post" if pd.notna(x) and x >= 50 else "Pre"
)

# Relapse
follow_up = None
if (RAW_DIR / "follow_up.tsv").exists():
    follow_up = pd.read_csv(RAW_DIR / "follow_up.tsv", sep="\t", low_memory=False)
    follow_up = follow_up.drop_duplicates(subset=["cases.case_id"], keep="last")
    relapse_raw = get_col(follow_up, "follow_ups.progression_or_recurrence", "follow_ups.recurrence")
    if relapse_raw is not None:
        relapse_map = follow_up.set_index("cases.case_id")[relapse_raw.name]
        clinical["Relapse_Free_Status"] = clinical["cases.case_id"].map(relapse_map)
        clinical["Relapse_Free_Status"] = clinical["Relapse_Free_Status"].astype(str).str.strip().str.lower().map({
            "yes": "Relapse", "no": "No Relapse"
        })

# Treatments
treatment_type = get_col(clinical, "treatments.treatment_type")
if treatment_type is not None:
    t_str = treatment_type.astype(str).str.lower()
    clinical["Chemotherapy"] = t_str.str.contains("chemotherapy", na=False).map({True: "Yes", False: "No"})
    clinical["Hormone_Therapy"] = t_str.str.contains("hormone|endocrine", na=False).map({True: "Yes", False: "No"})
    clinical["Radio_Therapy"] = t_str.str.contains("radiation|radiotherapy", na=False).map({True: "Yes", False: "No"})
else:
    clinical["Chemotherapy"] = None
    clinical["Hormone_Therapy"] = None
    clinical["Radio_Therapy"] = None

# --- Clean and save ---
canonical = [
    "Patient_ID", "Age_at_diagnosis", "Tumor_Size", "Tumor_Stage",
    "Neoplasm_Histologic_Grade", "Lymph_nodes_examined_positive",
    "ER_Status", "PR_Status", "HER2_Status", "Inferred_Menopausal_State",
    "Chemotherapy", "Hormone_Therapy", "Radio_Therapy",
    "Overall_Survival_Status", "Overall_Survival_Months",
    "Relapse_Free_Status", "Relapse_Free_Months",
]
keep = [c for c in canonical if c in clinical.columns]
df = clinical[keep].copy()

OUTPUT.parent.mkdir(parents=True, exist_ok=True)
df.to_csv(OUTPUT, index=False)

print(f"Saved {len(df)} TCGA patients to {OUTPUT}")
print(f"Columns: {df.columns.tolist()}")
print(f"\nMissing values:\n{df.isnull().sum()}")
print(f"\nSample:\n{df.head()}")