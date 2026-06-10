import pandas as pd
import itertools
from core.model_loader import model

def _rename_columns(df):
    """Convert underscore column names to space-separated names for model compatibility."""
    rename_map = {
        'Age_at_Diagnosis': 'Age at Diagnosis',
        'Tumor_Size': 'Tumor Size',
        'Tumor_Stage': 'Tumor Stage',
        'Neoplasm_Histologic_Grade': 'Neoplasm Histologic Grade',
        'Lymph_nodes_examined_positive': 'Lymph nodes examined positive',
        'ER_Status': 'ER Status',
        'PR_Status': 'PR Status',
        'HER2_Status': 'HER2 Status',
        'Inferred_Menopausal_State': 'Inferred Menopausal State',
        'Chemotherapy': 'Chemotherapy',
        'Hormone_Therapy': 'Hormone Therapy',
        'Radio_Therapy': 'Radio Therapy'
    }
    return df.rename(columns=rename_map)

def recommend(patient_dict: dict):

    combinations = list(
        itertools.product(["Yes", "No"], ["Yes", "No"], ["Yes", "No"])
    )

    results = []

    for chemo, hormone, radio in combinations:

        temp = patient_dict.copy()
        temp["Chemotherapy"] = chemo
        temp["Hormone_Therapy"] = hormone
        temp["Radio_Therapy"] = radio

        df = pd.DataFrame([temp])
        df = _rename_columns(df)
        prob = model.predict_proba(df)[0, 1]

        results.append({
            "Chemotherapy": chemo,
            "Hormone_Therapy": hormone,
            "Radio_Therapy": radio,
            "success_probability": float(prob)
        })

    results = sorted(results, key=lambda x: x["success_probability"], reverse=True)

    return {
        "best_treatment": results[0],
        "all_options": results
    }