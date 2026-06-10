import pandas as pd
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

def predict(patient_dict: dict):

    df = pd.DataFrame([patient_dict])
    df = _rename_columns(df)

    prob = model.predict_proba(df)[0, 1]
    pred = int(prob >= 0.3)

    return {
        "probability": float(prob),
        "prediction": pred
    }