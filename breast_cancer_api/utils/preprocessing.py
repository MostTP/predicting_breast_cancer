import pandas as pd


def _rename_columns(df: pd.DataFrame) -> pd.DataFrame:
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


def prepare_patient_df(patient_dict: dict) -> pd.DataFrame:
    df = pd.DataFrame([patient_dict])
    return _rename_columns(df)
