from pydantic import BaseModel

class Patient(BaseModel):
    Age_at_Diagnosis: float
    Tumor_Size: float
    Tumor_Stage: float
    Neoplasm_Histologic_Grade: float
    Lymph_nodes_examined_positive: float

    ER_Status: str
    PR_Status: str
    HER2_Status: str
    Inferred_Menopausal_State: str

    Chemotherapy: str = "No"
    Hormone_Therapy: str = "No"
    Radio_Therapy: str = "No"