from enum import Enum
from pydantic import BaseModel


class YesNo(str, Enum):
    Yes = "Yes"
    No = "No"


class Status(str, Enum):
    Positive = "Positive"
    Negative = "Negative"


class MenopausalState(str, Enum):
    Pre = "Pre"
    Post = "Post"


class Patient(BaseModel):
    Age_at_Diagnosis: float
    Tumor_Size: float
    Tumor_Stage: float
    Neoplasm_Histologic_Grade: float
    Lymph_nodes_examined_positive: float

    ER_Status: Status
    PR_Status: Status
    HER2_Status: Status
    Inferred_Menopausal_State: MenopausalState

    Chemotherapy: YesNo = YesNo.No
    Hormone_Therapy: YesNo = YesNo.No
    Radio_Therapy: YesNo = YesNo.No
