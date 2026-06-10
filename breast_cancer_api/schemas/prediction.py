from pydantic import BaseModel

class PredictionResponse(BaseModel):
    probability: float
    prediction: int