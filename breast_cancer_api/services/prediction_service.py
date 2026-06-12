from core.model_loader import model
from core.model_metadata import get_model_metadata
from utils.preprocessing import prepare_patient_df
from services.disclaimer_service import get_disclaimer

PREDICTION_THRESHOLD = 0.3


def _prediction_warnings():
    return [
        "This is an estimated probability for a proxy outcome, not a confirmed clinical success probability.",
        "The threshold has not been clinically validated in this API.",
        "Treatment-effect causality is not established by this model."
    ]


def predict(patient_dict: dict):
    df = prepare_patient_df(patient_dict)

    prob = model.predict_proba(df)[0, 1]
    pred = int(prob >= PREDICTION_THRESHOLD)

    return {
        "estimated_outcome_probability": float(prob),
        "model_score": float(prob),
        "threshold": PREDICTION_THRESHOLD,
        "probability": float(prob),
        "prediction": pred,
        "warnings": _prediction_warnings(),
        "model_metadata": get_model_metadata(),
        "disclaimer": get_disclaimer()
    }
