from core.model_loader import model
from utils.preprocessing import prepare_patient_df


def predict(patient_dict: dict):
    df = prepare_patient_df(patient_dict)

    prob = model.predict_proba(df)[0, 1]
    pred = int(prob >= 0.3)

    return {
        "probability": float(prob),
        "prediction": pred
    }
