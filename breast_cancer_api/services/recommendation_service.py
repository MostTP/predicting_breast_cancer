import itertools
from core.model_loader import model
from utils.preprocessing import prepare_patient_df
from services.explain_service import explain


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

        df = prepare_patient_df(temp)
        prob = model.predict_proba(df)[0, 1]

        results.append({
            "Chemotherapy": chemo,
            "Hormone_Therapy": hormone,
            "Radio_Therapy": radio,
            "success_probability": float(prob)
        })

    results = sorted(results, key=lambda x: x["success_probability"], reverse=True)

    best = results[0]
    # add explanation for the best treatment
    try:
        expl = explain({**patient_dict, "Chemotherapy": best["Chemotherapy"], "Hormone_Therapy": best["Hormone_Therapy"], "Radio_Therapy": best["Radio_Therapy"]})
        best["explanation"] = expl
    except Exception:
        best["explanation"] = {"error": "explain_failed"}

    return {
        "best_treatment": best,
        "all_options": results
    }
