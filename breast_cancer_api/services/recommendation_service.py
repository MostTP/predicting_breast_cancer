import itertools
from core.model_loader import model
from core.model_metadata import get_model_metadata
from utils.preprocessing import prepare_patient_df
from services.explain_service import explain
from services.disclaimer_service import get_disclaimer


def _is_valid_combination(patient: dict, chemo: str, hormone: str, radio: str):
    """Apply conservative clinical plausibility checks.

    These rules are guardrails for unrealistic combinations. They are not a
    substitute for guideline-driven treatment planning.
    """
    er = patient.get("ER_Status")
    pr = patient.get("PR_Status")
    her2 = patient.get("HER2_Status")
    tumor_stage = patient.get("Tumor_Stage")
    tumor_size = patient.get("Tumor_Size")
    positive_nodes = patient.get("Lymph_nodes_examined_positive")

    if hormone == "Yes":
        if er != "Positive" and pr != "Positive":
            return False, "Hormone therapy unlikely for ER-/PR- patients"

    if hormone == "No" and (er == "Positive" or pr == "Positive"):
        return False, "Hormone receptor positive patients usually need endocrine therapy considered"

    try:
        if tumor_stage is not None:
            ts = float(tumor_stage)
            if ts >= 3 and chemo == "No":
                return False, "High tumor stage usually warrants chemotherapy"
            if ts >= 2 and radio == "No":
                return False, "Stage II+ disease often requires radiation to be considered"
    except Exception:
        pass

    if her2 == "Positive" and chemo == "No":
        return False, "HER2+ patients commonly receive systemic treatment; HER2-targeted therapy is not modeled"

    try:
        if tumor_size is not None and float(tumor_size) >= 50 and chemo == "No":
            return False, "Large tumors commonly require systemic therapy to be considered"
    except Exception:
        pass

    try:
        if positive_nodes is not None and float(positive_nodes) > 0 and radio == "No":
            return False, "Node-positive disease often requires radiation to be considered"
    except Exception:
        pass

    return True, None


def _confidence_label(results):
    if len(results) < 2:
        return {
            "level": "low",
            "reason": "Only one clinically plausible option remained after filtering."
        }

    gap = (
        results[0]["estimated_outcome_probability"]
        - results[1]["estimated_outcome_probability"]
    )

    if gap < 0.05:
        return {
            "level": "low",
            "reason": "The top two model scores are very close; options should be reviewed clinically."
        }

    if gap < 0.15:
        return {
            "level": "moderate",
            "reason": "The top model score is separated from the next option, but not decisively."
        }

    return {
        "level": "higher",
        "reason": "The top model score is meaningfully higher than the next ranked option."
    }


def _recommendation_warnings():
    return [
        "This model estimates an outcome proxy, not a causal treatment effect.",
        "HER2-targeted therapy, surgery, immunotherapy, genomic assays, comorbidities, and contraindications are not modeled.",
        "Use rankings for research or decision support review only, not direct clinical treatment selection."
    ]


def recommend(patient_dict: dict):
    combinations = list(
        itertools.product(["Yes", "No"], ["Yes", "No"], ["Yes", "No"])
    )

    results = []
    skipped = []

    for chemo, hormone, radio in combinations:

        temp = patient_dict.copy()
        temp["Chemotherapy"] = chemo
        temp["Hormone_Therapy"] = hormone
        temp["Radio_Therapy"] = radio

        valid, reason = _is_valid_combination(temp, chemo, hormone, radio)
        if not valid:
            skipped.append({
                "Chemotherapy": chemo,
                "Hormone_Therapy": hormone,
                "Radio_Therapy": radio,
                "reason": reason
            })
            continue

        df = prepare_patient_df(temp)
        prob = model.predict_proba(df)[0, 1]

        results.append({
            "Chemotherapy": chemo,
            "Hormone_Therapy": hormone,
            "Radio_Therapy": radio,
            "estimated_outcome_probability": float(prob),
            "model_score": float(prob),
            # Backward-compatible alias. Prefer estimated_outcome_probability.
            "success_probability": float(prob)
        })

    if not results:
        return {
            "error": "no_valid_treatment_combinations",
            "message": "No treatment combinations passed clinical constraints.",
            "skipped_combinations": skipped,
            "warnings": _recommendation_warnings(),
            "model_metadata": get_model_metadata(),
            "disclaimer": get_disclaimer()
        }

    results = sorted(
        results,
        key=lambda x: x["estimated_outcome_probability"],
        reverse=True
    )

    top = results[0]
    try:
        expl = explain({
            **patient_dict,
            "Chemotherapy": top["Chemotherapy"],
            "Hormone_Therapy": top["Hormone_Therapy"],
            "Radio_Therapy": top["Radio_Therapy"]
        })
        top["explanation"] = expl
    except Exception:
        top["explanation"] = {"error": "explain_failed"}

    return {
        "top_ranked_option": top,
        "ranked_treatment_options": results,
        "confidence": _confidence_label(results),
        "skipped_combinations": skipped,
        "warnings": _recommendation_warnings(),
        "model_metadata": get_model_metadata(),
        # Backward-compatible aliases. Prefer top_ranked_option and ranked_treatment_options.
        "best_treatment": top,
        "all_options": results,
        "disclaimer": get_disclaimer()
    }
