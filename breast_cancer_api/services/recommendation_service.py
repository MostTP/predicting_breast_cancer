"""Recommend the most effective treatment combination predicted by the SVM model."""

import itertools
from ..core.model_loader import model, t_learner_models
from ..core.model_metadata import get_model_metadata
from ..utils.preprocessing import prepare_patient_df
from .explain_service import explain
from .disclaimer_service import get_disclaimer


def _is_valid_combination(patient: dict, chemo: str, hormone: str, radio: str):
    """Apply conservative clinical plausibility checks.

    Only filters truly implausible combinations. The patient's actual
    current treatment is always preserved so comparisons remain valid.
    """
    er = patient.get("ER_Status")
    pr = patient.get("PR_Status")

    current_chemo = patient.get("Chemotherapy", "No")
    current_hormone = patient.get("Hormone_Therapy", "No")
    current_radio = patient.get("Radio_Therapy", "No")

    # NEVER filter out the patient's actual current treatment
    if chemo == current_chemo and hormone == current_hormone and radio == current_radio:
        return True, None

    # Truly implausible: hormone therapy without hormone receptors
    if hormone == "Yes":
        if er != "Positive" and pr != "Positive":
            return False, "Hormone therapy is not indicated for ER-/PR- patients"

    # All other combinations are clinically possible (patient may have refused,
    # been unfit, or had a different risk profile). Let the model score them.
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

    current_treatment = {
        "Chemotherapy": patient_dict.get("Chemotherapy", "No"),
        "Hormone_Therapy": patient_dict.get("Hormone_Therapy", "No"),
        "Radio_Therapy": patient_dict.get("Radio_Therapy", "No")
    }

    def _predict_option(chemo: str, hormone: str, radio: str):
        temp = patient_dict.copy()
        temp["Chemotherapy"] = chemo
        temp["Hormone_Therapy"] = hormone
        temp["Radio_Therapy"] = radio

        df = prepare_patient_df(temp)
        if t_learner_models is not None:
            key = f"{chemo}_{hormone}_{radio}"
            option_model = t_learner_models.get(key)
            if option_model is not None:
                prob = option_model.predict_proba(df)[0, 1]
            else:
                prob = model.predict_proba(df)[0, 1]
        else:
            prob = model.predict_proba(df)[0, 1]

        return {
            "Chemotherapy": chemo,
            "Hormone_Therapy": hormone,
            "Radio_Therapy": radio,
            "estimated_outcome_probability": float(prob),
            "model_score": float(prob),
            # Backward-compatible alias. Prefer estimated_outcome_probability.
            "success_probability": float(prob),
            "is_current_treatment": (
                chemo == current_treatment["Chemotherapy"]
                and hormone == current_treatment["Hormone_Therapy"]
                and radio == current_treatment["Radio_Therapy"]
            )
        }

    results = []
    skipped = []

    for chemo, hormone, radio in combinations:
        valid, reason = _is_valid_combination(patient_dict, chemo, hormone, radio)
        if not valid:
            skipped.append({
                "Chemotherapy": chemo,
                "Hormone_Therapy": hormone,
                "Radio_Therapy": radio,
                "reason": reason
            })
            continue

        results.append(_predict_option(chemo, hormone, radio))

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

    baseline_prob = None
    current_prob = None
    for r in results:
        if (
            r["Chemotherapy"] == "No"
            and r["Hormone_Therapy"] == "No"
            and r["Radio_Therapy"] == "No"
        ):
            baseline_prob = r["estimated_outcome_probability"]
        if r["is_current_treatment"]:
            current_prob = r["estimated_outcome_probability"]

    for rank, r in enumerate(results, start=1):
        r["rank"] = rank
        if baseline_prob is not None:
            r["estimated_treatment_effect_vs_no_treatment"] = (
                r["estimated_outcome_probability"] - baseline_prob
            )
        if current_prob is not None:
            r["estimated_treatment_effect_vs_current_treatment"] = (
                r["estimated_outcome_probability"] - current_prob
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
        "current_treatment": current_treatment,
        "current_treatment_probability": current_prob,
        "current_treatment_is_best": bool(current_prob is not None and top["is_current_treatment"]),
        "baseline_treatment": {
            "Chemotherapy": "No",
            "Hormone_Therapy": "No",
            "Radio_Therapy": "No",
            "estimated_outcome_probability": baseline_prob
        },
        "recommended_treatment": top,
        "recommended_treatment_improvement_over_current": (
            top["estimated_outcome_probability"] - current_prob
            if current_prob is not None
            else None
        ),
        "recommended_treatment_improvement_over_baseline": (
            top["estimated_treatment_effect_vs_no_treatment"]
            if baseline_prob is not None
            else None
        ),
        "top_ranked_option": top,
        "ranked_treatment_options": results,
        "confidence": _confidence_label(results),
        "skipped_combinations": skipped,
        "warnings": _recommendation_warnings(),
        "model_metadata": get_model_metadata(),
        # Backward-compatible aliases. Prefer recommended_treatment and ranked_treatment_options.
        "best_treatment": top,
        "most_effective_treatment": top,
        "all_options": results,
        "disclaimer": get_disclaimer()
    }