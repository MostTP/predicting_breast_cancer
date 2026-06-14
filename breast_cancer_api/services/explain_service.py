import os
import importlib
import pandas as pd
import numpy as np

from ..utils.preprocessing import prepare_patient_df
from ..core.model_loader import model

def _load_background(columns, n_samples=50):
    """Attempt to load a small background sample from the METABRIC CSV.
    Falls back to an empty result if the file isn't available."""
    try:
        root = os.getcwd()
        path = os.path.join(root, "Breast_Cancer_METABRIC.csv")
        if not os.path.exists(path):
            return None

        df = pd.read_csv(path)
        # keep only columns that overlap with model input
        avail = [c for c in df.columns if c in columns]
        if not avail:
            return None

        bg = df[avail].dropna().head(n_samples)
        # ensure all requested columns exist
        for c in columns:
            if c not in bg.columns:
                bg[c] = 0
        bg = bg[columns].fillna(bg.median(numeric_only=True))
        return bg
    except Exception:
        return None


def explain(patient_dict: dict):
    """Return SHAP feature attributions for the provided `patient_dict`.

    The returned structure is: {"features": [{"feature": name, "impact": float}, ...]}
    """
    try:
        shap = importlib.import_module("shap")
    except ImportError:
        return {
            "error": "explainability_unavailable",
            "message": "The 'shap' package is not installed in this Python environment."
        }

    df = prepare_patient_df(patient_dict)

    # Try to build a sensible background dataset for the explainer
    background = _load_background(list(df.columns), n_samples=50)
    if background is None or background.shape[0] == 0:
        background = df.iloc[[0]]

    try:
        # prediction function returning probability for positive class
        def f(x):
            x_df = pd.DataFrame(x, columns=df.columns)
            return model.predict_proba(x_df)[:, 1]

        explainer = shap.KernelExplainer(f, background.values)
        # compute shap values for the single input row
        shap_vals = explainer.shap_values(df.values, nsamples=100)

        # shap_vals may be a numpy array (n_samples, n_features)
        vals = np.array(shap_vals)
        if vals.ndim == 3:
            # sometimes returns (1, n_features) wrapped; flatten
            vals = vals[0]

        # take the first (and only) row
        row_vals = vals[0] if vals.ndim == 2 else vals

        features = [
            {"feature": name, "impact": float(impact)}
            for name, impact in zip(df.columns, row_vals)
        ]

        return {"features": features}
    except Exception as e:
        return {"error": "explainability_failed", "message": str(e)}
