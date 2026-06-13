from pathlib import Path

import joblib

# Prefer an S-learner model when available. A T-learner dictionary may also be available to score
# specific treatment combinations more directly.
MODEL_DIR = Path(__file__).resolve().parents[1] / "model"
S_LEARNER_PATH = MODEL_DIR / "svm_s_learner.pkl"
FALLBACK_MODEL_PATH = MODEL_DIR / "svm_breast_cancer_recommender.pkl"
T_LEARNER_PATH = MODEL_DIR / "svm_t_learner.pkl"

if S_LEARNER_PATH.exists():
    model = joblib.load(S_LEARNER_PATH)
elif FALLBACK_MODEL_PATH.exists():
    model = joblib.load(FALLBACK_MODEL_PATH)
else:
    raise FileNotFoundError(f"No model file found at {S_LEARNER_PATH} or {FALLBACK_MODEL_PATH}")

if T_LEARNER_PATH.exists():
    t_learner_models = joblib.load(T_LEARNER_PATH)
else:
    t_learner_models = None
