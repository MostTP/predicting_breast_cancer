from pathlib import Path

import joblib

MODEL_PATH = Path(__file__).resolve().parents[1] / "model" / "svm_breast_cancer_recommender.pkl"

model = joblib.load(MODEL_PATH)
