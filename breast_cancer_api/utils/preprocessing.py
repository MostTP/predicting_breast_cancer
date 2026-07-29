# Training preprocessing (added for multi-cohort training)
import json
from pathlib import Path

import pandas as pd
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer, KNNImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.feature_selection import SelectKBest, mutual_info_classif
from sklearn.base import BaseEstimator, TransformerMixin


class OutlierClipper(BaseEstimator, TransformerMixin):
    """IQR-based outlier clipping. Fits on train only."""
    
    def __init__(self, factor: float = 1.5):
        self.factor = factor
        self.lower_bounds_ = None
        self.upper_bounds_ = None
    
    def fit(self, X, y=None):
        X = pd.DataFrame(X)
        q1 = X.quantile(0.25)
        q3 = X.quantile(0.75)
        iqr = q3 - q1
        self.lower_bounds_ = q1 - self.factor * iqr
        self.upper_bounds_ = q3 + self.factor * iqr
        return self
    
    def transform(self, X):
        X = pd.DataFrame(X)
        X_clipped = X.copy()
        for col in X.columns:
            if col in self.lower_bounds_.index:
                X_clipped[col] = X_clipped[col].clip(
                    lower=self.lower_bounds_[col],
                    upper=self.upper_bounds_[col],
                )
        return X_clipped.values


def build_preprocessing_pipeline(
    numerical_features: list,
    categorical_features: list,
    k_best: int = "all",
) -> Pipeline:
    """
    Build full preprocessing + feature selection pipeline for training.
    """
    num_pipeline = Pipeline([
        ("imputer", KNNImputer(n_neighbors=5)),
        ("outlier_clipper", OutlierClipper(factor=1.5)),
        ("scaler", StandardScaler()),
    ])
    
    cat_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False, drop="first")),
    ])
    
    preprocessor = ColumnTransformer([
        ("num", num_pipeline, numerical_features),
        ("cat", cat_pipeline, categorical_features),
    ], remainder="drop")
    
    selector = SelectKBest(score_func=mutual_info_classif, k=k_best)
    
    return Pipeline([
        ("preprocessor", preprocessor),
        ("feature_selection", selector),
    ])


def prepare_patient_df(patient_dict: dict) -> pd.DataFrame:
    """
    Convert a raw patient dictionary (from API request) into a DataFrame
    that the trained model pipeline can consume.
    """
    # Wrap single dict into a one-row DataFrame
    df = pd.DataFrame([patient_dict])

    # Ensure treatment flags exist (default to "No" if omitted)
    for col in ["Chemotherapy", "Hormone_Therapy", "Radio_Therapy"]:
        if col not in df.columns:
            df[col] = "No"

    # If we saved a feature schema during training, use it to guarantee
    # column presence and order so the pipeline never sees a mismatch.
    schema_path = Path(__file__).resolve().parents[1] / "model" / "feature_schema.json"
    if schema_path.exists():
        with open(schema_path) as f:
            schema = json.load(f)
        expected_features = schema.get("all_features", [])

        # Add any missing columns as NaN so the imputer can handle them
        for col in expected_features:
            if col not in df.columns:
                df[col] = np.nan

        # Reorder to exactly match training
        df = df[expected_features]

    return df