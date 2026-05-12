"""Core ML pipeline for loan default prediction.

Separated from the Streamlit UI so the modelling code can be tested and
reused independently. Provides:
    - Data preparation and train/test split
    - Pipeline construction (ColumnTransformer + Random Forest)
    - Model training
    - Evaluation metrics
    - SHAP explanations
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

# -------------------- model hyperparameters --------------------

N_ESTIMATORS: int = 300
MAX_DEPTH: int = 8
TEST_SIZE: float = 0.2
RANDOM_STATE: int = 42


@dataclass
class TrainingResult:
    """Bundle of artefacts produced by a single training run."""
    pipeline: Pipeline
    X_train: pd.DataFrame
    X_test: pd.DataFrame
    y_train: pd.Series
    y_test: pd.Series
    feature_columns: List[str]
    categorical_cols: List[str]
    numeric_cols: List[str]


@dataclass
class EvaluationMetrics:
    """Classification metrics on the test set."""
    accuracy: float
    f1: float
    precision: float
    recall: float
    roc_auc: float
    confusion: np.ndarray


# -------------------- data preparation --------------------

def prepare_features(
    df: pd.DataFrame,
    target_column: str,
) -> Tuple[pd.DataFrame, pd.Series, List[str], List[str]]:
    """Split a dataframe into features (X) and target (y), drop ID-like columns.

    Returns:
        X, y, categorical_cols, numeric_cols
    """
    # Drop obvious ID columns to prevent label leakage
    id_like = [c for c in df.columns if "id" in c.lower() and c != target_column]
    df = df.drop(columns=id_like, errors="ignore")

    X = df.drop(columns=[target_column])
    y = df[target_column]

    # Forward-then-backward fill handles missing values robustly
    X = X.ffill().bfill()

    categorical_cols = X.select_dtypes(include=["object", "category", "bool"]).columns.tolist()
    numeric_cols = X.select_dtypes(include=["int64", "float64"]).columns.tolist()

    return X, y, categorical_cols, numeric_cols


# -------------------- pipeline construction --------------------

def build_pipeline(
    categorical_cols: List[str],
    numeric_cols: List[str],
    n_estimators: int = N_ESTIMATORS,
    max_depth: int = MAX_DEPTH,
) -> Pipeline:
    """Build a scikit-learn Pipeline: ColumnTransformer + class-balanced Random Forest.

    The pipeline is deliberately simple and interpretable:
        - OneHotEncoder for categorical features (handles unseen categories at predict time)
        - Numeric features pass through unscaled (tree models are scale-invariant)
        - Random Forest with class_weight="balanced" to handle realistic class imbalance
    """
    preprocess = ColumnTransformer(
        transformers=[
            (
                "cat",
                OneHotEncoder(handle_unknown="ignore", sparse_output=False),
                categorical_cols,
            ),
            ("num", "passthrough", numeric_cols),
        ],
        remainder="drop",
    )

    model = RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        class_weight="balanced",
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )

    return Pipeline(steps=[("preprocess", preprocess), ("model", model)])


# -------------------- training --------------------

def train(
    df: pd.DataFrame,
    target_column: str,
    test_size: float = TEST_SIZE,
) -> TrainingResult:
    """Train the full pipeline end-to-end and return all artefacts."""
    X, y, cat_cols, num_cols = prepare_features(df, target_column)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=RANDOM_STATE, stratify=y,
    )
    X_test = X_test.reset_index(drop=True)
    y_test = y_test.reset_index(drop=True)

    pipeline = build_pipeline(cat_cols, num_cols)
    pipeline.fit(X_train, y_train)

    return TrainingResult(
        pipeline=pipeline,
        X_train=X_train,
        X_test=X_test,
        y_train=y_train,
        y_test=y_test,
        feature_columns=X.columns.tolist(),
        categorical_cols=cat_cols,
        numeric_cols=num_cols,
    )


# -------------------- evaluation --------------------

def evaluate(
    pipeline: Pipeline,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    threshold: float = 0.5,
) -> EvaluationMetrics:
    """Compute classification metrics on the test set at a given decision threshold."""
    proba = pipeline.predict_proba(X_test)[:, 1]
    y_pred = (proba >= threshold).astype(int)

    return EvaluationMetrics(
        accuracy=accuracy_score(y_test, y_pred),
        f1=f1_score(y_test, y_pred, zero_division=0),
        precision=precision_score(y_test, y_pred, zero_division=0),
        recall=recall_score(y_test, y_pred, zero_division=0),
        roc_auc=roc_auc_score(y_test, proba),
        confusion=confusion_matrix(y_test, y_pred),
    )


def predict_proba_single(
    pipeline: Pipeline,
    sample: pd.DataFrame,
) -> float:
    """Return the predicted default probability for a single applicant row."""
    return float(pipeline.predict_proba(sample)[0, 1])


def predict_batch(
    pipeline: Pipeline,
    X: pd.DataFrame,
    threshold: float = 0.5,
) -> pd.DataFrame:
    """Score an entire dataframe and return per-row decision and probability.

    Useful for batch scoring, which is closer to how lenders actually deploy
    these models in production (overnight batch runs over an application backlog).
    """
    proba = pipeline.predict_proba(X)[:, 1]
    decision = (proba >= threshold).astype(int)
    out = X.copy()
    out["default_probability"] = proba.round(4)
    out["decision"] = np.where(decision == 1, "REJECT", "APPROVE")
    return out
