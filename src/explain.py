"""SHAP explanation helpers for the loan default model.

Wraps the SHAP TreeExplainer with a clean interface that returns ready-to-plot
Explanation objects for both local (per-applicant) and global views.
"""
from __future__ import annotations

from typing import List, Tuple

import numpy as np
import pandas as pd
import shap
from sklearn.pipeline import Pipeline


def _clean_feature_name(raw: str) -> str:
    """Strip the ColumnTransformer prefix from a feature name."""
    return raw.replace("cat__", "").replace("num__", "").replace("_", " ")


def get_feature_names(pipeline: Pipeline) -> List[str]:
    """Return the human-readable feature names after preprocessing."""
    preprocessor = pipeline.named_steps["preprocess"]
    raw_names = preprocessor.get_feature_names_out()
    return [_clean_feature_name(n) for n in raw_names]


def explain_single(
    pipeline: Pipeline,
    sample: pd.DataFrame,
) -> shap.Explanation:
    """Produce a single-row SHAP Explanation for the local waterfall plot."""
    model = pipeline.named_steps["model"]
    preprocessor = pipeline.named_steps["preprocess"]
    processed = preprocessor.transform(sample)

    explainer = shap.TreeExplainer(model)
    raw = explainer(processed)

    # Random Forest binary classifier returns a 3D array (n, features, 2)
    if raw.values.ndim == 3:
        values = raw.values[0, :, 1]
        base_value = explainer.expected_value[1]
    elif raw.values.ndim == 2:
        values = raw.values[0]
        base_value = (
            explainer.expected_value
            if np.isscalar(explainer.expected_value)
            else explainer.expected_value[0]
        )
    else:
        raise ValueError(f"Unexpected SHAP values shape: {raw.values.shape}")

    return shap.Explanation(
        values=values,
        base_values=base_value,
        data=processed[0],
        feature_names=get_feature_names(pipeline),
    )


def explain_global(
    pipeline: Pipeline,
    X: pd.DataFrame,
    max_samples: int = 500,
) -> shap.Explanation:
    """Produce a global SHAP Explanation for beeswarm and importance plots.

    Args:
        max_samples: Cap the number of rows used for the global explanation to
            keep the Streamlit app responsive on large datasets.
    """
    model = pipeline.named_steps["model"]
    preprocessor = pipeline.named_steps["preprocess"]

    # Sub-sample for speed; SHAP on 2000+ rows with a tree explainer is slow
    if len(X) > max_samples:
        X = X.sample(n=max_samples, random_state=42)

    processed = preprocessor.transform(X)
    explainer = shap.TreeExplainer(model)
    raw = explainer(processed)

    if raw.values.ndim == 3:
        # Use the positive-class SHAP values
        return shap.Explanation(
            values=raw.values[:, :, 1],
            base_values=raw.base_values[:, 1] if raw.base_values.ndim > 1 else raw.base_values,
            data=processed,
            feature_names=get_feature_names(pipeline),
        )
    return shap.Explanation(
        values=raw.values,
        base_values=raw.base_values,
        data=processed,
        feature_names=get_feature_names(pipeline),
    )


def top_n_features(
    explanation: shap.Explanation,
    n: int = 5,
) -> List[Tuple[str, float]]:
    """Return the top-N features by mean absolute SHAP value across the explanation."""
    values = np.abs(explanation.values)
    if values.ndim == 2:
        mean_abs = values.mean(axis=0)
    else:
        mean_abs = values

    feature_names = explanation.feature_names
    order = np.argsort(mean_abs)[::-1]
    return [(feature_names[i], float(mean_abs[i])) for i in order[:n]]
