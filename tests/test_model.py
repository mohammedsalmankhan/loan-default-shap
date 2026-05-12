"""Unit tests for the loan default ML pipeline."""
import sys
from pathlib import Path

# Make src importable
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import numpy as np
import pandas as pd
import pytest

from model import (
    build_pipeline,
    evaluate,
    predict_batch,
    predict_proba_single,
    prepare_features,
    train,
)
from generate_data import generate_loan_data


@pytest.fixture
def sample_data():
    """Generate a small reproducible loan dataset for tests."""
    return generate_loan_data(n_rows=200, seed=0)


class TestPrepareFeatures:

    def test_separates_x_and_y(self, sample_data):
        X, y, cat, num = prepare_features(sample_data, "Default")
        assert "Default" not in X.columns
        assert len(X) == len(y) == 200

    def test_drops_id_columns(self):
        df = pd.DataFrame({
            "loan_id": [1, 2, 3],
            "Income": [50000, 60000, 70000],
            "Default": [0, 1, 0],
        })
        X, _, _, _ = prepare_features(df, "Default")
        assert "loan_id" not in X.columns

    def test_categorises_columns_correctly(self, sample_data):
        _, _, cat, num = prepare_features(sample_data, "Default")
        # All synthetic features are numeric
        assert len(cat) == 0
        assert len(num) == 6


class TestPipeline:

    def test_pipeline_builds(self):
        pipe = build_pipeline(categorical_cols=[], numeric_cols=["Age", "Income"])
        assert pipe is not None
        assert "preprocess" in pipe.named_steps
        assert "model" in pipe.named_steps

    def test_pipeline_is_class_balanced(self):
        pipe = build_pipeline(categorical_cols=[], numeric_cols=["Age"])
        assert pipe.named_steps["model"].class_weight == "balanced"


class TestTraining:

    def test_train_returns_complete_result(self, sample_data):
        result = train(sample_data, "Default")
        assert result.pipeline is not None
        assert len(result.X_train) + len(result.X_test) == len(sample_data)
        assert len(result.y_train) == len(result.X_train)

    def test_train_test_split_is_stratified(self, sample_data):
        """The class balance should be similar in train and test."""
        result = train(sample_data, "Default")
        train_rate = result.y_train.mean()
        test_rate = result.y_test.mean()
        assert abs(train_rate - test_rate) < 0.10


class TestEvaluation:

    def test_metrics_in_valid_range(self, sample_data):
        result = train(sample_data, "Default")
        metrics = evaluate(result.pipeline, result.X_test, result.y_test)
        assert 0.0 <= metrics.accuracy <= 1.0
        assert 0.0 <= metrics.f1 <= 1.0
        assert 0.0 <= metrics.precision <= 1.0
        assert 0.0 <= metrics.recall <= 1.0
        assert 0.0 <= metrics.roc_auc <= 1.0
        assert metrics.confusion.shape == (2, 2)

    def test_model_beats_random_guess(self, sample_data):
        """A trained model on synthetic data with real signal should beat 0.5 ROC-AUC."""
        result = train(sample_data, "Default")
        metrics = evaluate(result.pipeline, result.X_test, result.y_test)
        assert metrics.roc_auc > 0.55, f"Model is no better than random: {metrics.roc_auc}"

    def test_threshold_changes_predictions(self, sample_data):
        """Stricter threshold should produce more rejections."""
        result = train(sample_data, "Default")
        loose = evaluate(result.pipeline, result.X_test, result.y_test, threshold=0.7)
        strict = evaluate(result.pipeline, result.X_test, result.y_test, threshold=0.3)
        # Strict threshold (0.3) means more positives -> higher recall
        assert strict.recall >= loose.recall


class TestPrediction:

    def test_single_prediction_returns_valid_probability(self, sample_data):
        result = train(sample_data, "Default")
        sample = result.X_test.iloc[0:1]
        proba = predict_proba_single(result.pipeline, sample)
        assert 0.0 <= proba <= 1.0

    def test_batch_prediction_returns_full_dataframe(self, sample_data):
        result = train(sample_data, "Default")
        batch = predict_batch(result.pipeline, result.X_test)
        assert len(batch) == len(result.X_test)
        assert "default_probability" in batch.columns
        assert "decision" in batch.columns
        assert set(batch["decision"].unique()).issubset({"APPROVE", "REJECT"})
