"""Tests for SHAP explanation and PDF report modules."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import pytest

from generate_data import generate_loan_data
from model import train
from explain import explain_global, explain_single, get_feature_names, top_n_features
from report import build_report


@pytest.fixture
def trained():
    """Train a small model for test reuse."""
    data = generate_loan_data(n_rows=200, seed=0)
    return train(data, "Default")


class TestExplain:

    def test_feature_names_are_clean(self, trained):
        names = get_feature_names(trained.pipeline)
        # No prefix garbage like cat__ or num__
        for name in names:
            assert "cat__" not in name
            assert "num__" not in name

    def test_single_explanation_has_correct_shape(self, trained):
        sample = trained.X_test.iloc[0:1]
        exp = explain_single(trained.pipeline, sample)
        assert exp.values is not None
        assert exp.feature_names is not None
        assert len(exp.values) == len(exp.feature_names)

    def test_global_explanation_handles_sampling(self, trained):
        # Pass a dataset larger than max_samples
        exp = explain_global(trained.pipeline, trained.X_test, max_samples=10)
        assert exp.values.shape[0] <= 10

    def test_top_n_features_returns_correct_count(self, trained):
        exp = explain_global(trained.pipeline, trained.X_test, max_samples=20)
        top = top_n_features(exp, n=3)
        assert len(top) == 3
        # Sorted descending by importance
        impacts = [t[1] for t in top]
        assert impacts == sorted(impacts, reverse=True)


class TestReport:

    def test_pdf_is_valid_bytes(self, trained):
        sample = trained.X_test.iloc[0:1]
        top = [("Income", 0.15), ("LoanAmount", 0.12), ("CreditScore", 0.08)]
        pdf_bytes = build_report(
            applicant_features=sample,
            decision="APPROVE",
            probability=0.23,
            top_reasons=top,
        )
        assert isinstance(pdf_bytes, bytes)
        assert len(pdf_bytes) > 1000
        # Valid PDFs start with %PDF
        assert pdf_bytes[:4] == b"%PDF"

    def test_report_handles_reject_decision(self, trained):
        sample = trained.X_test.iloc[0:1]
        top = [("LoanAmount", 0.20), ("Income", 0.15)]
        pdf_bytes = build_report(
            applicant_features=sample,
            decision="REJECT",
            probability=0.78,
            top_reasons=top,
        )
        assert pdf_bytes[:4] == b"%PDF"
