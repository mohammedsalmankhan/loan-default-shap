# Explainable Loan Default Prediction

> A Streamlit application that predicts loan default risk using a class-balanced Random Forest and explains every decision with **SHAP** — locally per-applicant and globally across the cohort. Built for the responsible AI workflow used in regulated consumer lending.

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Tests](https://img.shields.io/badge/tests-18%20passing-brightgreen.svg)](tests/)
[![Streamlit App](https://img.shields.io/badge/Streamlit-Live%20Demo-FF4B4B.svg)](https://loan-default-shap.streamlit.app)
[![SHAP](https://img.shields.io/badge/Explainability-SHAP-blueviolet.svg)](https://shap.readthedocs.io/)

** Live demo:** [loan-default-shap.streamlit.app](https://loan-default-shap.streamlit.app/)
---

## Why this matters

Modern credit-risk models can be highly accurate but opaque — and in regulated consumer lending, opaque is not enough. Under UK consumer credit and equality law, applicants have the right to a meaningful explanation of automated decisions and to request a manual review. This project demonstrates the kind of audit-friendly, explainable workflow that responsible AI teams in financial services are expected to deliver.

Every prediction in this app comes with:
1. A clear **APPROVE** or **REJECT** decision
2. The model's **predicted default probability**
3. A **local SHAP waterfall** showing which features pushed the decision
4. A **plain-English reason sentence** for customer communication
5. A **downloadable PDF audit report** suitable for regulatory review

---

## Features

| Feature | What it shows |
|---|---|
| **CSV upload** | Train on any loan dataset, or use the bundled synthetic sample |
| **Model evaluation panel** | Accuracy, F1, precision, recall, ROC-AUC + confusion matrix |
| **Decision threshold slider** | Tune the precision/recall trade-off interactively |
| **Local SHAP waterfall** | Per-applicant decision reasoning |
| **Global SHAP beeswarm** | Cohort-level feature importance |
| **Plain-English reasons** | Top SHAP drivers translated into a customer-facing sentence |
| **PDF decision report** | One-page audit-ready report with applicant data, decision, and top reasons |
| **Batch scoring** | Score the entire test set and download as CSV |
| **Model card** | Intended use, training data, limitations, regulatory context |

---

## Tech stack

| Layer | Tools |
|---|---|
| ML | Scikit-learn (Pipeline, ColumnTransformer, RandomForestClassifier) |
| Explainability | SHAP (TreeExplainer for tree models) |
| UI | Streamlit |
| PDF generation | ReportLab |
| Visualisation | Matplotlib |
| Testing | pytest (18 unit + integration tests) |
| Data | Pandas, NumPy |

---

## Results on the bundled sample dataset

The bundled synthetic dataset has **2,000 applicants** with a **40.3% default rate** — deliberately imbalanced to mirror realistic lending data.

| Metric | Value |
|---|---|
| Accuracy | 78.5% |
| F1 (default class) | 0.736 |
| Precision | 0.727 |
| Recall | 0.745 |
| ROC-AUC | **0.871** |
| Training rows | 1,600 |
| Test rows | 400 |
| Model | Random Forest, 300 trees, max depth 8, class-balanced |

> Numbers verified on `sample_data/loans.csv` with random seed 42. Exact values vary slightly between runs due to stochastic tree fitting.

---

## Project structure

```
loan-default-shap/
├── src/
│   ├── __init__.py
│   ├── generate_data.py     # Synthetic dataset generator
│   ├── model.py             # ML pipeline (train, evaluate, predict)
│   ├── explain.py           # SHAP TreeExplainer wrappers
│   └── report.py            # PDF decision report generator
├── tests/
│   ├── test_model.py        # 13 tests covering pipeline, training, eval
│   └── test_explain_report.py  # 5 tests for SHAP and PDF output
├── sample_data/
│   └── loans.csv            # 2,000-row synthetic dataset
├── docs/
│   └── screenshots/         # README screenshots
├── streamlit_app.py         # Full Streamlit UI
├── requirements.txt
├── .gitignore
├── LICENSE
└── README.md
```

---

## Quick start

### 1. Clone and install

```bash
git clone https://github.com/mohammedsalmankhan/loan-default-shap.git
cd loan-default-shap
pip install -r requirements.txt
```

### 2. Run the Streamlit app

```bash
streamlit run streamlit_app.py
```

A browser tab opens at `http://localhost:8501`. Choose **"Use bundled sample"** in the sidebar, click **Train model**, then explore.

### 3. Regenerate the dataset (optional)

```bash
python src/generate_data.py --rows 5000 --seed 7 --out sample_data/loans.csv
```

### 4. Run the tests

```bash
pytest tests/ -v
```

All 18 tests should pass in under 10 seconds.

---

## Usage examples

**Programmatic training and scoring:**

```python
import pandas as pd
from src.model import train, predict_batch
from src.explain import explain_single, top_n_features

data = pd.read_csv("sample_data/loans.csv")
result = train(data, target_column="Default")

# Score a single applicant
sample = result.X_test.iloc[0:1]
explanation = explain_single(result.pipeline, sample)
print(top_n_features(explanation, n=3))

# Batch score the whole test set
results = predict_batch(result.pipeline, result.X_test, threshold=0.4)
results.to_csv("decisions.csv", index=False)
```

---

## Responsible AI considerations

This project deliberately bakes in the considerations that production lending models must address:

- **Fairness:** Features like income, employment length, and geography can act as proxies for protected characteristics. The model card flags this; in production, group-wise error rate testing would be required before deployment.
- **Explainability:** Every decision is paired with SHAP attributions, satisfying the "meaningful explanation" requirement under UK consumer credit law.
- **Auditability:** The PDF report provides a permanent record of each decision, the applicant features, and the top reasons — supporting both regulatory review and adverse-action communication.
- **Class imbalance:** The Random Forest is class-balanced via `class_weight="balanced"` to prevent the majority class from dominating predictions.
- **Decision threshold tuning:** Lenders can set thresholds based on risk appetite (stricter for unsecured personal loans, more permissive for collateralised mortgages).
- **SHAP critique:** SHAP attributions can be unstable for rare or extreme applicant profiles; the model card documents this limitation explicitly.

---

## Future improvements

- Add a counterfactual explainer (DiCE) showing what would have to change for a rejection to flip to approval
- Group-wise fairness metrics (demographic parity, equalised odds) with a fairness dashboard
- Comparison with LIME and Anchors explainers for robustness analysis
- Stability testing of SHAP rankings across model retrains
- Model monitoring panel with drift detection and recalibration triggers

---

## License

MIT — see [LICENSE](LICENSE).

---

## About

Built by Mohammed Salman Khan, MSc Artificial Intelligence at Ulster University. This project began as MSc coursework and has been refactored to production quality for portfolio use.

Email: mohammedsalmankhans636@gmail.com
LinkedIn: https://www.linkedin.com/in/mohammedsalmankhans/
GitHub: https://github.com/mohammedsalmankhan
