# CreditOptima

Explainable credit underwriting from raw data to an instant decision API. CreditOptima trains a
cost-sensitive XGBoost model on imbalanced credit data, benchmarks it against logistic regression,
measures decile lift and KS, and exposes SHAP-based adverse-action reasons through FastAPI.

## Quick start

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate
# macOS/Linux: source .venv/bin/activate
pip install -e ".[dev]"
python -m creditoptima.training
uvicorn creditoptima.api:app --reload
```

Open `http://localhost:8000` for the underwriting console and `/docs` for OpenAPI. Training fetches
the Kaggle *Give Me Some Credit* dataset mirror, creates `artifacts/model.joblib`,
`artifacts/metrics.json`, and `artifacts/deciles.csv`, and prints the measured metrics.

## API example

```bash
curl -X POST http://localhost:8000/v1/score -H "Content-Type: application/json" -d '{
  "RevolvingUtilizationOfUnsecuredLines": 0.82,
  "age": 42,
  "NumberOfTime30-59DaysPastDueNotWorse": 1,
  "DebtRatio": 0.47,
  "MonthlyIncome": 5200,
  "NumberOfOpenCreditLinesAndLoans": 8,
  "NumberOfTimes90DaysLate": 0,
  "NumberRealEstateLoansOrLines": 1,
  "NumberOfTime60-89DaysPastDueNotWorse": 0,
  "NumberOfDependents": 2
}'
```

Routes: `GET /health`, `GET /v1/model/metrics`, `POST /v1/score`, `GET /docs`.

## Reproducible evaluation

The XGBoost/logit comparison holds approval volume at 80%, making “false approvals” (defaults among
approved applicants) operationally comparable. The claimed targets are evaluated honestly rather
than written into code. See [MODEL_CARD.md](MODEL_CARD.md) for limitations and governance notes.

```bash
pytest -q
docker compose up --build
```

## Architecture

```text
Kaggle CSV → validation/cleaning → stratified split → imputation
                                              ├─ balanced logit baseline
                                              └─ cost-sensitive XGBoost
                                                    ├─ ROC-AUC / KS / deciles
                                                    ├─ TreeSHAP reason codes
                                                    └─ versioned model artifact
                                                               ↓
                                      FastAPI + OpenAPI + responsive decision UI
```

## Responsible-use note

This repository is a technical demonstration, not financial advice or an automated lending system
approved for production. Real deployment requires jurisdiction-specific fair-lending review,
representativeness and bias analysis, monitoring, security, audit retention, and human oversight.

