# CreditOptima model card

## Intended use

CreditOptima ranks consumer default risk and returns reason codes for a human-governed lending
workflow. It is a portfolio demonstration, not a production lending policy or legal compliance
certification. A lender must independently validate it for its population, policy, jurisdiction,
and protected-class outcomes.

## Data and target

The model uses Kaggle's 150,000-row *Give Me Some Credit* dataset. The target is serious financial
distress within two years. The data is historical and anonymized; it is not representative of every
market or current economic conditions. `age` can create fair-lending concerns and should be removed
or carefully governed in a real deployment.

## Method

- Stratified 80/20 holdout, fixed seed 42
- Median missing-value imputation
- Cost-sensitive XGBoost using the train-set negative/positive class ratio
- Class-balanced logistic regression benchmark
- Models compared at the same 80% approval volume
- ROC-AUC, KS, false-approval rate, and risk deciles saved in `artifacts/`
- TreeSHAP local contributions converted into stable adverse-action reason codes

## Governance controls

Input schemas reject unknown or impossible values. Every response includes model version, request
ID, latency, PD, score, tier, and ranked reasons. The training run records time, seed, dataset URL,
and SHA-256 digest. Missing controls for production include authentication, encryption, immutable
audit storage, calibration monitoring, drift alerts, protected-class testing, human overrides, and
formal model-risk approval.

## Metric claims

The brief's 0.88 ROC-AUC, 22% false-approval reduction, and 44.2% KS are acceptance targets—not
constants embedded in the application. Run `make train` and read `artifacts/metrics.json` for the
measured holdout results. Dataset/library changes may move them. This prevents unsupported claims.

