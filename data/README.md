# Data

Training downloads `cs-training.csv` into `data/raw/`. Raw data is ignored by Git. The source is
the 150,000-row **Give Me Some Credit** Kaggle competition dataset, obtained through a public
GitHub mirror for a credential-free reproducible demo.

- Canonical source: https://www.kaggle.com/c/GiveMeSomeCredit/data
- Mirror used by the script: https://github.com/JLZml/Credit-Scoring-Data-Sets
- Target: `SeriousDlqin2yrs` (serious delinquency within two years)

The training artifact records a SHA-256 digest of the exact CSV. Dataset use remains subject to the
Kaggle competition rules. Do not commit personal or production applicant data.

