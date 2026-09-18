from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT / "data"
ARTIFACT_DIR = ROOT / "artifacts"
MODEL_PATH = ARTIFACT_DIR / "model.joblib"
METRICS_PATH = ARTIFACT_DIR / "metrics.json"
DECILES_PATH = ARTIFACT_DIR / "deciles.csv"

TARGET = "SeriousDlqin2yrs"
ID_COLUMN = "Unnamed: 0"
FEATURES = [
    "RevolvingUtilizationOfUnsecuredLines",
    "age",
    "NumberOfTime30-59DaysPastDueNotWorse",
    "DebtRatio",
    "MonthlyIncome",
    "NumberOfOpenCreditLinesAndLoans",
    "NumberOfTimes90DaysLate",
    "NumberRealEstateLoansOrLines",
    "NumberOfTime60-89DaysPastDueNotWorse",
    "NumberOfDependents",
]

# Human-readable, FCRA-style reasons. They explain model drivers; they are not legal advice.
REASON_CODES = {
    "RevolvingUtilizationOfUnsecuredLines": ("AA01", "High revolving credit utilization"),
    "age": ("AA02", "Limited length of credit history proxy"),
    "NumberOfTime30-59DaysPastDueNotWorse": ("AA03", "Recent 30–59 day delinquencies"),
    "DebtRatio": ("AA04", "High debt-to-income ratio"),
    "MonthlyIncome": ("AA05", "Income is low relative to obligations"),
    "NumberOfOpenCreditLinesAndLoans": ("AA06", "Number of open credit accounts"),
    "NumberOfTimes90DaysLate": ("AA07", "History of severe delinquency"),
    "NumberRealEstateLoansOrLines": ("AA08", "Real-estate loan exposure"),
    "NumberOfTime60-89DaysPastDueNotWorse": ("AA09", "Recent 60–89 day delinquencies"),
    "NumberOfDependents": ("AA10", "Household obligations"),
}

