from fastapi.testclient import TestClient

from creditoptima.api import app


def test_health_contract():
    with TestClient(app) as client:
        response = client.get("/health")
    assert response.status_code == 200
    assert set(response.json()) == {"status", "model_loaded", "model_version"}


def test_validation_rejects_invalid_age():
    payload = {
        "RevolvingUtilizationOfUnsecuredLines": 0.5, "age": 12,
        "NumberOfTime30-59DaysPastDueNotWorse": 0, "DebtRatio": 0.3,
        "MonthlyIncome": 5000, "NumberOfOpenCreditLinesAndLoans": 4,
        "NumberOfTimes90DaysLate": 0, "NumberRealEstateLoansOrLines": 1,
        "NumberOfTime60-89DaysPastDueNotWorse": 0, "NumberOfDependents": 0,
    }
    with TestClient(app) as client:
        assert client.post("/v1/score", json=payload).status_code == 422

