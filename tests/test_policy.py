import pytest

from creditoptima.model import probability_to_score, risk_policy


@pytest.mark.parametrize("probability, expected", [(0, 850), (0.5, 575), (1, 300)])
def test_score_bounds(probability, expected):
    assert probability_to_score(probability) == expected


def test_score_is_monotonic():
    assert probability_to_score(0.05) > probability_to_score(0.50)


@pytest.mark.parametrize("pd,tier,decision", [
    (0.01, "A", "APPROVE"), (0.07, "B", "APPROVE"), (0.15, "C", "REFER"),
    (0.25, "D", "REFER"), (0.40, "E", "DECLINE"),
])
def test_risk_policy(pd, tier, decision):
    assert risk_policy(pd) == (tier, decision)

