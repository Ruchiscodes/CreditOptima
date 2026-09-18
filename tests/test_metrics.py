import numpy as np

from creditoptima.metrics import decile_table, ks_statistic


def test_perfect_model_has_perfect_ks():
    ks, _ = ks_statistic([0, 0, 1, 1], [0.1, 0.2, 0.8, 0.9])
    assert ks == 1.0


def test_deciles_preserve_rows():
    actual = np.array(([0] * 90) + ([1] * 10))
    score = np.linspace(0, 1, 100)
    table = decile_table(actual, score)
    assert len(table) == 10
    assert table.accounts.sum() == 100
    assert table.defaults.sum() == 10

