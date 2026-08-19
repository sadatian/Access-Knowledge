import pytest
import numpy as np

def test_mrr_calculation():
    rank_positions = [1, 2, 4]
    reciprocals = [1.0 / r for r in rank_positions]
    mrr = float(np.mean(reciprocals))
    expected = (1.0 + 0.5 + 0.25) / 3.0
    assert pytest.approx(mrr, 0.001) == expected
