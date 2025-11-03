import numpy as np
from metrics import compute_volatility, compute_max_drawdown

def test_compute_volatility_basic():
    returns = np.array([0.01, -0.02, 0.015, -0.005])
    vol = compute_volatility(returns)
    expected = np.std(returns, ddof=1)
    assert np.isclose(vol, expected, atol=1e-10)

def test_compute_max_drawdown_basic():
    returns = np.array([0.05, 0.02, -0.1, 0.04, -0.03])
    mdd = compute_max_drawdown(returns)
    cumulative = np.cumprod(1 + returns)
    peaks = np.maximum.accumulate(cumulative)
    expected_mdd = np.min(cumulative / peaks - 1)
    assert np.isclose(mdd, expected_mdd, atol=1e-10)
