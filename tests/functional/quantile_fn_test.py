import numpy as np
from autonfeat.functional import quantile_tf
import pytest


def test_quantile_fn():
    """
    Test quantile functional form transform.
    """

    x = np.random.rand(100)

    for p in (10, 100, 10):
        y_hat = quantile_tf(x, p / 100)
        y = np.percentile(x, p)

        assert pytest.approx(y_hat) == y
