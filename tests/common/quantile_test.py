import numpy as np
from autofeat import QuantileTransform
import pytest


def test_quantile():
    """
    Test quantile transform.
    """

    x = np.random.rand(100)

    tf = QuantileTransform()

    for p in (10, 100, 10):
        y_hat = tf(x, p / 100)
        y = np.percentile(x, p)

        assert pytest.approx(y_hat) == y
