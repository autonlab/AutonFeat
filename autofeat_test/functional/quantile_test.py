import numpy as np
from autofeat.functional import quantile_tf
import pytest


def test_quantile():
    """
    Test quantile functional form transform.
    """

    x = np.random.rand(100)
    y_hat = quantile_tf(x, 0.5)

    y = np.percentile(x, 50)

    assert pytest.approx(y_hat) == y
