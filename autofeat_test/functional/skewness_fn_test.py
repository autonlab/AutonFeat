import numpy as np
import scipy.stats as stats
from autofeat.functional import skewness_tf
import pytest


def test_skewness_fn():
    """
    Test skewness functional form transform.
    """

    x = np.random.rand(100)

    y_hat = skewness_tf(x)

    y = stats.skew(x, bias=False)

    assert pytest.approx(y_hat) == y
