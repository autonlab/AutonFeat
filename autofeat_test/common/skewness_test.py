import numpy as np
import scipy.stats as stats
from autofeat import SkewnessTransform
import pytest


def test_skewness():
    """
    Test skewness transform.
    """

    x = np.random.rand(100)

    tf = SkewnessTransform()
    y_hat = tf(x)

    y = stats.skew(x, bias=False)

    assert pytest.approx(y_hat) == y
