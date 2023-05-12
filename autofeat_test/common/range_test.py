import numpy as np
from autofeat import RangeTransform
import pytest


def test_range():
    """
    Test range transform.
    """

    x = np.random.rand(100)

    tf = RangeTransform()
    y_hat = tf(x)

    y = np.max(x) - np.min(x)

    assert pytest.approx(y_hat) == y
