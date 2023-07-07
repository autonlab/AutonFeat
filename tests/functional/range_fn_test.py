import numpy as np
from autonfeat.functional import range_tf
import pytest


def test_range_fn():
    """
    Test range functional form transform.
    """

    x = np.random.rand(100)
    y_hat = range_tf(x)

    y = np.max(x) - np.min(x)

    assert pytest.approx(y_hat) == y
