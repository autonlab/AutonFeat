import numpy as np
from autofeat.functional import min_tf
import pytest


def test_min_fn():
    """
    Test min functional form transform.
    """

    x = np.random.rand(100)
    y_hat = min_tf(x)

    y = np.min(x)

    assert pytest.approx(y_hat) == y
