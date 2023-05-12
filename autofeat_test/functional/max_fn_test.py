import numpy as np
from autofeat.functional import max_tf
import pytest


def test_max_fn():
    """
    Test max functional form transform.
    """

    x = np.random.rand(100)
    y_hat = max_tf(x)

    y = np.max(x)

    assert pytest.approx(y_hat) == y
