import numpy as np
from autonfeat.functional import median_tf
import pytest


def test_median_fn():
    """
    Test median functional form transform.
    """

    x = np.random.rand(100)
    y_hat = median_tf(x)

    y = np.median(x)

    assert pytest.approx(y_hat) == y
