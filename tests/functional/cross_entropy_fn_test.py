import numpy as np
import scipy.stats as stats
from autonfeat.functional import cross_entropy_tf
import pytest


def test_cross_entropy_fn():
    """
    Test cross entropy functional form transform.
    """

    x1 = np.random.rand(100)
    x2 = np.random.rand(100)
    y_hat = cross_entropy_tf(x1, x2, base=2)

    y = stats.entropy(x1, base=2) + stats.entropy(x1, x2, base=2)

    assert pytest.approx(y_hat) == y
