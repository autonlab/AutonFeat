import numpy as np
import scipy.stats as stats
from autonfeat import CrossEntropyTransform
import pytest


def test_cross_entropy():
    """
    Test cross entropy transform.
    """

    x1 = np.random.rand(100)
    x2 = np.random.rand(100)

    tf = CrossEntropyTransform()
    y_hat = tf(x1, x2, base=2)

    y = stats.entropy(x1, base=2) + stats.entropy(x1, x2, base=2)

    assert pytest.approx(y_hat) == y
