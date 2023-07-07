import numpy as np
import scipy.stats as stats
from autonfeat import EntropyTransform
import pytest


def test_shannon_entropy():
    """
    Test Shannon entropy transform.
    """

    x = np.random.rand(100)

    tf = EntropyTransform()
    y_hat = tf(x)

    y = stats.entropy(x)

    assert pytest.approx(y_hat) == y


def test_kl_div():
    """
    Test KL Divergence transform.
    """

    x1 = np.random.rand(100)
    x2 = np.random.rand(100)

    tf = EntropyTransform()
    y_hat = tf(x1, x2)

    y = stats.entropy(x1, x2)

    assert pytest.approx(y_hat) == y
