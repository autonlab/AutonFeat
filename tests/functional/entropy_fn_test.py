import numpy as np
import scipy.stats as stats
from autonfeat.functional import entropy_tf
import pytest


def test_shannon_entropy_fn():
    """
    Test Shannon Entropy functional form transform.
    """

    x = np.random.rand(100)
    y_hat = entropy_tf(x)

    y = stats.entropy(x)

    assert pytest.approx(y_hat) == y


def test_kl_div_fn():
    """
    Test KL Divergence functional form transform.
    """

    x1 = np.random.rand(100)
    x2 = np.random.rand(100)
    y_hat = entropy_tf(x1, x2)

    y = stats.entropy(x1, x2)

    assert pytest.approx(y_hat) == y
