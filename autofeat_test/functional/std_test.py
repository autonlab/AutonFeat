import numpy as np
from autofeat.functional import std_tf
import pytest


def test_std():
    """
    Test standard deviation functional form transform.
    """

    x = np.random.rand(100)
    y_hat = std_tf(x)

    y = np.std(x)

    assert pytest.approx(y_hat) == y
