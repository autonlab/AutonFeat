import numpy as np
from autonfeat import StdTransform
import pytest


def test_std():
    """
    Test standard deviation transform.
    """

    x = np.random.rand(100)

    tf = StdTransform()
    y_hat = tf(x)

    y = np.std(x)

    assert pytest.approx(y_hat) == y
