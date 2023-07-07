import numpy as np
from autonfeat import VarTransform
import pytest


def test_var():
    """
    Test variance transform.
    """

    x = np.random.rand(100)

    tf = VarTransform()
    y_hat = tf(x)

    y = np.var(x)

    assert pytest.approx(y_hat) == y
