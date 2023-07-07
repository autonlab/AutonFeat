import numpy as np
from autonfeat.functional import var_tf
import pytest


def test_var_fn():
    """
    Test variance functional form transform.
    """

    x = np.random.rand(100)
    y_hat = var_tf(x)

    y = np.var(x)

    assert pytest.approx(y_hat) == y
