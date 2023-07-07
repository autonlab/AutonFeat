import numpy as np
from autofeat import MaxTransform
import pytest


def test_max():
    """
    Test max transform.
    """

    x = np.random.rand(100)

    tf = MaxTransform()
    y_hat = tf(x)

    y = np.max(x)

    assert pytest.approx(y_hat) == y
