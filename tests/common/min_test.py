import numpy as np
from autonfeat import MinTransform
import pytest


def test_min():
    """
    Test min transform.
    """

    x = np.random.rand(100)

    tf = MinTransform()
    y_hat = tf(x)

    y = np.min(x)

    assert pytest.approx(y_hat) == y
