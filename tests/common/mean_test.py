import numpy as np
from autonfeat import MeanTransform
import pytest


def test_mean():
    """
    Test mean transform.
    """

    x = np.random.rand(100)

    tf = MeanTransform()
    y_hat = tf(x)

    y = 0.0
    for i in range(len(x)):
        y += x[i]
    y /= len(x)

    assert pytest.approx(y_hat) == y
