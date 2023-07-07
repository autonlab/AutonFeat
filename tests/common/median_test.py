import numpy as np
from autofeat import MedianTransform
import pytest


def test_median():
    """
    Test median transform.
    """

    x = np.random.rand(100)

    tf = MedianTransform()
    y_hat = tf(x)

    y = np.median(x)

    assert pytest.approx(y_hat) == y
