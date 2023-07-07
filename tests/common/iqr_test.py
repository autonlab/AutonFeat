import numpy as np
from autonfeat import IQRTransform
import pytest


def test_iqr():
    """
    Test inter-quartile range transform.
    """

    x = np.random.rand(100)

    tf = IQRTransform()
    y_hat = tf(x)

    y = np.percentile(x, 75) - np.percentile(x, 25)

    assert pytest.approx(y_hat) == y
