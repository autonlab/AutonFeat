import numpy as np
from autofeat import SampleEntropyTransform
import pytest


def test_sample_entropy():
    """
    Test sample entropy transform.
    """

    x = np.random.rand(100)

    tf = SampleEntropyTransform()
    y_hat = tf(x, m=2, r=0.2)

    y = 0.0

    assert pytest.approx(y_hat) == y
