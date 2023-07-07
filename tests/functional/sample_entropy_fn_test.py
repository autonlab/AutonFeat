import numpy as np
from autonfeat.functional import sample_entropy_tf
import pytest


def test_sample_entropy_fn():
    """
    Test sample entropy functional form transform.
    """

    x = np.random.rand(100)
    y_hat = sample_entropy_tf(x, m=2, r=0.2)

    y = 0.0

    # assert pytest.approx(y_hat) == y
