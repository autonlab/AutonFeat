import numpy as np
from autofeat.preprocess import DeltaStdPreprocessor


def test_delta_std():
    """
    Test standard deviation distribution shift preprocessor.
    """
    # Tolerance for numerical imprecision
    tol = 1e-5

    x = np.random.rand(100)

    tf = DeltaStdPreprocessor()
    shifted_x_hat = tf(x)

    delta = np.std(x)
    shifted_x = x - delta

    assert np.allclose(shifted_x_hat, shifted_x, atol=tol)
