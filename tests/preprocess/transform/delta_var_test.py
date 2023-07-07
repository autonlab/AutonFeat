import numpy as np
from autofeat.preprocess import DeltaVarPreprocessor


def test_delta_var():
    """
    Test variance distribution shift preprocessor.
    """
    # Tolerance for numerical imprecision
    tol = 1e-5

    x = np.random.rand(100)

    tf = DeltaVarPreprocessor()
    shifted_x_hat = tf(x)

    delta = np.var(x)
    shifted_x = x - delta

    assert np.allclose(shifted_x_hat, shifted_x, atol=tol)
