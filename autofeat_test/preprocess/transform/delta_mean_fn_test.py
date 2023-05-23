import numpy as np
from autofeat.preprocess.transform import DeltaMeanPreprocessor


def test_delta_mean():
    """
    Test mean distribution shift preprocessor.
    """
    # Tolerance for numerical imprecision
    tol = 1e-5

    x = np.random.rand(100)

    tf = DeltaMeanPreprocessor()
    shifted_x_hat = tf(x)

    delta = 0.0
    for i in range(x.shape[0]):
        delta += x[i]
    delta /= x.shape[0]

    shifted_x = x - delta

    assert np.allclose(shifted_x_hat, shifted_x, atol=tol)
