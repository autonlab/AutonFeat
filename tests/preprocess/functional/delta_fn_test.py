import numpy as np
from autonfeat.preprocess.transform import DeltaPreprocessor


def test_delta_fn():
    """
    Test delta distribution shift preprocessor.
    """
    # Tolerance for numerical imprecision
    tol = 1e-5

    x = np.random.rand(100)

    for _ in (10, 100, 10):
        random_delta = np.random.rand()

        tf = DeltaPreprocessor()
        shifted_x_hat = tf(x, delta=random_delta)

        shifted_x = x - random_delta

        assert np.allclose(shifted_x_hat, shifted_x, atol=tol)
