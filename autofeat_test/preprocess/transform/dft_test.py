import numpy as np
import scipy as sp
from autofeat.preprocess import DFTPreprocessor


def test_dft():
    """
    Test 1D discrete fourier transform preprocessor.
    """
    # Tolerance for numerical imprecision
    tol = 1e-5

    x = np.random.rand(100)

    tf = DFTPreprocessor()
    x_hat = tf(x)

    x_true = sp.fft.fft(x)

    assert np.allclose(x_hat, x_true, atol=tol)
