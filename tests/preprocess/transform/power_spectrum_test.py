# Author(s): Dhruv Srikanth
# Email(s): dsrikant (at) andrew (dot) cmu (dot) edu
# Acknowledgements:
# Copyright (c) 2023 Carnegie Mellon University, Auton Lab
# This code is subject to the license terms contained in the code repo.

import numpy as np
import scipy as sp
from autonfeat.preprocess import PowerSpectrumPreprocessor


def test_power_spectrum():
    """
    Test power spectrum transform preprocessor.
    """
    # Tolerance for numerical imprecision
    rng = np.random.default_rng()

    # Tolerance for numerical imprecision
    tol = 1e-5

    # Test Parameters
    fs = rng.uniform(1e3, 1e6)  # sampling frequency
    N = rng.integers(1e3, 1e6)  # number of samples
    freq = rng.uniform(1e2, 1e3)  # signal frequency
    amp = 2 * np.sqrt(2)  # signal amplitude

    # Compute the signal
    noise_power = 0.001 * fs / 2
    time = np.arange(N) / fs
    x = amp * np.sin(2 * np.pi * freq * time)
    x += rng.normal(scale=np.sqrt(noise_power), size=time.shape)

    # Compute the power spectrum
    preprocessor = PowerSpectrumPreprocessor()
    ps_hat = preprocessor(x)  # using autonfeat

    ps = np.abs(sp.fft.fft(x))**2  # using scipy

    assert np.allclose(ps_hat, ps, atol=tol)
