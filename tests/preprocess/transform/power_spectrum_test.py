# MIT License

# Copyright (c) 2023 Carnegie Mellon University, Auton Lab

# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

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

    ps = np.abs(sp.fft.fft(x)**2)  # using scipy

    assert np.allclose(ps_hat, ps, atol=tol)
