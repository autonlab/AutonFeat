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
import scipy.stats as stats
from autonfeat.functional import entropy_tf
import pytest


def test_shannon_entropy_fn():
    """
    Test Shannon Entropy functional form transform.
    """

    x = np.random.rand(100)
    y_hat = entropy_tf(x)

    y = stats.entropy(x)

    assert pytest.approx(y_hat) == y


def test_kl_div_fn():
    """
    Test KL Divergence functional form transform.
    """

    x1 = np.random.rand(100)
    x2 = np.random.rand(100)
    y_hat = entropy_tf(x1, x2)

    y = stats.entropy(x1, x2)

    assert pytest.approx(y_hat) == y
