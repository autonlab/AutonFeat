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
from autonfeat import DataSparsityTransform


def test_data_sparsity():
    """
    Test data sparsity transform.
    """
    n_values = 100
    n_invalid_values = 10
    invalid_values = [
        np.nan,
        np.inf,
    ]

    invalid_fns = [
        lambda x: not np.isnan(x),
        lambda x: not np.isinf(x),
    ]

    for invalid_value, invalid_fn in zip(invalid_values, invalid_fns):
        x = np.random.rand(n_values)

        # Random n_invalid values
        invalid_idx = np.random.choice(
            np.arange(n_values),
            size=n_invalid_values,
            replace=False,
        )
        x[invalid_idx] = invalid_value

        tf = DataSparsityTransform()
        y_hat = tf(x, where=invalid_fn)

        assert y_hat == n_invalid_values / n_values
