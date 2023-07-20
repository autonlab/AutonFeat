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
import numba as nb
from typing import Union


def check_num_threads(num_threads: Union[int, np.int_]) -> None:
    """
    Check if the number of threads is valid.

    Args:
        `num_threads`: The number of threads to use.

    Raises:
        `ValueError`: If `num_threads` is not a positive integer.

        `ValueError`: If `num_threads` is greater than the number of available threads.
    """
    if not isinstance(num_threads, (int, np.int_)):
        raise ValueError("num_threads must be an integer.")
    if num_threads < 1:
        raise ValueError("num_threads must be greater than 0.")
    if num_threads > nb.get_num_threads():
        raise ValueError("num_threads must be less than or equal to the number of available threads.")


def set_num_threads(num_threads: Union[int, np.int_]) -> None:
    """
    Set the number of threads to use.
    """
    # Check if the number of threads is valid
    check_num_threads(num_threads)
    nb.set_num_threads(num_threads)
