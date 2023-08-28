# Author(s): Dhruv Srikanth
# Email(s): dsrikant (at) andrew (dot) cmu (dot) edu
# Acknowledgements:
# Copyright (c) 2023 Carnegie Mellon University, Auton Lab
# This code is subject to the license terms contained in the code repo.

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
