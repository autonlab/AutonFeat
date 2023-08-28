# Author(s): Dhruv Srikanth
# Email(s): dsrikant (at) andrew (dot) cmu (dot) edu
# Acknowledgements:
# Copyright (c) 2023 Carnegie Mellon University, Auton Lab
# This code is subject to the license terms contained in the code repo.

import numpy as np
import numba as nb
from typing import Union, Callable


def sample_entropy_tf(x: np.ndarray, m: Union[int, np.int_], r: Union[int, np.int_], where: Callable[[Union[int, float, np.int_, np.float_]], Union[bool, np.bool_]] = lambda x: not np.isnan(x)) -> Union[float, np.float_]:
    """
    Compute the sample entropy of the values in `x` where `where` is `True`. This is a measure of the complexity of a signal.

    Args:
        x: The signal to find the sample entropy of.

        m: The length of the template vector.

        r: The tolerance.

        where: A function that takes a value and returns `True` or `False`. Default is `lambda x: not np.isnan(x)` i.e. a measurement is valid if it is not a `NaN` value.

    Returns:
        The sample entropy of the values in `x` where `where` is `True`.

    References:
        Sample Entropy -https://en.wikipedia.org/wiki/Sample_entropy
    """
    # Vectorize where fn
    where_fn = np.vectorize(pyfunc=where)
    # Get the valid values
    filtered_x = x[where_fn(x)]

    N = len(filtered_x)
    B = 0.0
    A = 0.0

    # Split signal and save all templates of length m
    xmi = np.array([filtered_x[i: i + m] for i in nb.prange(N - m)])
    xmj = np.array([filtered_x[i: i + m] for i in nb.prange(N - m + 1)])

    # Save all matches minus the self-match, compute B
    B = np.sum([np.sum(np.abs(xmi[i] - xmj).max(axis=1) <= r) - 1 for i in nb.prange(len(xmi))])

    # Similar for computing A
    m += 1
    xm = np.array([filtered_x[i: i + m] for i in nb.prange(N - m + 1)])

    A = np.sum([np.sum(np.abs(xm[i] - xm).max(axis=1) <= r) - 1 for i in nb.prange(len(xm))])

    # Return SampEn
    return -np.log(A / B)
