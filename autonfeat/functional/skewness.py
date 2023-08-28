# Author(s): Dhruv Srikanth
# Email(s): dsrikant (at) andrew (dot) cmu (dot) edu
# Acknowledgements:
# Copyright (c) 2023 Carnegie Mellon University, Auton Lab
# This code is subject to the license terms contained in the code repo.

import numpy as np
from typing import Union, Callable


def skewness_tf(x: np.ndarray, where: Callable[[Union[int, float, np.int_, np.float_]], Union[bool, np.bool_]] = lambda x: not np.isnan(x)) -> Union[float, np.float_]:
    """
    Compute the skewness of the values in `x` where `where` is `True`.
    The skewness is computed using the Fisher-Pearson standardized coefficient of skewness.
    The skewness is only computed for valid values i.e. values where `where` is `True`.
    The skewness computed is corrected for statistical bias.

    Args:
        x: The array to compute the skewness of.

        where: A function that takes a value and returns `True` or `False`. Default is `lambda x: not np.isnan(x)` i.e. a measurement is valid if it is not a `NaN` value.

    Returns:
        The skewness of the values in `x` where `where` is `True`.
    """
    # Vectorize where fn
    where_fn = np.vectorize(pyfunc=where)
    filtered_x = x[where_fn(x)]
    n = len(filtered_x)
    sample_mean = np.mean(filtered_x)
    m_3 = np.mean((filtered_x - sample_mean) ** 3)
    m_2 = np.mean((filtered_x - sample_mean) ** 2)
    g_1 = m_3 / (m_2 ** (3 / 2))
    G_1 = g_1 * np.sqrt(n * (n - 1)) / (n - 2)
    return G_1
