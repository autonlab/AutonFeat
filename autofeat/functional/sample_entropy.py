import numpy as np
import numba as nb
from typing import Union, Callable


def sample_entropy_tf(x: np.ndarray, m: Union[int, np.int_], r: Union[int, np.int_], where: Callable[[Union[int, float, np.int_, np.float_]], Union[bool, np.bool_]] = lambda x: not np.isnan(x)) -> Union[float, np.float_]:
    """
    Compute the sample entropy of the values in `x` where `where` is `True`.\n
    This is a measure of the complexity of a signal. It is a modification of the approximate entropy (ApEn) algorithm. It can be computed with the formula:\n
    `SampEn = -log(A / B)`\n
    where `A` is the number of matches for template vectors of length `m` and `B` is the number of matches for template vectors of length `m + 1`.\n
    A match is defined as a template vector `xmi` that is close to another template vector `xmj` in the sense that the maximum absolute difference between their corresponding scalar elements is less than or equal to `r`.\n

    Args:
        `x`: The signal to find the sample entropy of.

        `m`: The length of the template vector.

        `r`: The tolerance.

        `where`: A function that takes a value and returns `True` or `False`. Default is `lambda x: not np.isnan(x)` i.e. a measurement is valid if it is not a `NaN` value.

    Returns:
        The sample entropy of the values in `x` where `where` is `True`.
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
    B = np.sum([np.sum(np.abs(xmii - xmj).max(axis=1) <= r) - 1 for xmii in xmi])

    # Similar for computing A
    m += 1
    xm = np.array([filtered_x[i: i + m] for i in nb.prange(N - m + 1)])

    A = np.sum([np.sum(np.abs(xmi - xm).max(axis=1) <= r) - 1 for xmi in xm])

    # Return SampEn
    return -np.log(A / B)
