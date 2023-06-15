import numpy as np
from typing import Union, Callable, Optional


def entropy_tf(pk: np.ndarray, qk: Optional[np.ndarray] = None, base: Optional[Union[int, np.int_]] = None, where: Callable[[Union[int, float, np.int_, np.float_]], Union[bool, np.bool_]] = lambda x: not np.isnan(x)) -> Union[float, np.float_]:
    """
    Compute the entropy of the values in `pk` where `where` is `True`.

    Args:
        `pk`: The discrete probability distribution to find the entropy of.

        `qk`: The second discrete probability distribution to find the relative entropy with. Default is `None`.
                        If `qk` is `None`, Shannon entropy is computed using `H = -sum(pk * log(pk))`.\n
                        If `qk` is not `None`, relative entropy is computed using `H = sum(pk * log(pk / qk))`. This is also called the Kullback-Leibler (KL) divergence.

        `base`: The base of the logarithm used to compute the entropy. Default is `None` which means that the natural logarithm is used.

        `where`: A function that takes a value and returns `True` or `False`. Default is `lambda x: not np.isnan(x)` i.e. a measurement is valid if it is not a `NaN` value.

    Returns:
        The entropy of the values in `pk` optionally with respect to `qk` (relative entropy) where `where` is `True`.
    """
    if base is not None and base <= 0:
        raise ValueError("Base must be a positive integer or `None`.")

    # Vectorize where fn
    where_fn = np.vectorize(pyfunc=where)

    # Get the valid values
    pk = pk[where_fn(pk)]
    qk = qk[where_fn(qk)] if qk is not None else None

    # Normalize distributions
    pk = pk / np.sum(pk, axis=0)
    qk = qk / np.sum(qk, axis=0) if qk is not None else None

    # Compute Shannon entropy or KL divergence
    S = -np.sum(pk * np.log(pk), axis=0) if qk is None else np.sum(pk * np.log(pk / qk), axis=0)

    # Correct units e.g. bits, nats, etc. with base
    if base is not None:
        S /= np.log(base)
    return S
