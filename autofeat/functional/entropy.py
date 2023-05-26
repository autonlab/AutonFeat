import numpy as np
from typing import Union, Callable


def entropy_tf(pk: np.ndarray, qk: np.ndarray = None, base: Union[int, np.int_] = None, where: Callable[[Union[int, float, np.int_, np.float_]], Union[bool, np.bool_]] = lambda x: not np.isnan(x)) -> Union[float, np.float_]:
    """
    Compute the entropy of the values in `pk` where `where` is `True`.

    Args:
        `pk`: The discrete probability distribution to find the entropy of.

        `qk`: The second discrete probability distribution to find the relative entropy with. Default is `None`.
                        If `qk` is `None`, Shannon entropy is computed using `H = -sum(pk * log(pk))`.\n
                        If `qk` is not `None`, relative entropy is computed using `H = -sum(pk * log(pk / qk))`. This is also called the Kullback-Leibler (KL) divergence.


        `base`: The base of the logarithm used to compute the entropy. Default is `None` which means that the natural logarithm is used.

        `where`: A function that takes a value and returns `True` or `False`. Default is `lambda x: not np.isnan(x)` i.e. a measurement is valid if it is not a `NaN` value.

    Returns:
        The entropy of the values in `pk` optionally with respect to `qk` (relative entropy) where `where` is `True`.
    """
    # Vectorize where fn
    where_fn = np.vectorize(pyfunc=where)
    # Get the valid values
    pk = pk[where_fn(pk)]
    qk = qk[where_fn(qk)] if qk is not None else None
    # Compute entropy
    if qk is None:
        # Shannon entropy
        return -np.sum(pk * np.log(pk) / np.log(base))
    else:
        # KL divergence
        return -np.sum(pk * np.log(pk / qk) / np.log(base))
