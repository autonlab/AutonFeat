import numpy as np
from typing import Union, Callable
from autofeat.functional.entropy import entropy_tf


def cross_entropy_tf(pk: np.ndarray, qk: np.ndarray, base: Union[int, np.int_] = None, where: Callable[[Union[int, float, np.int_, np.float_]], Union[bool, np.bool_]] = lambda x: not np.isnan(x)) -> Union[float, np.float_]:
    """
    Compute the cross-entropy of the values in `pk` with respect to `qk` where `where` is `True`.

    Args:
        `pk`: A discrete probability distribution.

        `qk`: A second discrete probability distribution.

        `base`: The base of the logarithm used to compute the entropy. Default is `None` which means that the natural logarithm is used.

        `where`: A function that takes a value and returns `True` or `False`. Default is `lambda x: not np.isnan(x)` i.e. a measurement is valid if it is not a `NaN` value.

    Returns:
        The cross-entropy of the values in `pk` with respect to `qk` where `where` is `True`.
    """
    # Cross-entropy is the sum of entropy and relative entropy
    return entropy_tf(pk=pk, base=base, where=where) + entropy_tf(pk=pk, qk=qk, base=base, where=where)
