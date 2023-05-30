import numpy as np
from typing import Callable, Union, Optional

from autofeat.core import Transform
from autofeat.functional import cross_entropy_tf


class CrossEntropyTransform(Transform):
    """
    Compute the skewness of the values in `x`.

    Inherits from Transform.
    """
    # Dunder methods
    def __init__(self, name: str = "Cross Entropy") -> None:
        super().__init__(name=name)

    def __call__(self, pk: np.ndarray, qk: np.ndarray, base: Optional[Union[int, np.int_]] = None, where: Callable[[Union[int, float, np.int_, np.float_]], Union[bool, np.bool_]] = lambda x: not np.isnan(x)) -> Union[float, np.float_]:
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
        return cross_entropy_tf(pk, qk=qk, base=base, where=where)
