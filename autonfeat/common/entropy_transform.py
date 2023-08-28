# Author(s): Dhruv Srikanth
# Email(s): dsrikant (at) andrew (dot) cmu (dot) edu
# Acknowledgements:
# Copyright (c) 2023 Carnegie Mellon University, Auton Lab
# This code is subject to the license terms contained in the code repo.

import numpy as np
from typing import Callable, Union, Optional

from autonfeat.core import Transform
from autonfeat.functional import entropy_tf


class EntropyTransform(Transform):
    """
    Compute the entropy of a distribution, or the KL divergence between two distributions.
    """
    # Dunder methods
    def __init__(self, name: str = "Shannon Entropy/KL Divergence") -> None:
        super().__init__(name=name)

    def __call__(self, pk: np.ndarray, qk: Optional[np.ndarray] = None, base: Optional[Union[int, np.int_]] = None, where: Callable[[Union[int, float, np.int_, np.float_]], Union[bool, np.bool_]] = lambda x: not np.isnan(x)) -> Union[float, np.float_]:
        """
        Compute the entropy of the values in `pk` where `where` is `True`.

        Args:
            pk: The discrete probability distribution to find the entropy of.

            qk: The second discrete probability distribution to find the relative entropy with. Default is `None`.

            base: The base of the logarithm used to compute the entropy. Default is `None` which means that the natural logarithm is used.

            where: A function that takes a value and returns `True` or `False`. Default is `lambda x: not np.isnan(x)` i.e. a measurement is valid if it is not a `NaN` value.

        Returns:
            The entropy of the values in `pk` optionally with respect to `qk` (relative entropy) where `where` is `True`.
        """
        return entropy_tf(pk, qk=qk, base=base, where=where)
