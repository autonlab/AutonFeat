import numpy as np
from typing import Union, Callable

from autofeat.core import Transform
from autofeat.functional import range_tf

class RangeTransform(Transform):
    """
    Compute the range of the values.

    Inherits from Transform.
    """
    # Dunder methods
    def __init__(self):
        super().__init__(name="Range")
    
    def __call__(self, signal_window: np.ndarray, where: Callable[[Union[int, float, np.int_, np.float_]], Union[bool, np.bool_]]=None) -> Union[np.float_, np.int_]:
        """
        Compute the range of the values in `x`.

        Args:
            `x`: The array to compute the range of.

            `where`: A function that takes a value and returns `True` or `False` for whether it is to be included in the computation. Default is `None`.
        
        Returns:
            A scalar value representing the range of the signal.
        """
        return range_tf(signal_window, where=where)
