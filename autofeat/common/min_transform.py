import numpy as np
from typing import Callable, Union

from autofeat.core import Transform
from autofeat.functional import min_tf


class MinTransform(Transform):
    """
    Compute the min of the values in `x`.

    Inherits from Transform.
    """
    # Dunder methods
    def __init__(self):
        super().__init__(name="Min")
    
    def __call__(self, signal_window: np.ndarray, where: Callable[[Union[int, float, np.int_, np.float_]], Union[bool, np.bool_]]=None) -> Union[np.float_, np.int_]:
        """
        Compute the min of the signal window provided.

        Args:
            `signal_window`: The signal window to find the min of.
            
            `where`: A function that takes a value and returns `True` or `False` for whether it is to be included in the computation. Default is `None`.
        
        Returns:
            A scalar value representing the min of the signal.

        """
        return min_tf(signal_window, where=where)
