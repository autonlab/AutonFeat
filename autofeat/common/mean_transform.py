import numpy as np
from typing import Callable, Union

from autofeat.core import Transform
from autofeat.functional import mean_tf


class MeanTransform(Transform):
    """
    Compute the mean of the values in `x`.

    Inherits from Transform.
    """
    # Dunder methods
    def __init__(self):
        super().__init__(name="Mean")
    
    def __call__(self, signal_window: np.ndarray, where: Callable[[Union[int, float, np.int_, np.float_]], Union[bool, np.bool_]]=None) -> Union[np.float_, np.int_]:
        """
        Compute the mean of the signal window provided.

        Args:
            `signal_window`: The signal window to find the mean of.
            
            `where`: A function that takes a value and returns True or False. (Default: None)
        
        Returns:
            A scalar value representing the mean of the signal.

        """
        return mean_tf(signal_window, where=where)
