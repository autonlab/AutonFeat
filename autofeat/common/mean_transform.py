import numpy as np
from typing import Callable, Union, Any

from autofeat.core import Transform
from autofeat.functional import mean_tf


class MeanTransform(Transform):
    """
    Compute the mean of the values in x where `where` is True.

    Inherits from Transform.
    """
    # Dunder methods
    def __init__(self):
        super().__init__(name="Mean")
    
    def __call__(self, signal_window: np.ndarray, where: Callable[[Union[int, float, np.int_, np.float_]], Union[bool, np.bool_]]=None) -> Union[np.float_, np.int_]:
        """
        Apply the transformation to the signal window provided.

        Args:
            signal_window: The signal window to transform.
            where: A function that takes a value and returns True or False. (Default: None)
        
        Returns:
            A scalar value representing the transformation of the signal.

        """
        return mean_tf(signal_window, where=where)
