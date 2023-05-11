import numpy as np
from typing import Union

from autofeat.core import Transform
from autofeat.functional import median_tf

class MedianTransform(Transform):
    """
    Compute the median of the values.

    Inherits from Transform.
    """
    # Dunder methods
    def __init__(self):
        super().__init__(name="Median")
    
    def __call__(self, signal_window: np.ndarray, method: str='linear') -> Union[np.float_, np.int_]:
        """
        Compute the median of the values in `x`.

        Args:
            `x`: The array to compute the median of.
            
            `method`: The method to use when computing the quantile. Default is 'linear'. See `numpy.quantile` for more information.
        
        Returns:
            A scalar value representing the median of the signal.
        """
        return median_tf(signal_window, method=method)
