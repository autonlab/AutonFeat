import numpy as np
from typing import Union

from autofeat.functional import quantile_tf

def median_tf(x: np.ndarray, method: str='linear') -> Union[float, np.float_]:
    """
    Compute the median of the values in `x`.

    Args:
        `x`: The array to compute the median of.
        
        `method`: The method to use when computing the quantile. Default is 'linear'. See `numpy.quantile` for more information.

    Returns:
        The median of the values in `x`.
    """
    
    return quantile_tf(x, q=0.5, method=method)
