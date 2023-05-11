import numpy as np
from typing import Union

# numpy.quantile(a, q, axis=None, out=None, overwrite_input=False, method='linear', keepdims=False, *, interpolation=None)[source]

def quantile_tf(x: np.ndarray, q: Union[float, np.float_], method: str='linear') -> Union[float, np.float_]:
    """
    Compute the q-th quantile of the values in `x`.

    Args:
        `x`: The array to compute the q-th quantile of.

        `q`: The quantile to compute. `q` belongs to [0, 1].
        
        `method`: The method to use when computing the quantile. Default is 'linear'. See `numpy.quantile` for more information.

    Returns:
        The q-th quantile of the values in `x`.
    
    Raises:
        `ValueError`: If `q` is not in [0, 1].
    """
    if q < 0 or q > 1:
        raise ValueError('q must be in [0, 1].')
    
    return np.quantile(x, q, axis=0, method=method)