import numpy as np

from typing import Union, Callable

def min_tf(x: np.ndarray, where: Callable[[Union[int, float, np.int_, np.float_]], Union[bool, np.bool_]]=None) -> Union[float, np.float_]:
    """
    Compute the min of the values in x where `where` is True.

    Args:
        x: The array to compute the min of.
        where: A function that takes a value and returns True or False. (Default: None)

    Returns:
        The min of the values in x where `where` is True.
    """
    if where is None:
        return np.amin(x, axis=0)
    return np.amin(x, axis=0, where=[where(x_i) for x_i in x])