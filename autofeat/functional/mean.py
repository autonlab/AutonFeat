import numpy as np

from typing import Union, Callable

def mean_tf(x: np.ndarray, where: Callable[[Union[int, float, np.int_, np.float_]], Union[bool, np.bool_]]=None) -> Union[float, np.float_]:
    """
    Compute the mean of the values in x where `where` is True.

    Args:
        x: The array to compute the mean of.
        where: A function that takes a value and returns True or False. (Default: None)

    Returns:
        The mean of the values in x where `where` is True.
    """
    if where is None:
        return np.mean(x)
    return np.mean(x, where=[where(x_i) for x_i in x])