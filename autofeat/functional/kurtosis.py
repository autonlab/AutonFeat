import numpy as np
from typing import Union, Callable


def kurtosis_tf(x: np.ndarray, fischer: Union[bool, np.bool_] = True, where: Callable[[Union[int, float, np.int_, np.float_]], Union[bool, np.bool_]] = lambda x: not np.isnan(x)) -> Union[float, np.float_]:
    """
    Compute the krutosis of the values in `x` where `where` is `True`.\n
    The krutosis is a measure of the "tailedness" of a distribution. It is defined as the fourth standardized moment of a distribution, and is calculated as:
    ```
    kurtosis = mean((x - mean(x)) / std(x)) ** 4
    ```

    Args:
        `x`: The array to compute the krutosis of.

        `fischer`: Whether to use Fisher's definition of kurtosis i.e. subtract 3 from the result. Default is `True`. If `False`, the result is the Pearson's definition of kurtosis.

        `where`: A function that takes a value and returns `True` or `False`. Default is `lambda x: not np.isnan(x)` i.e. a measurement is valid if it is not a `NaN` value.

    Returns:
        The krutosis of the values in `x` where `where` is `True`.
    """
    # Vectorize where fn
    where_fn = np.vectorize(pyfunc=where)
    filtered_x = x[where_fn(x)]

    # Compute kurtosis
    sample_mean = np.mean(filtered_x)
    sample_std = np.std(filtered_x)
    kurtosis = np.mean(((filtered_x - sample_mean) / sample_std) ** 4)

    # Fisher's definition
    kurtosis = kurtosis - 3 if fischer else kurtosis

    return kurtosis
