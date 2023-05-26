import numpy as np
from typing import Union, Callable


def dft_tf(x: np.ndarray, n: Union[int, np.int_] = None, norm: str = 'backward', where: Callable[[Union[int, float, np.int_, np.float_]], Union[bool, np.bool_]] = lambda x: not np.isnan(x)) -> Union[float, np.float_]:
    """
    Compute the 1D discete Fourier Transform (DFT) using Fast-Fourier Transform (FFT) on the values in `x`.

    Args:
        `x`: The array to compute the DFT of.

        `n`: The number of points to use for the FFT. If `None`, the length of `x` is used. Default is `None`.

        `norm`: The normalization mode to use. Default is 'backward'. See `numpy.fft` for more information.
                    \tOptions include:\n
                    \t\t`'backward'`: The backward transform is scaled by `1/n`.\n
                    \t\t`'ortho'`: The forward and backward transforms are scaled by `1/sqrt(n)`.\n
                    \t\t`'forward'`: The forward transform is not scaled.\n

        `where`: A function that takes a value and returns `True` or `False`. Default is `lambda x: not np.isnan(x)` i.e. a measurement is valid if it is not a `NaN` value.

    Returns:
        The 1D DFT of `x`.
    """
    # Vectorize the where function
    where_fn = np.vectorize(where)
    filtered_x = x[where_fn(x)]
    return np.fft.fft(a=filtered_x, n=n, norm=norm, axis=0)