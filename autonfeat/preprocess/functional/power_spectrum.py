# Author(s): Dhruv Srikanth
# Email(s): dsrikant (at) andrew (dot) cmu (dot) edu
# Acknowledgements:
# Copyright (c) 2023 Carnegie Mellon University, Auton Lab
# This code is subject to the license terms contained in the code repo.

import numpy as np
from typing import Union, Callable
from autonfeat.preprocess.functional.dft import dft_tf


def power_spectrum_tf(x: np.ndarray, nfft: Union[int, np.int_] = None, normfft: str = 'backward', where: Callable[[Union[int, float, np.int_, np.float_]], Union[bool, np.bool_]] = lambda x: not np.isnan(x)) -> Union[float, np.float_]:
    """
    Compute the power spectrum on the values in `x`. This uses a 1D DFT to compute the power spectrum. See `autonfeat.preprocess.functional.dft` for more information.

    Args:
        x: The array to compute the power spectrum of.

        nfft: The number of points to use for the FFT (1D DFT). If `None`, the length of `x` is used. Default is `None`.

        normfft: The normalization mode to use when computng the FFT (1D DFT). Default is 'backward'. See `autonfeat.preprocess.functional.dft` for more information.
                    Options include:\n
                    'backward': The backward transform is scaled by `1/n`.
                    'ortho': The forward and backward transforms are scaled by `1/sqrt(n)`.
                    'forward': The forward transform is not scaled.

        where: A function that takes a value and returns `True` or `False`. Default is `lambda x: not np.isnan(x)` i.e. a measurement is valid if it is not a `NaN` value.

    Returns:
        The power spectrum of `x`.
    """
    # Vectorize the where function
    where_fn = np.vectorize(where)
    filtered_x = x[where_fn(x)]

    # Compute the DFT
    dft = dft_tf(x=filtered_x, n=nfft, norm=normfft, where=where)

    # Compute the power spectrum
    return np.abs(dft)**2
