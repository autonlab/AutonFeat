# Author(s): Dhruv Srikanth
# Email(s): dsrikant (at) andrew (dot) cmu (dot) edu
# Acknowledgements:
# Copyright (c) 2023 Carnegie Mellon University, Auton Lab
# This code is subject to the license terms contained in the code repo.

import numpy as np
from typing import Union, Callable
from autonfeat.core import Preprocess
from autonfeat.preprocess.functional import power_spectrum_tf


class PowerSpectrumPreprocessor(Preprocess):
    """
    Power Spectrum using a 1D DFT.
    """
    # Dunder methods
    def __init__(self, name: str = "Power Spectrumm") -> None:
        super().__init__(name=name)

    def __call__(self, signal: np.ndarray, nfft: Union[int, np.int_] = None, normfft: str = 'backward', where: Callable[[Union[int, float, np.int_, np.float_]], Union[bool, np.bool_]] = lambda x: not np.isnan(x)) -> np.ndarray:
        """
        Compute the power spectrum on the values in `x`. This uses a 1D DFT to compute the power spectrum. See `autonfeat.preprocess.functional.dft` for more information.

        Args:
            signal: The array to compute the power spectrum of.

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
        return power_spectrum_tf(x=signal, nfft=nfft, normfft=normfft, where=where)
