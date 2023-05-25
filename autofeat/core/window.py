import numpy as np
import numba as nb
from typing import Union, Callable, List


class SlidingWindow(object):
    """
    Represents a 1D sliding window over a time series signal.
    """
    def __init__(self, window_size: Union[int, np.int_], step_size: Union[int, np.int_], overflow: str = 'pad', padding: Union[int, float, np.int_, np.float_] = 0) -> None:
        """
        Initialize a new 1D sliding window.

        Args:
            `window_size`: The size of the window.

            `step_size`: The step size of the window.

            `overflow`: The method to take when there is an overflow of the window compared to the signal. Default is `pad`.
                          \tValid methods:
                          \t\t`restrict` - Reduces the window to the end of the signal (results in performance degradation due to broken cache lines).\n
                          \t\t`pad` - Pad signal with value passed to argument `padding`.\n
                          \t\t`stop` - Skips computation of feature over final window that contains overflow.

            `padding`: The value to pad the signal with in the case of an overflow. Default is `0`.
        """

        self._overflow_methods = [
            'restrict',
            'pad',
            'stop',
        ]

        # Checks
        self._check_window_size(window_size)
        self._check_step_size(step_size)
        self._check_overflow(overflow)

        self._window_size = window_size
        self._step_size = step_size
        self._overflow = overflow
        self._padding = padding

    def __str__(self) -> str:
        """
        Get the string representation of the sliding window.

        Returns:
            The string representation of the sliding window.
        """
        return f"Window(window_size={self._window_size}, step_size={self._step_size})"

    def __repr__(self) -> str:
        """
        Get the string representation of the sliding window.

        Returns:
            The string representation of the sliding window.
        """
        return self.__str__()

    # Getters and setters
    def get_window_size(self) -> Union[int, np.int_]:
        """
        Get the window size.

        Returns:
            The window size.
        """
        return self._window_size

    def get_step_size(self) -> Union[int, np.int_]:
        """
        Get the step size.

        Returns:
            The step size.
        """
        return self._step_size

    def get_overflow_methods(self) -> List[str]:
        """
        Get the overflow methods supported.

        Returns:
            A list of supported overflow methods.
        """
        return self._overflow_methods

    def get_overflow(self) -> str:
        """
        Get the overflow method.

        Returns:
            The overflow method.
        """
        return self._overflow

    def get_padding(self) -> Union[int, float, np.int_, np.float_]:
        """
        Get the padding value.

        Returns:
            The padding value.
        """
        return self._padding

    def set_window_size(self, window_size: Union[int, np.int_]) -> None:
        """
        Set the window size.

        Args:
            `window_size`: The window size.

        Raises:
            `TypeError`: If the window size is not an integer.
        """

        # Checks
        self._check_window_size(window_size)

        self._window_size = window_size

    def set_step_size(self, step_size: Union[int, np.int_]) -> None:
        """
        Set the step size.

        Args:
            `step_size`: The step size.

        Raises:
            `TypeError`: If the step size is not an integer.
        """
        # Checks
        self._check_step_size(step_size)

        self._step_size = step_size

    def set_overflow(self, overflow: str) -> None:
        """
        Set the overflow method.

        Args:
            `overflow`: The overflow method to set.

        Raises:
            `ValueError`: If the overflow method is not supported.
        """
        # Checks
        self._check_overflow(overflow)

        self._overflow = overflow

    def set_padding(self, padding: Union[int, float, np.int_, np.float_]) -> None:
        """
        Set the padding value.

        Args:
            `padding`: The padding value to set.
        """
        self._padding = padding

    # Checks
    def _check_signal(self, signal: np.ndarray) -> None:
        """
        Check if the signal is valid.

        Args:
            `signal`: The signal to check.

        Raises:
            `TypeError`: If the signal is not a numpy array.

            `Exception`: If the signal is not 1D.

            `ValueError`: If the window size is greater than the signal length.
        """
        if not isinstance(signal, np.ndarray):
            raise TypeError("Signal must be a numpy array.")

        if signal.ndim != 1:
            raise Exception("Signal must be 1D.")

        if self._window_size > signal.shape[0]:
            raise ValueError("Window size cannot be greater than signal length.")

    def _check_step_size(self, step_size: Union[int, np.int_]) -> None:
        """
        Check if the step size is valid.

        Args:
            `step_size`: The step size to check.

        Raises:
            `TypeError`: If the step size is not an integer.
        """

        if not isinstance(step_size, (int, np.int_)):
            raise TypeError("Step size must be an integer.")

    def _check_window_size(self, window_size: Union[int, np.int_]) -> None:
        """
        Check if the window size is valid.

        Args:
            `window_size`: The window size to check.

        Raises:
            `TypeError`: If the window size is not an integer.

            `ValueError`: If the window size is less than 1.
        """

        if not isinstance(window_size, (int, np.int_)):
            raise TypeError("Window size must be an integer.")

        if window_size < 1:
            raise ValueError("Window size must be greater than 0.")

    def _check_overflow(self, overflow):
        """
        Check if the overflow passed is a supported method of handling overflow.

        Args:
            `overflow`: The overflow method name to check.

        Raises:
            `ValueError`: If the method name passed is not a valid method. See documentation for more details on overflow handling methods supported.
        """
        supported_methods = self.get_overflow_methods()
        if overflow not in supported_methods:
            raise ValueError(f"{overflow} is not a supported method for handling window to signal overflow.\nSupported methods include {', '.join(supported_methods)}.")

    # Methods
    def use(self, transform: Callable[[np.ndarray], Union[np.float_, np.int_]]) -> Callable[[np.ndarray], np.ndarray]:
        """
        Use a transform function to transform each window.

        Args:
            `transform`: The transformation to apply to the signal.

        Returns:
            A function that applies the transformation to the signal using the sliding window.

        Raises:
            `TypeError`: If the transform is not callable.
        """

        # Checks
        if not callable(transform):
            raise TypeError("Transform must be callable.")

        def apply(signal: np.ndarray, start_idx: Union[int, np.int_] = 0, end_idx: Union[int, np.int_] = None) -> np.ndarray:
            """
            Apply the transformation to the signal using the sliding window.

            Args:
                `signal`: The signal to apply the transformation to.

                `start_idx`: The starting index of the window. Default is `0`.

                `end_idx`: The ending index of the window. Default is `None`.

            Returns:
                The transformed signal.
            """
            # Checks
            self._check_signal(signal)

            length = signal.shape[0]
            if end_idx is None:
                end_idx = length

            if start_idx < 0 or end_idx > length:
                raise IndexError("Window indices out of bounds. Start index must be greater than or equal to 0 and end index must be less than or equal to the signal length.")

            # Apply the transformation
            # transformed_signal = np.array(
            #     [
            #         transform(signal[i:i + self._window_size])           # Apply the transformation to the window
            #         # if i + self._window_size <= end_idx                  # This is important to avoid data leakage
            #         # else transform(signal[i:])                           # Apply the transformation to the last window
            #         for i in range(start_idx, end_idx, self._step_size)
            #     ]
            # )

            # transformed_signal = np.empty(shape=(len(nb.prange(start_idx, end_idx, self._step_size)),), dtype=np.float_)
            transformed_signal = []
            for i in nb.prange(start_idx, end_idx, self._step_size):
                if i + self._window_size <= end_idx:
                    transformed_signal.append(transform(signal[i:i + self._window_size]))
                else:
                    # Overflow occuring
                    if self._overflow == 'restrict':
                        # restrict the window to the end of the signal
                        transformed_signal.append(transform(signal[i:]))
                    elif self._overflow == 'pad':
                        # pad the signal by a value and the apply the transformation
                        n_over = i + self._window_size - end_idx
                        padded_signal = np.pad(array=signal[i:], pad_width=(0, n_over), constant_values=self._padding)
                        transformed_signal.append(transform(padded_signal))
                    else:
                        # stop computing features
                        break

            return transformed_signal

        return apply
