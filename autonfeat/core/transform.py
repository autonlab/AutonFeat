# MIT License

# Copyright (c) 2023 Carnegie Mellon University, Auton Lab

# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

import numpy as np
from typing import Union, Any


class Transform(object):
    """
    Represents a transformation to apply to a signal.
    """

    # Dunder methods
    def __init__(self, name: str = "Not specified") -> None:
        """
        Initialize a new transformation.
        """
        self._name = name

    def __str__(self) -> str:
        """
        Get the string representation of the transformation.

        Returns:
            The string representation of the transformation.
        """
        return f"Transform({self._name})"

    def __repr__(self) -> str:
        """
        Get the string representation of the transformation.

        Returns:
            The string representation of the transformation.
        """
        return self.__str__()

    def __call__(self, signal_window: np.ndarray, *args: Any, **kwargs: Any) -> Union[np.float_, np.int_]:
        """
        Apply the transformation to the signal window provided.

        Args:
            signal_window: The signal window to transform.

            *args: Additional arguments to pass to the transformation.

            **kwargs: Additional keyword arguments to pass to the transformation.

        Returns:
            A scalar value representing the transformation of the signal.

        Raises:
            NotImplementedError: If the transformation is not implemented.
        """
        raise NotImplementedError("This method is not implemented.")

    # Getters and setters
    def get_name(self) -> str:
        """
        Get the name of the transformation.

        Returns:
            The name of the transformation.
        """
        return self._name

    def set_name(self, name: str) -> None:
        """
        Set the name of the transformation.

        Args:
            name: The new name of the transformation.
        """
        self._name = name
