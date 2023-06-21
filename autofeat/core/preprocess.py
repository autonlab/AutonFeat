import numpy as np
from typing import Any


class Preprocess(object):
    """
    Represents a preprocessor to apply to a signal.
    """

    # Dunder methods
    def __init__(self, name: str = "Not specified") -> None:
        """
        Initialize a new preprocessor.
        """
        self._name = name

    def __str__(self) -> str:
        """
        Get the string representation of the preprocessor.

        Returns:
            The string representation of the preprocessor.
        """
        return f"Preprocess({self._name})"

    def __repr__(self) -> str:
        """
        Get the string representation of the preprocessor.

        Returns:
            The string representation of the preprocessor.
        """
        return self.__str__()

    def __call__(self, signal: np.ndarray, *args: Any, **kwargs: Any) -> np.ndarray:
        """
        Apply the preprocessor to the signal provided.

        Args:
            signal: The signal window to apply the preprocessor to.

            *args: Additional arguments to pass to the preprocessor.

            **kwargs: Additional keyword arguments to pass to the preprocessor.

        Returns:
            The preprocessed signal.

        Raises:
            NotImplementedError: If the preprocessor is not implemented.
        """
        raise NotImplementedError("This method is not implemented.")

    # Getters and setters
    def get_name(self) -> str:
        """
        Get the name of the preprocessor.

        Returns:
            The name of the preprocessor.
        """
        return self._name

    def set_name(self, name: str) -> None:
        """
        Set the name of the preprocessor.

        Args:
            name: The new name of the preprocessor.
        """
        self._name = name
