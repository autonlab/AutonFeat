import numpy as np

from typing import Callable, Union, Any

class Transform(object):
    """
    Represents a transformation to apply to a signal.
    """

    def __init__(self) -> None:
        """
        Initialize a new transformation.
        """
        self._name = "Not specified"
    
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

        """
        
        raise NotImplementedError("This method is not implemented.")

    