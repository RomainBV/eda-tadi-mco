import copy
from typing import Callable

import numpy as np


def make_fade_in(
    rate: float, duration: float = 0.2
) -> Callable[[np.ndarray], np.ndarray]:
    """Generate a fade in function.

    Args:
        rate: Sample rate in Hz.
        duration: The duration of the fade in in seconds. Defaults to 0.2.

    Returns:
        The fade in as a function.

    """
    if type(rate) is not float:
        raise ValueError("Please pass a float rate.")

    def fade_in(signal: np.ndarray) -> np.ndarray:
        """Perform a fade in on the input signal.

        Args:
            signal: A shape-(n,) array containing an input signal.

        Returns:
            A shape-(n,) array containing the faded in signal.

        Examples:
            >>> n = 10
            >>> rate = 20.
            >>> signal = np.ones(n)
            >>> faded_in_signal = fade_in(signal)
            >>> faded_in_signal
            array([0.        , 0.33333334, 0.66666669, 1.        , 1.        ,
               1.        , 1.        , 1.        , 1.        , 1.        ])

        """
        n_duration = np.ceil(duration * rate).astype(int)
        if signal.size < n_duration:
            raise ValueError(
                "Fade in duration is larger than the signal duration. "
                "Please decrease the fade in duration."
            )

        faded_in_signal = copy.deepcopy(signal)
        faded_in_signal[:n_duration] = faded_in_signal[:n_duration] * np.linspace(
            0, 1, n_duration, dtype="float32"
        )
        return faded_in_signal

    return fade_in


def make_fade_out(
    rate: float, duration: float = 0.2
) -> Callable[[np.ndarray], np.ndarray]:
    """Generate a fade out function.

    Args:
        rate: Sample rate in Hz.
        duration: The duration of the fade out in seconds. Defaults to 0.2.

    Returns:
        The fade out as a function.

    """
    if type(rate) is not float:
        raise ValueError("Please pass a float rate.")

    def fade_out(signal: np.ndarray) -> np.ndarray:
        """Perform a fade out on the input signal.

        Args:
            signal: A shape-(n,) array containing an input signal.

        Returns:
            A shape-(n,) array containing the faded out signal.

        Examples:
            >>> n = 10
            >>> rate = 20.
            >>> signal = np.ones(n)
            >>> faded_out_signal = fade_out(signal)
            >>> faded_out_signal
            array([1.        , 1.        , 1.        , 1.        , 1.        ,
               1.        , 1.        , 0.66666669, 0.33333334, 0.        ])

        """
        n = len(signal)
        n_duration = int(duration * rate)
        if n < n_duration:
            raise ValueError(
                "Fade out duration is larger than the signal duration. "
                "Please decrease the fade out duration."
            )

        faded_out_signal = copy.deepcopy(signal)
        faded_out_signal[-n_duration:] = faded_out_signal[-n_duration:] * np.linspace(
            1, 0, n_duration, dtype="float32"
        )
        return faded_out_signal

    return fade_out


def make_fade_in_out(
    rate: float, fade_in_duration: float = 0.2, fade_out_duration: float = 0.2
) -> Callable[[np.ndarray], np.ndarray]:
    """Generate a fade in and out function.

    Args:
        rate: Sample rate in Hz.
        fade_in_duration: The duration of the fade in in seconds. Defaults to 0.2.
        fade_out_duration: The duration of the fade out in seconds. Defaults to 0.2.

    Returns:
        The fade in out as a function.

    """

    def fade_in_out(signal: np.ndarray) -> np.ndarray:
        """Perform a fade in and out on the input signal.

        Args:
            signal: A shape-(n,) array containing an input signal.

        Returns:
            A shape-(n,) array containing the faded in and out signal.

        Examples:
            >>> n = 10
            >>> rate = 20.
            >>> signal = np.ones(n)
            >>> faded_in_out_signal = fade_in_out(signal)
            >>> faded_in_out_signal
            array([0.        , 0.33333334, 0.66666669, 1.        , 1.        ,
               1.        , 1.        , 0.66666669, 0.33333334, 0.        ])

        """
        fade_in = make_fade_in(rate, fade_in_duration)
        fade_out = make_fade_out(rate, fade_out_duration)
        return fade_out(fade_in(signal))

    return fade_in_out
