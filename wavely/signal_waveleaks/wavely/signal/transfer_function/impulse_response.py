import numpy as np


def select_impulse_response(
    deconvolved_signal: np.ndarray, window_boundary: list
) -> np.ndarray:
    """Select a limited part of the impulse response.

    Extract the impulse response from the raw deconvolved signal based on window
    boundaries. The window encompasses the impulse response. For example, a
    `window_boundary` of [100, 500] means that the response will be extracted on the
    following slice : [response_maximum_index - 100 : response_maximum_index + 500].

    Args:
        deconvolved_signal: deconvolved signal obtained after deconvolution
        window_boundary: left and right boundaries (as time vector indexes), starting
            from the impulse response maximum value index.

    Returns:
        The limited impulse response.

    """
    # Get boundaries
    left_boundary = window_boundary[0]
    right_boundary = window_boundary[1]
    # Get index of the impulse response maximum
    maximum_index = np.argmax(np.abs(deconvolved_signal))
    # Select i
    extracted_impulse_response = deconvolved_signal[
        slice(maximum_index - left_boundary, maximum_index + right_boundary)
    ]
    return extracted_impulse_response


def extract_impulse_response(
    excitation_signal: np.ndarray, measured_signal: np.ndarray
) -> np.ndarray:
    """Deconvolve measured signal and extract full impulse response.

    Deconvolve signal to get the whole impulse response signal.
    This signal contains the impulse response at its center and noise or echos
    on the rest of the signal.

    Args:
        excitation_signal: signal used to excite the system.
        measured_signal: signal recorded as response to the excitation signal.

    Returns:
        The deconvolved signal with the time reversed excitation signal.

    """
    # Deconvolve measured signal with time reversed excitation signal
    reversed_excitation_sig = excitation_signal[::-1]
    deconvolved_signal = np.convolve(reversed_excitation_sig, measured_signal)

    return deconvolved_signal


def select_impulse_response_over_window(
    deconvolved_signal: np.ndarray, window_boundary: dict
) -> list:
    """Extract windowed impulse responses with arbitrary number of windows.

    Args:
        deconvolved_signal: signal deconvolved containing the impulse response
        window_boundary: list of boundaries characterizing the windows on which
            the impulse response is extracted; Example [[1, 2], [5, 7]] are two
            sets of boundaries : [1, 2] and [5, 7].

    Returns:
        The list of windowed impulse responses.

    """
    impulse_response_windowed = []
    for wb in window_boundary:
        # Select impulse response in the deconvolved signal
        time_impulse_response = select_impulse_response(
            deconvolved_signal, window_boundary[wb]
        )
        impulse_response_windowed.append(time_impulse_response)

    return impulse_response_windowed
