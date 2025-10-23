import numpy as np

from wavely.signal.settings import settings

microphones = settings.MICROPHONES


def _get_microphone_parameters(microphone: str) -> dict:
    """Get the audio parameters of a microphone from the settings file.

    Args:
        microphone: Key name of the microphone.

    Returns:
        Return the parameters of the microphone in a dictionary and raise an error
            if it is not in the settings file.

    """
    try:
        microphone_parameters = microphones[microphone]
    except KeyError as e:
        raise ValueError(
            "Unknown microphone {}. Microphone must be one of {}".format(
                microphone, ", ".join(map(str, microphones.keys()))
            )
        ) from e
    return microphone_parameters


def is_clipped(blocks: np.ndarray) -> bool:
    """Check if signal is hard-clipped.

    Args:
        blocks: A shape-(n, k) array containing the audio signal split into blocks.

    Returns:
        True if the audio signal is hard-clipped.

    """
    return np.max(np.abs(blocks)) >= 1


def is_underloaded(spl: np.ndarray, microphone: str) -> bool:
    """Check if the microphone's output is underloaded.

    The `spl` values are compared to the microphone's background noise (BGN) which is
    calculated by the value 94 dB minus the microphone's signal noise ratio (SNR).
    Under the BGN, the microphone no longer works effectively as a sound sensor.

    Args:
        spl: Microphone's output sound pressure level (dB).
        microphone: Model of the microphone used for the acoustical recording. See
            `wavely.signal.settings` for the exhaustive list of microphones.

    Returns:
        True if the microphone's output is underloaded.

    """
    microphone_parameters = _get_microphone_parameters(microphone)
    mic_snr = microphone_parameters.get("SNR")

    try:
        mic_background = 94 - mic_snr
    except TypeError as e:
        raise ValueError(
            "Undefined SNR value for the microphone {}".format(microphone)
        ) from e

    return spl.min() <= mic_background


def is_overloaded(spl: np.ndarray, microphone: str) -> bool:
    """Check if the microphone's output is overloaded.

    The `spl` feature is compared to the microphone Acoustic Overload Point (AOP).
    Above the AOP, the microphone no longer works effectively as a sound sensor. The
    microphone's output signal reaches 10% of distortion.

    Args:
        spl: Microphone's output Sound Pressure Level (dB).
        microphone: Model of microphone used for the acoustical recording. See
            `wavely.signal.settings` for the exhaustive list of microphones.

    Returns:
        True if the microphone's output is distorted.

    """
    microphone_parameters = _get_microphone_parameters(microphone)
    mic_aop = microphone_parameters.get("AOP")
    try:
        return spl.max() >= mic_aop
    except TypeError as e:
        raise ValueError(
            "Undefined AOP value for the microphone {}".format(microphone)
        ) from e
