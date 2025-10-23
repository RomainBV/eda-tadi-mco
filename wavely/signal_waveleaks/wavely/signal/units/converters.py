import numpy as np

from wavely.signal.features import acousticfeatures, temporalfeatures


def lin2db(x: np.ndarray) -> np.ndarray:
    """Convert linear values to the decibel scale.

    Args:
        x: A shape-(n,) array containing the values to be converted.

    Returns:
        A shape-(n,) array containing the values in dB scale.

    """
    with np.errstate(divide="ignore"):
        return 10.0 * np.log10(x + np.finfo(float).eps)


def db2lin(x: np.ndarray) -> np.ndarray:
    """Convert decibel values to the linear scale.

    Args:
        x: A shape-(n,) array containing the values in dB scale to be converted.

    Returns:
        A shape-(n,) array containing the values in linear scale.

    """
    return 10.0 ** (x / 10.0)


def compensate_spl(x: np.ndarray, previous_spl: float) -> np.ndarray:
    """Compensate the current spl of the input signal by the previous one.

    Args:
        x: A shape-(n,) array containing the input signal.
        previous_spl: the spl used to compensate.

    Returns:
        A shape-(n,) array containing the compensated signal.

    """
    current_spl = acousticfeatures.spl(temporalfeatures.rms(temporalfeatures.power(x)))
    compensated_x = x * 10 ** ((previous_spl - current_spl) / 20)

    return compensated_x


def compensate_mean(x: np.ndarray, previous_mean: float) -> np.ndarray:
    """Compensate the current mean of the input signal by the previous one.

    Args:
        x: A shape-(n,) array containing the input signal.
        previous_mean: the mean used to compensate.

    Returns:
        A shape-(n,) array containing the compensated signal.

    """
    current_mean = x.mean()
    compensated_x = x + previous_mean - current_mean

    return compensated_x


def g_to_ms2(x: np.ndarray) -> np.ndarray:
    """Convert acceleration given in g to m/s².

    The standard gravity (g) equals 9.8 m/s².

    Args:
        x: A shape-(n,) array containing the values in g to be converted.

    Returns:
        A shape-(n,) array containing the values in m/s².

    """
    return x * 9.81


def ms2_to_g(x: np.ndarray) -> np.ndarray:
    """Convert acceleration given in m/s² to g.

    Args:
        x: A shape-(n,) array containing the values in m/s² to be converted.

    Returns:
        A shape-(n,) array containing the values in g.

    """
    return x / 9.81
