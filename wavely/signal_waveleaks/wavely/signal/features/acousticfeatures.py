from typing import Callable, List, Optional

import numpy as np

from wavely.signal.features.features import feature, noexport
from wavely.signal.features.spectralfeatures import spectralcrest, spectralflatness
from wavely.signal.units.converters import db2lin, lin2db

REF_PRESSURE = 2.0e-5  # Pa
REF_PRESSURE_SQUARED = 4.0e-10  # Pa^2
REF_POWER = 1.0e-12  # W
REF_INTENSITY = 1.0e-12  # W/m^2
REF_SPECIFIC_ACOUSTIC_IMPEDANCE = 410.0  # Pa.s/m


@noexport
@feature
def enbw(window: np.ndarray) -> float:
    r"""Compute the equivalent noise bandwidth of a window.

    It is the width of a rectangular window that passes the same amount of
    noise.

    Args:
        window: A shape-(k,) array containing the window.

    Returns:
        The equivalent noise bandwidth.

    """
    # inherent power gain
    ipg = np.dot(window, window) / window.size
    # coherent power gain
    cpg = window.mean() ** 2
    return ipg / cpg


@feature("acoustic", dims=["block"])
def spl(rms: np.ndarray) -> np.ndarray:
    r"""Compute the sound pressure level (SPL) given a sound pressure RMS level.

    :math:`L_P = 10*\log_{10}(\frac{p^2}{p_0^2})`, where :math:`p` (in Pa) is an RMS
    level.

    Args:
        rms: A shape-(n,) array containing the RMS pressure values.

    Returns:
        A shape-(n,) containing the sound pressure level.

    """
    return 2.0 * lin2db(rms / REF_PRESSURE)


def spl2(sq_pressure: np.ndarray) -> np.ndarray:
    r"""Compute the sound pressure level (SPL) given a squared sound pressure RMS level.

    :math:`L_P = 10*\log_{10}(\frac{p^2}{p_0^2})`, where :math:`p` (in Pa) is an RMS
    level.

    Args:
        sq_pressure: A shape-(n,) array containing the squared RMS pressure values.

    Returns:
        A shape-(n,) array containing the sound pressure level.

    """
    return lin2db(sq_pressure / REF_PRESSURE_SQUARED)


def swl(acousticpower: np.ndarray) -> np.ndarray:
    r"""Compute the sound Power Level given an acoustic power.

    Sound power level (SWL) is the level (a logarithmic quantity) of the power
    :math:`P` (in W) of a sound relative to a reference power value :math:`P_0`.

    :math:`L_W = 10*\log_{10}(\frac{P}{P_0})`

    Args:
        acousticpower: A shape-(n,) array containing the Power values [W].

    Returns:
        A shape-(n,) array containing the sound power level values [dB SPL].

    """
    return lin2db(acousticpower / REF_POWER)


def sil(intensity: np.ndarray) -> np.ndarray:
    r"""Compute the sound intensity level (SIL).

    Sound intensity level (SIL) or acoustic intensity level is the level (a
    logarithmic quantity) of the intensity of a sound relative to a reference
    value. It is denoted :math:`L_I`, expressed in dB, and defined by:
    :math:`L_I = 10*\log_{10}(\frac{I}{I_0})`

    Args:
        intensity: A shape=(n,) array containing the intensity values [W/m^2].

    Returns:
        A shape-(n,) array containing the sound intensity level values [dB SPL].

    """
    return lin2db(intensity / REF_INTENSITY)


def intensitylevel(spectralpower: np.ndarray) -> np.ndarray:
    """Compute the sound intensity level using spectral values.

    Args:
        spectralpower: A shape-(n,) array containing the spectral power values.

    Returns:
        A shape-(n,) containing the intensity level values (dB).

    """
    return sil(spectralpower / REF_SPECIFIC_ACOUSTIC_IMPEDANCE)


def soundpressurelevel(spectralpower: np.ndarray) -> np.ndarray:
    """Compute the sound pressure level using spectral values.

    Args:
        spectralpower: A shape-(n,) array containing the spectral power values.

    Returns:
        A shape=(n,) containing the sound pressure level.

    """
    return spl2(spectralpower)


@feature("acoustic", dims=["block"])
def ultrasoundlevel(periodogram: np.ndarray, binfactor: float) -> np.ndarray:
    """Compute the ultrasound sound pressure level using spectral values.

    Args:
        periodogram: A shape-(n, m) array containing the periodogram. `n` is the
            number of audio blocks, and `m` the length of the periodogram.
        binfactor: The factor used to convert a bin index into a frequency.

    Returns:
        A shape-(n,) array containing the ultrasound intensity level values (dB).

    """
    binultrasound = int(np.ceil(2e4 / binfactor))
    ultrasound = periodogram[:, binultrasound:].sum(axis=-1)
    return spl2(ultrasound)


@feature("acoustic", dims=["block"])
def audiblelevel(periodogram: np.ndarray, binfactor: float) -> np.ndarray:
    """Compute the audible sound pressure level using spectral values.

    Args:
        periodogram: A shape-(n, m) array containing the periodogram. `n` is the
            number of audio blocks, and `m` the length of the periodogram.
        binfactor: The factor used to convert a bin index into a frequency.

    Returns:
        A shape-(n,) array containing the audible intensity level values (dB).

    """
    binaudible = int(np.ceil(2e4 / binfactor))
    audible = periodogram[:, :binaudible].sum(axis=-1)
    return spl2(audible)


@noexport
@feature
def band_periodogram(periodogram: np.ndarray, band_index: np.ndarray) -> np.ndarray:
    """Split the periodogram according to frequency bands.

    Args:
        periodogram: A shape-(n, m) array containing the periodogram. `n` is the
            number of audio blocks, and `m` the length of the periodogram.
        band_index: An array of indexes associated with the periodogram
            frequency axis, for each upper frequency of the bands specified
            in the FeaturesComputer object parameters.

    Returns:
        The periodogram, split over `b` frequency bands. The list holds `b` numpy
        arrays where the first axis has `n` elements (the blocks). The second
        axis is the frequency axis, whose size may not be equal for all of the
        list's elements.

    """
    assert len(periodogram.shape) == 2
    # Information above the last upper cutoff frequency is dropped.
    return np.split(periodogram, band_index, axis=-1)[:-1]


def apply_feature_over_bands(
    band_periodogram: List[np.ndarray], feature_fn: Callable[[np.ndarray], np.ndarray]
) -> np.ndarray:
    """Apply feature function `feature_fn` over each frequency band.

    Args:
        band_periodogram: The periodogram, split over `b` frequency bands. The list
            holds `b` numpy arrays where the first axis has `n` elements (the blocks).
            The second axis is the frequency axis, whose size may not be equal for all
            of the list's elements.
        feature_fn: Feature function to apply.

    Returns:
        A shape=(n, b) array containing the given feature for each
        calculated frequency band.

    """
    return np.array([feature_fn(x) for x in band_periodogram]).T


@feature("acoustic", dims=["block", "frequency_bands"])
def bandleq(band_periodogram: List[np.ndarray]) -> np.ndarray:
    """Compute the :math:`L_eq` level for a series of frequency bands.

    Args:
        band_periodogram: The periodogram, split over `b` frequency bands. The list
            holds `b` numpy arrays where the first axis has `n` elements (the blocks).
            The second axis is the frequency axis, whose size may not be equal for all
            of the list's elements.

    Returns:
        A shape-(n, b) array containing the level for each frequency band.

    """
    return apply_feature_over_bands(
        band_periodogram=band_periodogram, feature_fn=lambda bp: spl2(bp.sum(axis=-1))
    )


@feature("acoustic", dims=["block", "frequency_bands"])
def bandflatness(band_periodogram: List[np.ndarray]) -> np.ndarray:
    """Compute the spectral flatness for a series of frequency bands.

    Args:
        band_periodogram: The periodogram, split over `b` frequency bands. The list
            holds `b` numpy arrays where the first axis has `n` elements (the blocks).
            The second axis is the frequency axis, whose size may not be equal for all
            of the list's elements.

    Returns:
        A shape-(n, b) array containing the spectral flatness for
        each frequency band.

    """
    return apply_feature_over_bands(
        band_periodogram=band_periodogram,
        feature_fn=lambda bp: spectralflatness(
            periodogram=bp, spectralpower=bp.sum(axis=-1)
        ),
    )


@feature("acoustic", dims=["block", "frequency_bands"])
def bandcrest(band_periodogram: List[np.ndarray]) -> np.ndarray:
    """Compute the spectral crest for a series of frequency bands.

    Args:
        band_periodogram: The periodogram, split over `b` frequency bands. The list
            holds `b` numpy arrays where the first axis has `n` elements (the blocks).
            The second axis is the frequency axis, whose size may not be equal for all
            of the list's elements.

    Returns:
        A shape=(n, b) array containing the spectral crest for each
        frequency band.

    """
    return apply_feature_over_bands(
        band_periodogram=band_periodogram,
        feature_fn=lambda bp: spectralcrest(
            periodogram=bp, spectralpower=bp.sum(axis=-1)
        ),
    )


def leq(
    levels: np.ndarray, int_time: Optional[float] = 1.0, int_window: Optional[int] = 0
) -> np.ndarray:
    r"""Compute the equivalent level of a series of levels.

    :math:`L_{eq} = 10*\log{10}( \frac1{T} \int_0^T 10^{\frac{L}{10}} )`.

    Args:
        levels: Levels as function of time.
        int_time: Integration time of each level. Default is 1.0 second.
        int_window: If 0 compute the Leq of the whole array,
            else compute the Leq for each window of size int_window.

    Returns:
        The equivalent level.

    """
    levels = np.asarray(levels)
    int_window = int_window or levels.size
    # remove ending levels so that size is a multiple of int_window
    levels = levels[: -(levels.size % int_window) or None]
    # restructure array so that each row is a window
    levels = levels.reshape(-1, int_window)
    # time
    time = levels.shape[1] * int_time

    # compute leq
    return lin2db((db2lin(levels) / time).sum(axis=1))


def ln(levels: np.array, n: int) -> float:
    r"""Compute the :math:`n` th percentile level of a series of levels.

    Args:
        levels: a series of levels.
        n: Percentile to compute. If :math:`n` is 0, compute :math:`L_{\min}`,
                 if :math:`n` is 100, compute :math:`L_{\max}`.

    Returns:
        The :math:`n` th percentile level.

    """
    return np.percentile(levels, n)
