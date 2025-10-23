import numpy as np
from scipy.signal import hilbert
from scipy.stats import kurtosis

from wavely.signal.features.features import feature, noexport

#######################
# TIME BASED FEATURES #
#######################


@feature("temporal", dims=["block"])
def power(blocks: np.ndarray) -> np.ndarray:
    r"""Compute power mean of a block :math:`\frac{1}{N} \sum^N x^2`.

    Args:
        blocks: A shape-(n, k) array containing the signal blocks whose periodogram
            is to be computed along. `n` is the number of blocks and `k` the block
            size.

    Returns:
        A shape-(n,) array containing the average power for each block.

    """
    return np.mean(np.square(blocks), axis=-1)


@feature("temporal", dims=["block"])
def rms(power: np.ndarray) -> np.ndarray:
    r"""Compute the root mean square (RMS) of signal samples.

    The RMS is defined as :math:`\sqrt{\frac{1}{N} \sum^N x^2}`.

    Args:
        power: A shape-(n,) array containing the array of signal power values.

    Returns:
        A shape-(n,) array containing the RMS values.

    """
    if len(power.shape) > 1 and power.shape[1] > 1:
        raise ValueError(
            "Invalid input shape, should be like (n,) but passed {}. "
            "You should pass power values, not signal blocks.".format(power.shape)
        )
    return np.sqrt(power)


@noexport
@feature
def crest(blocks: np.ndarray) -> np.ndarray:
    r"""Compute the crest value :math:`\max(|s(t)|)` of a signal :math:`s`.

    Args:
        blocks: A shape-(n, k) array containing the signal blocks whose periodogram
            is to be computed along. `n` is the number of blocks and `k` the block
            size.

    Returns:
        A shape-(n,) array containing the crest value for each block.

    """
    return np.abs(blocks).max(axis=-1)


@noexport
@feature
def peak_to_peak(blocks: np.ndarray) -> np.ndarray:
    r"""Compute the peak-to-peak value of a signal.

    The peak-to-peak value of a signal :math:`s` is given by:

    :math:`\max(s(t)) - \min(s(t))`.

    Args:
        blocks: A shape-(n, k) array containing the signal blocks whose periodogram
            is to be computed along. `n` is the number of blocks and `k` the block
            size.

    Returns:
        A shape-(n,) array containing the peak to peak for each block.

    """
    return blocks.max(axis=-1) - blocks.min(axis=-1)


@feature("temporal", dims=["block"])
def crest_factor(crest: np.ndarray, rms: np.ndarray) -> np.ndarray:
    """Compute the crest factor of a signal.

    The crest factor is defined as the ratio between the crest value and the
    RMS.

    Args:
        crest: A shape-(n,) array containing the array of crest values.
        rms: A shape-(n,) array containing the array of RMS values.

    Returns:
        A shape-(n,) array containing the crest factor values.

    """
    assert crest.shape == rms.shape

    return np.divide(crest, rms, out=np.zeros_like(crest), where=rms != 0)


@feature("temporal", dims=["block"])
def temporal_kurtosis(blocks: np.ndarray) -> np.ndarray:
    """Compute the temporal kurtosis of signal blocks.

    The kurtosis is computed over the temporal signal values seen as a distribution.

    Args:
        blocks: A shape-(n, k) array containing the signal blocks whose periodogram
            is to be computed along. `k` is the number of blocks and `k` the
            block size.

    Return:
        A shape-(n,) array containing the kurtosis value for each block.

    """
    return kurtosis(blocks, axis=-1, fisher=False)


@feature
def zero_crossing_rate(blocks: np.ndarray) -> np.ndarray:
    """Compute the zero crossing rate of signal blocks.

    The zero crossing rate is the rate of sign-changes along a signal.

    Args:
        blocks: A shape-(n, k) array containing the signal blocks whose zero crossing
            rate is to be computed along. `n` is the number of blocks and `k` the
            block size.

    Return:
        A shape-(n,) array containing the zero crossing rate for each block.

    Examples:
       >>> np.random.seed(123)
       >>> blocks = np.random.randn(3, 16)
       >>> zero_crossing_rate(blocks)
       array([0.5   , 0.3125, 0.5625])

    """
    blocks_sign = np.signbit(blocks)

    crossings = np.pad(
        (blocks_sign[:, :-1] != blocks_sign[:, 1:]), [(0, 0), (0, 1)], mode="constant"
    )

    return np.mean(crossings, axis=1)


@noexport
@feature
def hilbert_transform(blocks: np.ndarray) -> np.ndarray:
    """Compute the analytic signal blocks, using Hilbert transform.

    The analytic signal :math:`x_a(t)` of signal :math:`x(t)` is:

    :math:`x_{a}=F^{-1}(F(x) 2 U)=x+i y`

    where F is the Fourier transform, U the unit step function,
    and y the Hilbert transform of x.

    Args:
        blocks: A shape-(n, k) array containing the signal blocks where `n` is the
            number of blocks and `k` the block size.

    Returns:
        A shape-(n, k) array containing the amplitude envelope of each block.

    References:
        [1] Leon Cohen, “Time-Frequency Analysis”, 1995. Chapter 2. Alan V.

        [2] Oppenheim, Ronald W. Schafer. Discrete-Time Signal Processing,
        Third Edition, 2009. Chapter 12. ISBN 13: 978-1292-02572-8

    """
    return hilbert(blocks, axis=-1)


@feature("spectral", dims=["block", "time"])
def amplitude_envelope(hilbert_transform: np.ndarray) -> np.ndarray:
    """Compute the amplitude envelope of signal blocks.

    The instantaneous amplitude is the magnitude of the analytic signal.

    Args:
        hilbert_transform: A shape-(n, k) array containing the analytic signal blocks
            where `n` is the number of blocks and `k` the block size.

    Returns:
        A shape-(n, k) array containing the amplitude envelope of each block.

    """
    return np.abs(hilbert_transform)
