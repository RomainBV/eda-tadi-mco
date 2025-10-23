from typing import Optional

import numpy as np

from wavely.signal.features.features import feature, noexport
from wavely.signal.units import converters


def compensate_dft(
    dft: np.ndarray, n_fft: int, window: np.ndarray, inplace: bool = False
) -> np.ndarray:
    """Normalize and compensate the DFT with respect to power and frequencies.

    Args:
        dft: A shape-(n, n_fft//2+1) DFT for each block.
        n_fft: The size of the fft used for the DFT. See `numpy.fft.rfft`.
        window: A shape-(k,) window to be applied to the signal blocks.
        inplace: If True, modifies the DFT in place and returns None.

    Returns:
        A shape-(n, n_fft//2+1) DFT for each block.

    """
    if not inplace:
        dft = dft.copy()

    # normalize by n_fft and window size to get the power spectral density (/Hz /s)
    dft[:] /= n_fft * window.size
    # compensate for negative frequencies since |X[f]| = |X[-f]|
    dft[:, 1:] *= 2.0

    if inplace:
        return None

    return dft


@noexport
@feature
def dft(blocks: np.ndarray, n_fft: int, window: np.ndarray) -> np.ndarray:
    r"""Compute Discrete Fourier Transform (DFT) of signal blocks.

    Args:
        blocks: A shape-(n, k) signal blocks whose periodogram
            is to be computed along. `n` is the number of blocks and `k` the
            block size.
        n_fft: The size of the fft used for the DFT. See `numpy.fft.rfft`.
        window: A shape-(k,) window to be applied to the signal blocks.

    Returns:
        A shape-(n, n_fft//2+1) DFT for each block.

    """
    blocks = np.asarray(blocks).copy()
    blocks *= window
    # real signal DFT
    dft = np.fft.rfft(blocks, n_fft, axis=-1)

    return dft


@noexport
@feature
def periodogram(
    dft: np.ndarray,
    n_fft: int,
    window: np.ndarray,
    background_spectrum: Optional[np.ndarray] = None,
) -> np.ndarray:
    r"""Compute an estimation of the Power Spectral Density (PSD) through a periodogram.

    The Power Spectral Density is defined in
    :math:`(\text{physical_units})^2/Hz`.

    Args:
        dft: A shape-(n, n_fft//2+1) DFT for each block.
        n_fft: The size of the fft used for the periodogram. See `numpy.fft.rfft`.
        window: A shape-(k,) window to be applied to the signal blocks.
        background_spectrum: A shape-(k,) array containing the background spectrum.

    Returns:
        A shape-(n, n_fft//2+1) periodogram for each block.

    """
    power_spectrum = np.asarray(dft).copy()
    power_spectrum = (dft * dft.conj()).real
    compensate_dft(power_spectrum, n_fft, window, inplace=True)

    # Compute spectral emergence
    if background_spectrum is not None:
        background_db = converters.lin2db(background_spectrum).repeat(
            power_spectrum.shape[0], axis=0
        )
        power_db = converters.lin2db(power_spectrum)
        power_spectrum = converters.db2lin(power_db - background_db)

    return power_spectrum


@feature("spectral", dims=["block", "frequency"])
def spectrum(
    dft: np.ndarray,
    blocks: np.ndarray,
    n_fft: int,
    window: np.ndarray,
    background_spectrum: Optional[np.ndarray] = None,
) -> np.ndarray:
    r"""Compute the complex spectrum of signal blocks.

    Args:
        dft: A shape-(n, n_fft//2+1) DFT for each block.
        blocks: A shape-(n, k) signal blocks whose periodogram
            is to be computed along. `n` is the number of blocks and `k` the
            block size.
        n_fft: The size of the fft used for the periodogram. See `numpy.fft.rfft`.
        window: A shape-(k,) window to be applied to the signal blocks.
        background_spectrum: A shape-(k,) array containing the background spectrum.

    Returns:
        A shape-(n, n_fft//2+1) spectrum for each block.

    """
    spectrum = np.asarray(dft).copy()

    # Compute spectral emergence
    if background_spectrum is not None:
        background_db = converters.lin2db(background_spectrum).repeat(
            spectrum.shape[0], axis=0
        )
        power_db = converters.lin2db((spectrum * spectrum.conj()).real)
        spectrum = converters.db2lin(power_db - background_db) * np.exp(
            1j * np.angle(spectrum)
        )

    compensate_dft(spectrum, n_fft, window, inplace=True)

    # compensate for `n_fft` lower than window size.
    # However, window size should be always higher than `n_fft`
    if n_fft < window.size:
        spectrum[:] *= np.sum(blocks ** 2) / np.sum(blocks[:, :n_fft] ** 2)

    return spectrum


@feature("spectral", dims=["block", "mel_bands"])
def melspectrogram(spectrum: np.ndarray, mel_basis: np.ndarray) -> np.ndarray:
    r"""Compute the spectrogram in Mel scale of the spectrum.

    Args:
        spectrum: A shape-(n, n_fft//2 + 1) array containing the spectrum for each
            block.
        mel_basis: the mel coefficient basis.

    Returns:
        A shape-(n, n_mels) array containing the melspectrogram for each block.

    """
    spectrogram = np.power(np.abs(spectrum), 2)
    melspectrogram = np.dot(spectrogram, mel_basis)

    return melspectrogram


@noexport
@feature
def binfactor(rate: float, n_fft: int) -> float:
    r"""Compute the frequency resolution.

    It is the scalar such that the discrete spectrum evaluated at index `n`
    yields the value of the continuous spectrum evaluated at frequency `n * binfactor`.

    Args:
        rate: The sampling rate of the corresponding signal.
        n_fft: The size of fft computation.

    Returns:
        binfactor: The frequency resolution.

    """
    return rate / n_fft


@feature("spectral", dims=["block"])
def spectralcentroid(
    periodogram: np.ndarray, spectralpower: np.ndarray, binfactor: float
) -> np.ndarray:
    r"""Compute the spectral centroid.

     The spectral centroid is the weighted mean of the frequencies in the signal.
     It is close to the brightness indicator.

    Args:
        periodogram: A shape-(n, m) array containing the periodogram. `n` is the
            number of signal blocks, and `m` the length of the periodogram.
        spectralpower: A shape-(n) array containing the spectral power for each signal
            block.
        binfactor: The factor used to convert a bin index into a frequency.

    Returns:
        A shape-(n,) array containing the spectral centroid values.

    """
    assert periodogram.shape[0] == spectralpower.shape[0]

    freq_bins = np.arange(periodogram.shape[-1])
    eps = np.finfo(float).eps
    bincentroid = np.multiply(freq_bins, periodogram).sum(axis=-1) / (
        spectralpower + eps
    )
    return bincentroid * binfactor


@feature("spectral", dims=["block"])
def spectralspread(
    periodogram: np.ndarray, spectralcentroid: np.ndarray, binfactor: float
) -> np.ndarray:
    r"""Compute the spread of a periodogram.

    The spectral spread is the standard deviation of the spectral density
    computed over the periodogram frequency bins.

    Args:
        periodogram: A shape-(n, m) array containing the periodogram. `n` is the
            number of signal blocks, and `m` the length of the periodogram.
        spectralcentroid: A shape-(n) array containing the spectral centroid for each
            signal block.
        binfactor: The factor used to convert a bin index into a frequency.

    Returns:
        A shape-(n,) array containing the spectral spread values.

    """
    assert periodogram.shape[0] == spectralcentroid.shape[0]

    if np.all(periodogram == 0):
        return np.zeros((periodogram.shape[0]))

    nbins = periodogram.shape[-1]
    # frequencies minus centroid
    deviation = np.linspace(
        0, nbins * binfactor, num=nbins, endpoint=False
    ) - np.reshape(spectralcentroid, (-1, 1))
    deviation[:] **= 2

    spread = np.sqrt(np.ma.average(deviation, weights=periodogram, axis=-1))
    return spread


@feature("spectral", dims=["block"])
def spectralskewness(
    periodogram: np.ndarray,
    spectralcentroid: np.ndarray,
    spectralspread: np.ndarray,
    binfactor: float,
) -> np.ndarray:
    r"""Compute the periodogran skewness over the frequency bins.

    Skewness of a distrubtion is given by
    :math:`\mathbb{E} \left[ \left(\frac{X - \mu}{\sigma} \right)^3 \right]`
    where :math:`\mu` is the centroid and :math:`\sigma` is the standard deviation,
    i.e. the square root of the variance.

    Args:
        periodogram: A shape-(n, m) array containing the periodogram. `n` is the
            number of signal blocks, and `m` the length of the periodogram.
        spectralcentroid: A shape-(n) array containing the spectral centroid for each
            signal block.
        spectralspread: A shape-(n) array containing the spectral spread for each signal
            block.
        binfactor: The factor used to convert a bin index into a frequency.

    Returns:
        A shape-(n,) array containing the spectral skewness values.

    """
    assert periodogram.shape[0] == spectralcentroid.shape[0]
    assert periodogram.shape[0] == spectralspread.shape[0]

    if np.all(periodogram == 0):
        return np.zeros((periodogram.shape[0]))

    nbins = periodogram.shape[-1]
    # frequencies minus centroid
    deviation = np.linspace(
        0, nbins * binfactor, num=nbins, endpoint=False
    ) - np.reshape(spectralcentroid, (-1, 1))
    deviation[:] **= 3

    skewness = np.ma.average(deviation, weights=periodogram, axis=-1)
    skewness /= np.power(spectralspread, 3.0)

    return skewness


@feature("spectral", dims=["block"])
def spectralkurtosis(
    periodogram: np.ndarray,
    spectralcentroid: np.ndarray,
    spectralspread: np.ndarray,
    binfactor: float,
) -> np.ndarray:
    r"""Compute the periodogram kurtosis over the frequency bins.

    The kurtosis of a distribution is given by
    :math:`\mathbb{E} \left[ \left( \frac{X - \mu}{\sigma} \right)^4 \right]`
    where :math:`\mu` is the centroid and :math:`\sigma` is the distribution
    standard deviation.

    Args:
        periodogram: A shape-(n, m) array containing the periodogram. `n` is the
            number of signal blocks, and `m` the length of the periodogram.
        spectralcentroid: A shape-(n) array containing the spectral centroid for each
            signal block.
        spectralspread: A shape-(n) array containing the spectral spread for each signal
            block.
        binfactor: The factor used to convert a bin index into a frequency.

    Returns:
        A shape-(n,) array containing the spectral kurtosis values.

    """
    assert periodogram.shape[0] == spectralcentroid.shape[0]
    assert periodogram.shape[0] == spectralspread.shape[0]

    if np.all(periodogram == 0):
        return np.zeros((periodogram.shape[0]))

    nbins = periodogram.shape[-1]
    # frequencies minus centroid
    deviation = np.linspace(
        0, nbins * binfactor, num=nbins, endpoint=False
    ) - np.reshape(spectralcentroid, (-1, 1))
    deviation[:] **= 4

    kurtosis = np.ma.average(deviation, weights=periodogram, axis=-1, returned=False)
    kurtosis /= np.power(spectralspread, 4)

    return kurtosis


@noexport
@feature
def spectralpower(periodogram: np.ndarray) -> np.ndarray:
    """Compute the total power of the periodogram.

     .. note:: This might be different to time-based power due to windowing

    Args:
        periodogram: A shape-(n, m) array containing the periodogram. `n` is the
            number of signal blocks, and `m` the length of the periodogram.

    Returns:
        A shape-(n,) array containing the spectral power values.

    """
    return periodogram.sum(axis=-1)


@feature("spectral", dims=["block"])
def spectralflatness(periodogram: np.ndarray, spectralpower: np.ndarray) -> np.ndarray:
    r"""Compute the spectral flatness.

    Spectral flatness quantifies how noise-like a sound is, as opposed to
    being tone-like.
    :math:`SF(t_n) = \frac{\left(\prod_{m=1}^M a_m(t_n)\right)^{1 / M}}{\frac{1}{M}
    \sum_{m=1}^{M} a_m(t_n)}`

    Args:
        periodogram: A shape-(n, m) array containing the periodogram. `n` is the
            number of signal blocks, and `m` the length of the periodogram.
        spectralpower: A shape-(n) array containing the spectral power for each signal
            block.

    Returns:
        A shape-(n,) array containing the spectral flatness values.

    """
    assert periodogram.shape[0] == spectralpower.shape[0]

    eps = np.finfo(float).eps

    nbins = periodogram.shape[-1]
    log_x = np.log(periodogram + eps)

    return np.exp(np.mean(log_x, axis=-1)) / (spectralpower / nbins + eps)


@feature("spectral", dims=["block"])
def spectralcrest(periodogram: np.ndarray, spectralpower: np.ndarray) -> np.ndarray:
    r"""Compute the spectral crest factor.

    The crest factor measures the ratio between the power density peak value
    and the total block power.
    :math:`SC(t_n) = \frac{\max_{m} a_m(t_n)}{\frac{1}{M} \sum_{m=1}^{M} a_m(t_n)}`

    Args:
        periodogram: A shape-(n, m) array containing the periodogram. `n` is the
            number of signal blocks, and `m` the length of the periodogram.
        spectralpower: A shape-(n) array containing the spectral power for each signal
            block.

    Returns:
        A shape-(n,) array containing the spectral crest factor values.

    """
    assert periodogram.shape[0] == spectralpower.shape[0]

    eps = np.finfo(float).eps
    nbins = periodogram.shape[-1]

    return np.max(periodogram, axis=-1) / (spectralpower / nbins + eps) + eps


@noexport
@feature
def normspectrum(periodogram: np.ndarray, spectralpower: np.ndarray) -> np.ndarray:
    """Compute the normalized power density periodogram.

    The normalized power periodogram is useful when one needs to compute
    features that are independent of the total power.

    Args:
        periodogram: A shape-(n, m) array containing the periodogram. `n` is the
            number of signal blocks, and `m` the length of the periodogram.
        spectralpower: A shape-(n) array containing the spectral power for each signal
            block.

    Returns:
        A shape-(n, m) array containing the normalized periodogram.

    """
    assert periodogram.shape[0] == spectralpower.shape[0]

    eps = np.finfo(float).eps
    periodogram_norm = periodogram / (spectralpower.reshape(-1, 1) + eps)

    assert periodogram.shape == periodogram_norm.shape  # temp check

    return periodogram_norm


@feature("spectral", dims=["block"])
def spectralentropy(normspectrum: np.ndarray) -> np.ndarray:
    """Compute the spectral Entropy.

    Args:
        normspectrum: A shape-(n, m) array containing the normalised periodogram. `n` is
            the number of signal blocks, and `m` the length of the periodogram.

    Returns:
        A shape-(n,) array containing the spectral entropy values.

    """
    return -np.multiply(normspectrum, np.log(normspectrum + np.finfo(float).eps)).sum(
        axis=-1
    )


@feature("spectral", dims=["block"])
def spectralflux(periodogram: np.ndarray, prev_periodogram: np.ndarray) -> np.ndarray:
    """Compute the spectral flux.

    The spectral flux measures the signal amplitude variations over time through an
    :math:`L_1` norm. It only detects onsets using a rectifier function.

    Args:
        periodogram: A shape-(n, m) array containing the periodogram. `n` is the
            number of signal blocks, and `m` the length of the periodogram.
        prev_periodogram: A shape-(1, m) array containing the previous periodogram.

    Returns:
        A shape-(n,) array containing the spectral flux values.

    """
    prev = np.roll(periodogram, 1, axis=0)
    prev[0, :] = prev_periodogram

    # Half wave rectifier(only increasing in energy)
    hwr = np.clip(periodogram - prev, 0.0, np.inf)
    return np.sum(hwr, axis=-1)


@feature("spectral", dims=["block"])
def peakfreq(periodogram: np.ndarray, binfactor: float) -> np.ndarray:
    """Compute the peak frequency.

    The peak frequency is associated with the frequency bin with maximum
    spectral power density.

    Args:
        periodogram: A shape-(n, m) array containing the periodogram. `n` is the
            number of signal blocks, and `m` the length of the periodogram.
        binfactor: The factor used to convert a bin index into a frequency.

    Returns:
        A shape-(n,) array containing the preak frequency values.

    """
    return periodogram.argmax(axis=-1) * binfactor


@feature("spectral", dims=["block"])
def highfrequencycontent(
    periodogram: np.ndarray, spectralpower: np.ndarray, binfactor: float
) -> np.ndarray:
    """Compute the high frequency content.

    High frequency content is the sum of the amplitudes weighted by the
    squared frequency. It indicates the energy associated with high frequencies.

    Args:
        periodogram: A shape-(n, m) array containing the periodogram. `n` is the
            number of signal blocks, and `m` the length of the periodogram.
        spectralpower: A shape-(n) array containing the spectral spectralpower for each
            signal block.
        binfactor: The factor used to convert a bin index into a frequency.

    Returns:
        A shape-(n,) array containing the high frequency content values.

    """
    nbins = periodogram.shape[-1]
    sqfreqs = np.square(np.linspace(0, nbins * binfactor, num=nbins, endpoint=False))
    eps = np.finfo(float).eps
    return np.sum(sqfreqs * periodogram, axis=-1) / (spectralpower + eps)


@feature("spectral", dims=["block"])
def spectralirregularity(periodogram: np.ndarray) -> np.ndarray:
    """Compute the spectral irregularity.

    Spectral irregularity is defined as the sum of absolute differences of
    the power spectral density.

    Args:
        periodogram: A shape-(n, m) array containing the periodogram. `n` is the
            number of signal blocks, and `m` the length of the periodogram.

    Returns:
        A shape-(n,) array containing the spectral irregularity values.

    """
    return np.sum(np.abs(np.diff(periodogram, axis=-1)), axis=-1)


@feature("spectral", dims=["block"])
def spectralrolloff(periodogram: np.ndarray, binfactor: float) -> np.ndarray:
    """Compute the spectral roll-off.

    Spectral roll-off is defined as the frequency below which 95% of the power
    spectral density is concentrated. It is hence the 95th percentile of the PSD seen as
    a distribution.

    Args:
        periodogram: A shape-(n, m) array containing the periodogram. `n` is the
            number of signal blocks, and `m` the length of the periodogram.
        binfactor: The factor used to convert a bin index into a frequency.

    Returns:
        A shape-(n,) array containing the spectral roll-off values.

    """
    quantile = 0.95
    total_energy = np.cumsum(periodogram, axis=-1)
    threshold = quantile * total_energy[:, -1]

    ix = np.argmax(total_energy.T > threshold, axis=0)
    freq_bins = np.arange(periodogram.shape[-1])
    return np.multiply(freq_bins, binfactor)[ix]


@feature("spectral", dims=["block", "time"])
def instantaneous_phase(hilbert_transform: np.ndarray) -> np.ndarray:
    """Compute the instantaneous phase of signal blocks.

    The instantaneous phase is the angle of the analytic signal.

    Args:
        hilbert_transform: A shape-(n, k) array containing the analytic signal blocks
            where `n` is the number of blocks and `k` the block size.

    Returns:
        A shape-(n, k) array containing the instantaneous phase of each block.

    """
    return np.angle(hilbert_transform)


@feature("spectral", dims=["block", "time"])
def instantaneous_frequency(instantaneous_phase: np.ndarray, rate: float) -> np.ndarray:
    """Compute the instantaneous frequency of signal blocks.

    The instantaneous frequency can be obtained by differentiating the instantaneous
    phase.

    Args:
        instantaneous_phase: A shape-(n, k) array containing the instantaneous phase of
            each block where `n` is the number of instantaneous phases and `k` its size.
        rate: The sampling rate.

    Returns:
        A shape-(n, k) array containing the instantaneous frequency of each block.

    """
    return (
        np.diff(np.unwrap(instantaneous_phase, axis=-1), axis=-1) / (2.0 * np.pi) * rate
    )
