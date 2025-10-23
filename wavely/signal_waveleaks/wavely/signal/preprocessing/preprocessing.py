import warnings
from typing import Callable, Dict, List, Optional, Tuple, Union

import numpy as np
from scipy.signal import bilinear, butter, filtfilt, iirpeak, lfilter

from wavely.signal.settings import settings
from wavely.signal.units import helpers

# Constants from the international standard IEC 61672-1:2013.
# ---- Annex E - Analytical expressions for frequency-weightings C, A, and Z.-#

# Approximate values for pole frequencies f_1, f_2, f_3 and f_4.
# See section E.4.1 of the standard.
_POLE_FREQUENCIES = {1: 20.60, 2: 107.7, 3: 737.9, 4: 12194.0}

# Normalization constants :math:`C_{1000}` and :math:`A_{1000}`.
# See section E.4.2 of the standard.
_NORMALIZATION_CONSTANTS = {"A": -2.000, "C": -0.062}
# Octave and third-octave bands nominal frequencies.
_NOMINAL_CENTER_FREQ = {
    "octave": np.array(
        [31.5, 63.0, 125.0, 250.0, 500.0, 1000.0, 2000.0, 4000.0, 8000.0, 16000.0]
    ),
    "third": np.array(
        [
            25.0,
            31.5,
            40.0,
            50.0,
            63.0,
            80.0,
            100.0,
            125.0,
            160.0,
            200.0,
            250.0,
            315.0,
            400.0,
            500.0,
            630.0,
            800.0,
            1000.0,
            1250.0,
            1600.0,
            2000.0,
            2500.0,
            3150.0,
            4000.0,
            5000.0,
            6300.0,
            8000.0,
            10000.0,
            12500.0,
            16000.0,
            20000.0,
        ]
    ),
}


def calibrate_audio_signal(
    blocks: np.ndarray,
    microphone: str,
    preamp_gain: float = 0,
    rss_calibration: float = 0,
) -> np.ndarray:
    """Calibrate the audio signal.

    Args:
        blocks: A shape-(n, m) array containing the audio blocks whose periodogram
            is to be computed along. `n` is the number of blocks and `m`, the
            block size.
        microphone: Model of microphone used for the acoustical
            recording (dB V/Pa). See `settings.microphones`.
        preamp_gain: gain applied by the microphone preamplifier (dB).
            Defaults to 0.
        rss_calibration: Fine tuning correction of the environmental
            conditions based on a Reference Sound Source (RSS). Default 0.

    Returns:
        A shape-(n, m) array containing the calibrated audio blocks.

    """
    # get the microphone sensitivity
    try:
        mic = settings.MICROPHONES[microphone]
        mic_sensitivity = mic["sensitivity"]
    except (KeyError, AttributeError):
        raise ValueError(
            "Unknown microphone {}. Microphone must be one of {}".format(
                microphone, ", ".join(list(settings.MICROPHONES.keys()))
            )
        )

    # preamp gain correction (amplitude)
    a = blocks * np.power(10, -preamp_gain / 20.0)
    # mic sensitivity correction (amplitude)
    b = a * np.power(10, -mic_sensitivity / 20.0)
    # adc gain correction
    c = b * np.power(10, rss_calibration / 20.0)

    return c


def apply_gain(blocks: np.ndarray, gain: float = -40.0) -> np.ndarray:
    """Apply a gain to a signal.

    Args:
        blocks: A shape-(n, m) array containing the signal blocks whose periodogram
            is to be computed along. `n` is the number of blocks and `m`, the
            block size.
        gain: The gain to apply in dB. Defaults to -40dB.

    Returns:
        A shape-(n, m) array containing the normalised audio blocks.

    """
    return blocks * np.power(10, gain / 20)  # Maximum value = 134 dB SPL


def min_max_scaler(blocks: np.ndarray) -> np.ndarray:
    """Scale a signal by the min-max rule.

    The min-max scale rule is defined by:
    :math:`y = (x - min(x))/(max(x) - min(x))`.

    Args:
        blocks: A shape-(n, m) array containing the signal blocks whose periodogram
            is to be computed along. `n` is the number of blocks and `m`, the
            block size.

    Returns:
        A shape-(n, m) array containing the scaled signal blocks.

    """
    return (blocks - blocks.min()) / (blocks.max() - blocks.min())


def max_abs_scaler(blocks: np.ndarray, rate: float) -> np.ndarray:
    """Scale a signal by the maximum absolute value.

    This scaling is defined by:
    :math:`x = dc_offset(x)`,
    :math:`y = x / max(abs(x))`.

    Args:
        blocks: A shape-(n, m) array containing the signal blocks whose periodogram
            is to be computed along. `n` is the number of blocks and `m`, the
            block size.
        rate: Sampling frequency in Hz.

    Returns:
        A shape-(n, m) array containing the scaled signal blocks.

    """
    dc_offset = make_dc_offset(rate)
    blocks = np.array([dc_offset(block) for block in blocks])
    return blocks / np.abs(blocks).max()


def make_dc_offset(
    rate: float,
    cutoff: float = 15.0,
    filter_operation: Callable[
        [np.ndarray, np.ndarray, np.ndarray], np.ndarray
    ] = lfilter,
) -> Callable[[np.ndarray], np.ndarray]:
    """Generate a function that removes DC offset from a signal.

    Args:
        rate: Sampling frequency in Hz.
        cutoff: The cutoff frequency of the DC offset highpass filter function.
        filter_operation: The callable used to perform the filtering. See scipy
            filtering function documentation.

    Returns:
        The DC offset highpass filter function.

    """
    dc_offset_filter = make_butter_highpass_filter(
        cutoff, rate, filter_operation=filter_operation
    )

    def dc_offset(x: np.ndarray) -> np.ndarray:
        """Remove DC offset from a signal.

        Args:
            x: an array-like of dimension 1.

        Returns:
            The filtered array.

        """
        return dc_offset_filter(x)

    return dc_offset


def make_butter_highpass_filter(
    cutoff: float,
    rate: float,
    order: int = 5,
    filter_operation: Callable[
        [np.ndarray, np.ndarray, np.ndarray], np.ndarray
    ] = lfilter,
) -> Callable[[np.ndarray], np.ndarray]:
    """Generate a high-pass Infinite Impulse Response filter.

    Args:
        cutoff: High-pass cut-off frequency in Hz.
        rate: Audio sampling frequency in Hz.
        order: Order of the IIR filter. Default is 5.
        filter_operation: The callable used to perform the filtering. See scipy
            filtering function documentation.

    Returns:
        The high pass filter as a function.

    """
    butter_highpass_filter = make_butter_filter(
        cutoff=cutoff,
        rate=rate,
        filter_type="highpass",
        order=order,
        filter_operation=filter_operation,
    )

    return butter_highpass_filter


def make_butter_filter(
    cutoff: Union[float, List[float]],
    rate: float,
    filter_type: str = "lowpass",
    order: int = 5,
    filter_operation: Callable[
        [np.ndarray, np.ndarray, np.ndarray], np.ndarray
    ] = lfilter,
) -> Callable[[np.ndarray], np.ndarray]:
    """Generate an Infinite Impulse Response filter.

    Args:
        cutoff: cut-off frequency in Hz for low and high pass filters, list of
            cut-off frequencies in Hz for band pass filters.
        rate: Audio sampling frequency in Hz.
        filter_type: the type of filter to generate.
            Valid values are 'lowpass', bandpass', and 'highpass'.
        order: Order of the IIR filter.
        filter_operation: The callable used to perform the filtering. See scipy
            filtering function documentation.

    Returns:
        The filter as a function.

    """
    if not isinstance(cutoff, list):
        cutoff = [cutoff]
    cutoff = np.array(cutoff)
    if cutoff[-1] > rate // 2:
        raise ValueError(
            "The cutoff frequency is larger than the Nyquist frequency. "
            "Consider increase the sample rate or decrease the cutoff "
            "frequency"
        )

    nyquist_frequency = 0.5 * rate
    norm_cutoff = cutoff / nyquist_frequency
    b, a = butter(order, norm_cutoff, btype=filter_type, analog=False, output="ba")

    # TODO add method to butter_filter argument if needed
    def butter_filter(x: np.ndarray) -> np.ndarray:
        """Apply a Butterworth filter.

        Args:
            x: an array-like of dimension 1.

        Returns:
            The filtered array.

        """
        return filter_operation(b, a, x)

    return butter_filter


def overlap_add_window(block_size: int, block_step: int) -> np.ndarray:
    """Get the correct window for an overlap and add process.

    These windows are used to used COLA (Constant Overlap and Add). See
    https://www.dsprelated.com/freebooks/sasp/Overlap_Add_Decomposition.html

    Args:
        block_size: The block size.
        block_step: The step between two blocks.

    Returns:
        A shape-(n,) array containing the window.

    """
    if block_step == (block_size - 1) / 2:
        # Hamming window
        return 0.54 - 0.46 * np.cos(
            2 * np.pi * np.arange(block_size - 1) / (block_size - 1)
        )
    elif block_step == (block_size + 1) / 2:
        # Hanning window
        return 0.5 - 0.5 * np.cos(2 * np.pi * np.arange(block_size) / (block_size + 1))
    else:
        # Hanning window
        return 0.5 - 0.5 * np.cos(2 * np.pi * np.arange(block_size) / block_size)


def make_overlap_and_add_filter(
    rate: float,
    filter_callable: Callable,
    block_duration: float = 1.0,
    overlap_ratio: float = 0.5,
) -> np.ndarray:
    """Use Overlap and Add to perform filtering on a signal.

    Args:
        rate: Sample rate in Hz.
        filter_callable: A Callable used to filter the signal. Should be a filter
            created using `make_butter_filter` or equivalent.
        block_duration: Duration of the block used to split the signal, in seconds.
        overlap_ratio: The overlap ratio.

    Returns:
        The shape-(n,) array containing the filtered signal.

    """

    def overlap_and_add(signal: np.ndarray) -> np.ndarray:
        blocks = helpers.split_signal(
            signal=signal,
            rate=rate,
            block_duration=min(len(signal) / rate, block_duration),
            overlap_ratio=overlap_ratio,
        )
        block_size = blocks.shape[-1]
        block_step = block_size - int(np.ceil(overlap_ratio * block_size))
        window = np.sqrt(overlap_add_window(block_size, block_step))
        blocks = blocks * window

        filtered_blocks = np.array([filter_callable(block) for block in blocks])
        filtered_blocks = filtered_blocks * window

        remaining_signal = helpers.get_remaining_signal(
            signal=signal, blocks=blocks, overlap_ratio=overlap_ratio
        )
        if remaining_signal is not None:
            remaining_signal = filter_callable(remaining_signal)
        filtered_signal = helpers.add_and_merge_blocks(
            blocks=filtered_blocks,
            overlap_ratio=overlap_ratio,
            remaining_signal=remaining_signal,
        )

        return filtered_signal

    return overlap_and_add


class OverlapAndAddFilter:
    def __init__(
        self,
        filter_func: Callable[[np.ndarray], np.ndarray],
        window: Optional[np.ndarray] = None,
        block_duration: float = 0.1,
        rate: float = 96e3,
        overlap_ratio: float = 0.5,
        are_blocks_with_overlap: bool = False,
    ):
        """Initialize an OverlapAndAddFilter.

        Args:
            filter_func: The callable used to perform the filtering.
            window: A shape-(n,) array containing the window. If None, use the square
                root of a Hanning window.
            block_duration: duration of the block used to split the signal in seconds.
            rate: sample rate in Hz.
            overlap_ratio: The overlap ratio.
            are_blocks_with_overlap: If False, apply an overlap of overlap_ratio to
                the blocks.

        """
        self._block_duration = block_duration
        self._rate = rate
        self._block_size = int(self._block_duration * self._rate)
        self._overlap_ratio = overlap_ratio

        if window is None:
            block_step = self._block_size - int(
                np.ceil(overlap_ratio * self._block_size)
            )
            self._window = np.sqrt(overlap_add_window(self._block_size, block_step))
        self._window = self._window[None, :]
        self._filter_func = filter_func
        self._are_blocks_with_overlap = are_blocks_with_overlap

        self._last_block = np.zeros((1, self._block_size))
        self._filtered_blocks = []

    def push(self, blocks: np.ndarray):
        """Push a block of signal to filter.

        Args:
            blocks: A shape-(n_blocks, block_size) array containing the blocks.

        """
        if not self._are_blocks_with_overlap:
            # Reshape the blocks to have `self._overlap_ratio` overlap
            # in order to correctly apply overlap and add.
            blocks = helpers.split_signal(
                signal=helpers.add_and_merge_blocks(blocks, overlap_ratio=0),
                rate=self._rate,
                block_duration=self._block_duration,
                overlap_ratio=self._overlap_ratio,
            )
        # Append last block to ensure continuity between two batches
        # of blocks
        blocks = np.vstack([self._last_block, blocks])

        filtered_blocks = self.filter_blocks(blocks)
        self._filtered_blocks.append(filtered_blocks)

        self._last_block = blocks[-1:, :]

    def filter_blocks(self, blocks: np.ndarray) -> np.ndarray:
        """Filter the blocks with overlap and add.

        Args:
            blocks: A shape-(n,k) array containing the blocks to filter.

        Returns:
            A shape-(n,k) array containing the filtered blocks.

        """
        # The blocks are first windowed before being filtered.
        # The filtered blocks are again window in order to ensure a
        # perfect reconstruction when calling `add_and_merge_blocks`.
        filtered_blocks = self._filter_func(blocks * self._window) * self._window

        # Discard the first block which corresponds to the last block
        # of the previous call to `push`
        filtered_blocks = filtered_blocks[1:, :]

        if not self._are_blocks_with_overlap:
            overlap_split = 0
        else:
            overlap_split = self._overlap_ratio

        filtered_blocks = helpers.split_signal(
            signal=helpers.add_and_merge_blocks(
                filtered_blocks, overlap_ratio=self._overlap_ratio
            ),
            rate=self._rate,
            block_duration=self._block_duration,
            overlap_ratio=overlap_split,
        )
        return filtered_blocks

    def pop_blocks(self) -> Optional[np.ndarray]:
        """Return the filtered blocks.

        Returns:
            The filtered blocks.

        """
        if self._filtered_blocks:
            return self._filtered_blocks.pop()
        else:
            return None


def _weighting_system_a() -> Tuple[np.ndarray, np.ndarray]:
    """Compute the parameters of an A-weighting filter.

    Returns an A-weighting filter polynomial transfer function.

    Returns:
        2-tuple containing

        - A shape-(4,) array containing the numerator.
        - A shape-(4,) array containing the denominator.

    """
    f1 = _POLE_FREQUENCIES[1]
    f2 = _POLE_FREQUENCIES[2]
    f3 = _POLE_FREQUENCIES[3]
    f4 = _POLE_FREQUENCIES[4]

    offset = _NORMALIZATION_CONSTANTS["A"]
    numerator = np.array(
        [(2.0 * np.pi * f4) ** 2.0 * (10 ** (-offset / 20.0)), 0.0, 0.0, 0.0, 0.0]
    )

    part1 = [1.0, 4.0 * np.pi * f4, (2.0 * np.pi * f4) ** 2.0]
    part2 = [1.0, 4.0 * np.pi * f1, (2.0 * np.pi * f1) ** 2.0]
    part3 = [1.0, 2.0 * np.pi * f3]
    part4 = [1.0, 2.0 * np.pi * f2]

    denominator = np.convolve(np.convolve(np.convolve(part1, part2), part3), part4)

    return numerator, denominator


def make_a_filtering(rate: float) -> Callable[[np.ndarray], np.ndarray]:
    """Generate a A-weighting filter.

    Args:
        rate: The desired sampling rate in Hz.

    Returns:
        The filter as a function.

    """
    b, a = bilinear(*_weighting_system_a(), fs=rate)

    def a_filter(x: np.ndarray) -> np.array:
        """Apply A-weighting filter from audio signal.

        Args:
            x: an array-like of dimension 1.

        Returns:
            The filtered array.

        """
        filtered = lfilter(b, a, x)

        return filtered

    return a_filter


def _weighting_system_c() -> Tuple[np.ndarray, np.ndarray]:
    """Compute the parameters of an C-weighting filter.

    Returns an C-weighting filter polynomial transfer function.

    Returns:
        2-tuple containing

        - A shape-(2,) array containing the numerator.
        - A shape-(2,) array containing the denominator.

    """
    f1 = _POLE_FREQUENCIES[1]
    f4 = _POLE_FREQUENCIES[4]

    offset = _NORMALIZATION_CONSTANTS["C"]
    numerator = np.array(
        [(2.0 * np.pi * f4) ** 2.0 * (10 ** (-offset / 20.0)), 0.0, 0.0]
    )

    part1 = [1.0, 4.0 * np.pi * f4, (2.0 * np.pi * f4) ** 2.0]
    part2 = [1.0, 4.0 * np.pi * f1, (2.0 * np.pi * f1) ** 2.0]

    denominator = np.convolve(part1, part2)

    return numerator, denominator


def make_c_filtering(rate: float) -> Callable[[np.ndarray], np.ndarray]:
    """Generate a C-weighting filter.

    Args:
        rate: The desired sampling rate in Hz.

    Returns:
        The filter as a function.

    """
    b, a = bilinear(*_weighting_system_c(), fs=rate)

    def c_filter(x: np.ndarray) -> np.array:
        """Apply A-weighting filter from audio signal.

        Args:
            x: an array-like of dimension 1.

        Returns:
            The filtered array.

        """
        filtered = lfilter(b, a, x)

        return filtered

    return c_filter


def compensate_mems_resonance(
    rate: float, lowpass_cutoff: float = 10000.0, highpass_cutoff: float = 16000.0
) -> Callable[[np.ndarray], np.ndarray]:
    """Generate a MEMS-resonance filter.

    Note:
        Default arguments for `lowpass_cutoff` and `highpass_cutoff` are
        used for VESPER VM1000 microphone.

    Args:
        rate: The desired sampling rate in Hz.
        lowpass_cutoff: Cutoff frequency for the lowpass filter.
        highpass_cutoff: Cutoff frequency for the highpass filter.

    Returns:
        The filter as a function.

    """
    b_hp, a_hp = butter(9, highpass_cutoff / (rate / 2), "high")  # Highpass filter
    b_lp, a_lp = butter(6, lowpass_cutoff / (rate / 2), "low")  # Lowpass filter

    def mems_filter(signal: np.ndarray) -> np.array:
        """Apply MEMS-resonance filter from audio signal.

        Args:
            signal: audio input signal to filter.

        Returns:
            Filtered audio signal to reduce the resonance of the MEMS sensor.

        """
        filtred_hp = filtfilt(b_hp, a_hp, signal)
        filtred_lp = filtfilt(b_lp, a_lp, signal)
        filtred = filtred_hp + filtred_lp

        return filtred

    return mems_filter


def make_linear_filter(
    b_coeffs: Optional[Union[list, np.ndarray]] = None,
    a_coeffs: Optional[Union[list, np.ndarray]] = None,
) -> Callable[[np.ndarray], np.ndarray]:
    """Create a linear filter.

    Args:
        b_coeffs: Numerator coefficients (feedforward)
        a_coeffs: Denominator coefficients (feedback)

    Returns:
        Filter function that applies the linear filter following :
        a[0]*y[n] = b[0]*x[n] + b[1]*x[n-1] + ... + b[M]*x[n-M]
                  - a[1]*y[n-1] - ... - a[N]*y[n-N]
    """
    if b_coeffs is None:
        b_coeffs = [1]

    if len(b_coeffs) == 0:
        raise ValueError("b_coeffs cannot be empty")

    if a_coeffs is None:
        a_coeffs = [1]

    if not isinstance(b_coeffs, (list, np.ndarray)) or not isinstance(
        a_coeffs, (list, np.ndarray)
    ):
        raise ValueError("b_coeffs and a_coeffs must be lists or np.array")

    if a_coeffs[0] == 0:
        raise ValueError("a[0] cannot be zero (division by zero)")

    if a_coeffs[0] != 1.0:
        warnings.warn(
            "a[0] is not 1. All coefficients will be normalized by this value."
        )

    b_coeffs = np.array(b_coeffs, dtype=np.float64)
    a_coeffs = np.array(a_coeffs, dtype=np.float64)

    def linear_filter(signal: np.ndarray) -> np.ndarray:
        """Apply the linear filter.

        Args:
            signal: Input signal

        Returns:
            Filtered signal
        """
        filtered_signal = lfilter(b_coeffs, a_coeffs, signal)
        return filtered_signal  # type: ignore

    return linear_filter


def frequency_filters(
    sample_rates: Optional[List[int]] = None,
    cutoff: float = 30.0,
    order: int = 5,
    use_overlap_add: bool = False,
    overlap_add_block_duration: float = 1.0,
) -> Dict[str, Dict[int, Callable]]:
    """Generate frequency filters.

    Args:
        sample_rates: An array containing sampling frequencies
            that will be used to create the appropriate filters.
        cutoff: a cutoff frequency for filters.
        order: The order of Butterworth filters.
        use_overlap_add: If True, performs filtering using overlap and add
            algorithm.
        overlap_add_block_duration: duration of the block in seconds, used to
            split the signal in overlap and add filtering.

    Returns:
        A dictionary containing frequency filters. To each key in {'A','C', 'highpass'}
        corresponds a dictionary of filters, indexed by sampling frequency.

    """
    if sample_rates is None:
        sample_rates = [16e3, 44.1e3, 48e3, 96e3, 192e3]

    A_filters = {}
    C_filters = {}
    hp_filters = {}
    lp_filters = {}

    for sr in sample_rates:
        A_filters[sr] = make_a_filtering(sr)

        C_filters[sr] = make_c_filtering(sr)

        try:
            hp_filters[sr] = make_butter_filter(
                cutoff=cutoff, rate=sr, filter_type="highpass", order=order
            )
        except ValueError:
            hp_filters[sr] = None

        try:
            lp_filters[sr] = make_butter_filter(
                cutoff=cutoff, rate=sr, filter_type="lowpass", order=order
            )
        except ValueError:
            lp_filters[sr] = None

    filters = {
        "A": A_filters,
        "C": C_filters,
        "highpass": hp_filters,
        "lowpass": lp_filters,
    }

    if use_overlap_add:
        for _, filter in filters.items():
            filter[sr] = make_overlap_and_add_filter(
                rate=sr,
                filter_callable=filter[sr],
                block_duration=overlap_add_block_duration,
            )

    return filters


def band_indexes(
    rate: float, n_fft: int, freqs: Union[np.ndarray, str, float] = 5e3
) -> Tuple[np.ndarray, np.ndarray]:
    """Get frequency bands and their associated indexes.

    Computes the frequency bands upper bounds and the associated indexes in
    the periodogram frequency axis.

    Args:
        rate: The sampling rate.
        n_fft: Number of points used in the fft computation.
        freqs: If freqs is a float, it defines the bandwidth of bands.
            If it is an array, it defines the upper frequencies of bands.
            If it is a str, it should be either 'octave' or 'third'.

    Returns:
        2-tuple containing

        - A shape-(n,) array containing the frequency bands upper bounds
        - A shape-(n,) array containng the associated indexes.

    """
    delta_f = rate / n_fft
    if isinstance(freqs, np.ndarray):
        freqs = freqs[freqs <= rate / 2]
        return freqs, np.ceil(1 / delta_f * freqs).astype(int)
    elif isinstance(freqs, float):
        if freqs > rate // 2:
            raise ValueError(
                "Bandwidth larger than Nyquist frequency. Consider increasing \
                'rate'."
            )
        step = int(freqs / delta_f)
        if step == 0:
            raise ValueError(
                "Bandwidth smaller than frequency resolution. Consider increasing  \
                'band_freq'."
            )
        freqs = np.arange(freqs, rate // 2, freqs)
        return band_indexes(rate, n_fft, freqs)
    elif isinstance(freqs, str):
        freqs = octave_bands_frequencies(freqs)
        return band_indexes(rate, n_fft, freqs)
    else:
        raise TypeError(
            '"freqs" should be either a float, np.ndarray or str \
        in {"octave", "third"}.'
        )


def octave_bands_frequencies(band_type: str) -> np.ndarray:
    """Generate frequency octave or third-octave bands.

    Args:
        band_type: The desired band type. Should be either 'octave' or 'third'.

    Returns:
        A shape-(b,) array containing the frequency bands upper bounds.

    """
    if band_type not in {"octave", "third"}:
        raise ValueError('`band_type` should be in {"octave", "third"}')
    center_freq = _NOMINAL_CENTER_FREQ[band_type]
    exp = 1 / 2 if band_type == "octave" else 1 / 6
    return (2**exp) * center_freq


def hz_to_mel(frequencies: np.ndarray, htk: bool = False) -> np.ndarray:
    """Convert Hz to Mels.

    Args:
        frequencies: A shape-(n,) array containing scalar or array of frequencies.
        htk: use Hidden Markov Toolkit formula instead of Slaney. Defaults to False.

    Returns:
        A shape-(n,) array containing input frequencies in Mels.

    Examples:
        >>> hz_to_mel(60)
        0.8999999999999999
        >>> hz_to_mel([110, 220, 440])
        array([1.65, 3.3 , 6.6 ])

    """
    frequencies = np.asanyarray(frequencies)

    if htk:
        return 2595.0 * np.log10(1.0 + frequencies / 700.0)

    # Fill in the linear part
    f_min = 0.0
    f_sp = 200.0 / 3

    mels = (frequencies - f_min) / f_sp

    # Fill in the log-scale part

    min_log_hz = 1000.0  # beginning of log region (Hz)
    min_log_mel = (min_log_hz - f_min) / f_sp  # same (Mels)
    logstep = np.log(6.4) / 27.0  # step size for log region

    if frequencies.ndim:
        # If we have array data, vectorize
        log_t = frequencies >= min_log_hz
        mels[log_t] = min_log_mel + np.log(frequencies[log_t] / min_log_hz) / logstep
    elif frequencies >= min_log_hz:
        # If we have scalar data, heck directly
        mels = min_log_mel + np.log(frequencies / min_log_hz) / logstep

    return mels


def mel_to_hz(mels: np.ndarray, htk: bool = False) -> np.ndarray:
    """Convert mel bin numbers to frequencies.

    Args:
        mels: A shape=(n,) array containing mel bins to convert.
        htk: use Hidden Markov Toolkit formula instead of Slaney. Defaults to False.

    Returns:
        A shape-(n,) array containing input mels in Hz.

    Examples:
        >>> mel_to_hz(3)
        200.0
        >>> mel_to_hz([1,2,3,4,5])
        array([ 66.66666667, 133.33333333, 200.        , 266.66666667,
           333.33333333])

    """
    mels = np.asanyarray(mels)

    if htk:
        return 700.0 * (10.0 ** (mels / 2595.0) - 1.0)

    # Fill in the linear scale
    f_min = 0.0
    f_sp = 200.0 / 3
    freqs = f_min + f_sp * mels

    # And now the nonlinear scale
    min_log_hz = 1000.0  # beginning of log region (Hz)
    min_log_mel = (min_log_hz - f_min) / f_sp  # same (Mels)
    logstep = np.log(6.4) / 27.0  # step size for log region

    if mels.ndim:
        # If we have vector data, vectorize
        log_t = mels >= min_log_mel
        freqs[log_t] = min_log_hz * np.exp(logstep * (mels[log_t] - min_log_mel))
    elif mels >= min_log_mel:
        # If we have scalar data, check directly
        freqs = min_log_hz * np.exp(logstep * (mels - min_log_mel))

    return freqs


def fft_frequencies(sr: float = 22050.0, n_fft: int = 2048) -> np.ndarray:
    """Alternative implementation of `np.fft.fftfreq`.

    Args:
        sr: Audio sampling rate. Defaults to 22050.
        n_fft: FFT window size. Defaults to 2048.

    Returns:
        A shape=(1 + n_fft/2,) array containing the frequencies
        `(0, sr/n_fft, 2*sr/n_fft, ..., sr/2)`.

    Examples:
        >>> fft_frequencies(sr=22050, n_fft=16)
        array([    0.   ,  1378.125,  2756.25 ,  4134.375,  5512.5  ,  6890.625,
            8268.75 ,  9646.875, 11025.   ])

    """
    return np.linspace(0, float(sr) / 2, int(1 + n_fft // 2), endpoint=True)


def mel_frequencies(
    n_mels: int = 128, fmin: float = 0.0, fmax: float = 11025.0, htk: bool = False
) -> np.ndarray:
    """Compute an array of acoustic frequencies tuned to the mel scale.

    The mel scale is a quasi-logarithmic function of acoustic frequency
    designed such that perceptually similar pitch intervals (e.g. octaves)
    appear equal in width over the full hearing range.

    Because the definition of the mel scale is conditioned by a finite number
    of subjective psychoaoustical experiments, several implementations coexist
    in the audio signal processing literature [1]. By default, librosa replicates
    the behavior of the well-established MATLAB Auditory Toolbox of Slaney [2].
    According to this default implementation,  the conversion from Hertz to mel is
    linear below 1 kHz and logarithmic above 1 kHz. Another available implementation
    replicates the Hidden Markov Toolkit [3] (HTK) according to the following formula:

    `mel = 2595.0 * np.log10(1.0 + f / 700.0).`

    The choice of implementation is determined by the `htk` keyword argument:
    setting `htk=False` leads to the Auditory toolbox implementation, whereas
    setting it `htk=True` leads to the HTK implementation.

    Args:
        n_mels: Number of mel bins. Defaults to 128.
        fmin: Minimum frequency (Hz). Defaults to 0.
        fmax: Maximum frequency (Hz). Defaults to 11025.
        htk: If True, use Hidden Markov Tookit formula to convert Hz to mel.
            Otherwise (False), use Slaney's Auditory Toolbox. Defaults to False.

    Returns:
        A shape-(n_mels,) array containing n_mels frequencies in Hz which are
        uniformly spaced on the Mel axis.

    References:
        [1] Umesh, S., Cohen, L., & Nelson, D. Fitting the mel scale.
        In Proc. International Conference on Acoustics, Speech, and Signal Processing
        (ICASSP), vol. 1, pp. 217-220, 1998.

        [2] Slaney, M. Auditory Toolbox: A MATLAB Toolbox for Auditory
        Modeling Work. Technical Report, version 2, Interval Research Corporation, 1998.

        [3] Young, S., Evermann, G., Gales, M., Hain, T., Kershaw, D., Liu, X.,
        Moore, G., Odell, J., Ollason, D., Povey, D., Valtchev, V., & Woodland, P.
        The HTK book, version 3.4. Cambridge University, March 2009.

    Examples:
        >>> mel_frequencies(n_mels=40)
        array([    0.        ,    85.31725552,   170.63451104,   255.95176656,
             341.26902209,   426.58627761,   511.90353313,   597.22078865,
             682.53804417,   767.85529969,   853.17255522,   938.48981074,
            1024.85554588,  1119.11407321,  1222.04179301,  1334.43603258,
            1457.16745142,  1591.18678575,  1737.5322134 ,  1897.33739598,
            2071.84026081,  2262.39259049,  2470.47049443,  2697.68584352,
            2945.79875645,  3216.73123442,  3512.58204988,  3835.64300466,
            4188.41668329,  4573.63583931,  4994.2845644 ,  5453.62140461,
            5955.20460265,  6502.91966169,  7101.00944433,  7754.10703988,
            8467.27165444,  9246.02780196, 10096.40809975, 11025.        ])

    """
    # 'Center freqs' of mel bands - uniformly spaced between limits
    min_mel = hz_to_mel(fmin, htk=htk)
    max_mel = hz_to_mel(fmax, htk=htk)

    mels = np.linspace(min_mel, max_mel, n_mels)

    return mel_to_hz(mels, htk=htk)


def mel(
    sr: float,
    n_fft: float,
    n_mels: int = 128,
    fmin: float = 0.0,
    fmax: Optional[float] = None,
    htk: bool = False,
    norm: int = 1,
    dtype: np.dtype = np.float32,
) -> np.ndarray:
    """Create a Filterbank matrix to combine FFT bins into Mel-frequency bins.

    Args:
        sr: Sampling rate of the incoming signal.
        n_fft: Number of FFT components.
        n_mels: Number of Mel bands to generate. Defaults to 128.
        fmin: Lowest frequency (in Hz). Defaults to 0.
        fmax: Hhighest frequency (in Hz). If `None`, use `fmax = sr / 2.0`. Defaults to
            None.
        htk: use HTK formula instead of Slaney. Defaults to False.
        norm: {None, 1, np.inf} if 1, divide the triangular mel weights by
            the width of the mel band (area normalization).  Otherwise, leave all
            the triangles aiming for a peak value of 1.0. Defaults to 1.
        dtype: The data type of the output basis. By default, uses
            32-bit (single-precision) floating point.

    Returns:
        A shape-(n_mels, 1 + n_fft/2) array containing the Mel transform matrix.

    Examples:
        >>> melfb = mel(22050, 2048)
        >>> melfb
        array([[0.        , 0.        , 0.        , ..., 0.        , 0.        ,
            0.        ],
           [0.01618285, 0.        , 0.        , ..., 0.        , 0.        ,
            0.        ],
           [0.03236571, 0.        , 0.        , ..., 0.        , 0.        ,
            0.        ],
           ...,
           [0.        , 0.        , 0.        , ..., 0.        , 0.        ,
            0.00026052],
           [0.        , 0.        , 0.        , ..., 0.        , 0.        ,
            0.00013026],
           [0.        , 0.        , 0.        , ..., 0.        , 0.        ,
            0.        ]], dtype=float32)
        >>> # Clip the maximum frequency to 8KHz
        >>> mel(22050, 2048, fmax=8000)
        array([[0.        , 0.        , 0.        , ..., 0.        , 0.        ,
            0.        ],
           [0.01969188, 0.        , 0.        , ..., 0.        , 0.        ,
            0.        ],
           [0.03938375, 0.        , 0.        , ..., 0.        , 0.        ,
            0.        ],
           ...,
           [0.        , 0.        , 0.        , ..., 0.        , 0.        ,
            0.        ],
           [0.        , 0.        , 0.        , ..., 0.        , 0.        ,
            0.        ],
           [0.        , 0.        , 0.        , ..., 0.        , 0.        ,
            0.        ]], dtype=float32)

    """
    if fmax is None:
        fmax = float(sr) / 2

    if norm is not None and norm != 1 and norm != np.inf:
        raise ValueError("Unsupported norm: {}".format(repr(norm)))

    # Initialize the weights
    n_mels = int(n_mels)
    weights = np.zeros((n_mels, int(1 + n_fft // 2)), dtype=dtype)

    # Center freqs of each FFT bin
    fftfreqs = fft_frequencies(sr=sr, n_fft=n_fft)

    # 'Center freqs' of mel bands - uniformly spaced between limits
    mel_f = mel_frequencies(n_mels + 2, fmin=fmin, fmax=fmax, htk=htk)

    fdiff = np.diff(mel_f)
    ramps = np.subtract.outer(mel_f, fftfreqs)

    for i in range(n_mels):
        # lower and upper slopes for all bins
        lower = -ramps[i] / fdiff[i]
        upper = ramps[i + 2] / fdiff[i + 1]

        # .. then intersect them with each other and zero
        weights[i] = np.maximum(0, np.minimum(lower, upper))

    if norm == 1:
        # Slaney-style mel is scaled to be approx constant energy per channel
        enorm = 2.0 / (mel_f[slice(2, n_mels + 2)] - mel_f[:n_mels])
        weights *= enorm[:, np.newaxis]

    # Only check weights if f_mel[0] is positive
    if not np.all((mel_f[:-2] == 0) | (weights.max(axis=1) > 0)):
        # This means we have an empty channel somewhere
        warnings.warn(
            "Empty filters detected in mel frequency basis. "
            "Some channels will produce empty responses. "
            "Try increasing your sampling rate (and fmax) or "
            "reducing n_mels."
        )

    return weights.T


def _is_outlier(signal: np.ndarray, threshold: float) -> np.ndarray:
    """Detect outliers in signal.

    Args:
        points: The signal containing possible outliers.
        thresholds : The modified z-score to use as a threshold.
            Observations with a modified z-score (based on the median absolute
            deviation) greater than this value will be classified as outliers.

    Returns:
        A boolean array indicating outliers in signal.

    References:
        Boris Iglewicz and David Hoaglin (1993), "Volume 16: How to Detect and
        Handle Outliers", The ASQC Basic References in Quality Control:
        Statistical Techniques, Edward F. Mykytka, Ph.D., Editor.

    Examples:
        >>> signal = np.random.rand(10)
        >>> signal[5] = 10
        >>> mask = _is_outlier(signal)
        >>> mask
        array([False, False, False, False, False,  True, False, False, False,
            False])

    """
    if len(signal.shape) == 1:
        signal = signal[:, None]
    median = np.median(signal, axis=0)
    diff = np.sum((signal - median) ** 2, axis=-1)
    diff = np.sqrt(diff)
    med_abs_deviation = np.median(diff)

    if med_abs_deviation == 0:
        return np.zeros_like(signal, dtype=bool).squeeze()
    else:
        modified_z_score = 0.6745 * diff / med_abs_deviation
        return modified_z_score > threshold


def make_declipper(threshold: float = 10) -> Callable[[np.ndarray], np.ndarray]:
    """Generate a declipper for signal.

    Args:
        threshold: threshold use to detect outliers. Defaults to 10.

    Returns:
        The declipper as a function.

    """

    def declipper(signal: np.ndarray) -> np.ndarray:
        """Declip a signal.

        The declipper is used to reduce outlier sample in signal. For instance,
        the following signal is saturated because of outlier values:

        [1e-5, 2e-5, 0, -1e-5, 0.9, -2e-5]

        Therefore normalization could not properly be performed on this signal.

        Args:
            signal: A shape-(n,) array containing the input signal to declip, where
                 n is the signal length.

        Returns:
            A shape-(n,) array containing the declipped signal.

        Examples:
            >>> np.random.seed(123)
            >>> n = 100
            >>> signal = 0.01 * np.random.randn(n)
            >>> signal[n//2-5:n//2+5] *= 100
            >>> declipped_signal = declipper(signal)
            >>> declipped_signal[n//2-5:n//2+5]
            array([-0.01183049,  0.72849568,  0.12573528,  0.29803348,  0.68153379,
               -0.39406005, -0.31631989,  0.53097529, -0.24301693,  0.02968323])

        """
        idx_outliers = _is_outlier(signal / np.abs(signal).max(), threshold)
        if not any(idx_outliers):
            return signal

        outliers = signal[idx_outliers]
        outliers_std = outliers.std()
        if outliers_std == 0:
            return signal

        signal_std = signal.std()
        signal[idx_outliers] = outliers * (signal_std / outliers_std)
        return signal

    return declipper


def rectify(signal: np.ndarray) -> np.ndarray:
    """Rectify a signal and clip its values in [0,inf).

    Args:
        signal: the signal to rectify.

    Returns:
        The rectified signal.

    """
    return np.clip(signal, 0.0, np.inf)


def principal_argument(phases: np.ndarray) -> np.ndarray:
    """Compute the principal argument of a phase array.

    Args:
        phases: the phases used to compute the principal argument.

    Returns:
        The principal arguments of the phases.

    """
    return np.where(np.abs(phases) > np.pi, phases % np.pi - np.pi, phases)


def make_background_spectrum(
    periodogram: np.ndarray, estimator: str = "mean", percentile_level: float = 0.1
) -> np.ndarray:
    """Compute a background spectrum based on a periodogram.

    The background spectrum is computed by taking the mean along
    the time axis of a given periodogram. The periodogram is assumed to contain
    only background noise when `estimator` is set to "mean".

    Args:
        periodogram: A shape-(n,k) array containing the periodogram. `n` is the
            number of signal blocks, and `k` is the length of the periodogram.
        estimator: Defines how to compute the background spectrum. Use "mean" if you
            are sure the signal contains only background noise, else you can
            use "percentile". Defualts to "mean".
        percentile_level: Defines the percentile used by the corresponding estimator.
            Defaults to 0.1.

    Returns:
        A shape-(k,) array containing the background spectrum.

    """
    if estimator == "mean":
        return np.mean(periodogram, axis=0, keepdims=True)
    elif estimator == "percentile":
        return np.percentile(periodogram, q=percentile_level, axis=0, keepdims=True)
    else:
        raise ValueError(
            "{} is not a valid estimator to compute the background spectrum. "
            "Please use either `mean` or `percentile`."
        )


def simulate_resonance_peak(
    signal: np.ndarray,
    peak_frequency: float,
    lp_cutoff_frequency: float,
    hp_cutoff_frequency: float,
    Q: float,
    G: float,
    sample_rate: float,
    lp_order: int = 6,
    hp_order: int = 9,
) -> np.ndarray:
    """Simulate a resonance peak.

    The main purpose is to modify a mic frequency response by adding a resonance peak.
    To achieve this, a band extractor filter is applied to isolate the frequency range
    where to add resonance. Then a band-pass filter with a narrow bandwidth
    (high quality factor) and gain amplification will simulate resonance within the
    extracted frequency band.

    Args:
        signal: Audio time series.
        peak_frequency: Frequency cutoff of the expected resonance (Hz).
        lp_cutoff_frequency: Low cutoff frequency (Hz) for the band extractor filter.
        hp_cutoff_frequency: High cutoff frequency (Hz) for the band extractor filter.
        Q: Quality factor to use for the resonance peak. High quality factor leads to a
            narrow resonance peak.
        G: Gain to amplify resonance peak.
        sample_rate: Sampling rate.
        lp_order: Order of the low pass filter for the band extractor filter.
            Defaults to 6.
        hp_order: Order of the high pass filter for the band extractor filter.
            Defaults to 9.

    Returns:
        The filtered signal.

    """

    def band_extractor(
        lp_cutoff_frequency: float,
        hp_cutoff_frequency: float,
        lp_order: int,
        hp_order: int,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Band extractor filter.

        Use a combination of low pass filter and a high pass filter to make a band
        extractor filter. The band extractor works with a low pass filter at a
        `lp_cutoff_frequency` lower then the `hp_cutoff_frequency`. Otherwise it will
        have the effet of a band pass filter.

        Args:
            lp_cutoff_frequency: Low cutoff frequency (Hz) of the band extractor filter.
            hp_cutoff_frequency: High cutoff frequency (Hz) of the band extractor
                filter.
            lp_order: Order of the low pass filter for the band extractor filter.
            hp_order: Order of the high pass filter for the band extractor filter.

        Returns:
            hp_filtered_signal, lp_filtered_signal.

        """
        b_hp, a_hp = butter(
            hp_order, hp_cutoff_frequency, "high", fs=sample_rate
        )  # Highpass filter
        b_lp, a_lp = butter(
            lp_order, lp_cutoff_frequency, "low", fs=sample_rate
        )  # Lowpass filter

        filtred_hp_ = filtfilt(b_hp, a_hp, signal)
        filtred_lp_ = filtfilt(b_lp, a_lp, signal)

        return filtred_hp_, filtred_lp_

    # Design peak filter
    b, a = iirpeak(peak_frequency, Q, fs=sample_rate)
    # Gain amplification
    b = b * G
    resonance_peak = filtfilt(b, a, signal)

    filtered_hp, filtered_lp = band_extractor(
        lp_cutoff_frequency=lp_cutoff_frequency,
        hp_cutoff_frequency=hp_cutoff_frequency,
        lp_order=lp_order,
        hp_order=hp_order,
    )

    filtered = filtered_lp + resonance_peak + filtered_hp

    return filtered


def simulate_aging_mic_response(
    signal: np.ndarray,
    sample_rate: float,
    lp_cutoff_frequency: float = 10000.0,
    hp_cutoff_frequency: float = 500.0,
    lp_order: int = 1,
    hp_order: int = 1,
) -> np.ndarray:
    """Simulate aging mic response.

    Simulate an old mic frequency response by smoothing the extreme frequencies
    This is done by applying a large band pass filter (high + low pass filter).

    Args:
        signal: Audio time series.
        sample_rate: Sampling rate.
        lp_cutoff_frequency: Low cutoff frequency (Hz) for the band pass filter.
            Defaults to 10000.0.
        hp_cutoff_frequency: High cutoff frequency (Hz) for the band pass filter.
            Defaults to 500.0.
        lp_order: Order of the low pass filter. Defaults to 1.
        hp_order: Order of the high pass filter. Defaults to 1.

    Returns:
        The filtered signal.

    """
    # Low pass filter
    b_lp, a_lp = butter(lp_order, lp_cutoff_frequency, "low", fs=sample_rate)
    # High pass filter
    b_hp, a_hp = butter(hp_order, hp_cutoff_frequency, "high", fs=sample_rate)

    # Apply low pass filter to the high pass filtered signal
    filtered_high = filtfilt(b_hp, a_hp, signal)
    filtered_low_high = filtfilt(b_lp, a_lp, filtered_high)

    return filtered_low_high
