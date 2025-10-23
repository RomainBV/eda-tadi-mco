import copy
from typing import Dict, Tuple

import numpy as np

from wavely.signal import settings
from wavely.signal.features import acousticfeatures, temporalfeatures
from wavely.signal.transfer_function import utils
from wavely.signal.units import converters, helpers


def transfer_function_coefficients(
    source_microphone: str,
    target_microphone: str,
    method: str = "amplitude_measurements",
) -> Tuple[np.ndarray, np.ndarray]:
    """Return polynomial coefficients of a transfer function.

    Return coefficients of the polynomial estimating the transfer function between
    source_microphone and target_microphone. Coefficients were pre-computed using
    `np.polyfit`. See the `notebook` directory of this package.

    Args:
        source_microphone: source microphone name.
        target_microphone: target microphone name.
        method: name of method used for transfer function parameters computation.

    Returns:
        2-tuple containing

        - A shape-(n,) array containing the weights of the numpy polynomial used
          to model the transfer function.
        - A shape-(2,) array containing the frequency domain where the transfer
          model is valid.

    """
    data = settings.settings.TRANSFER_PARAMETERS[method][
        utils.tf_coefficients_file[(source_microphone, target_microphone)]
    ]
    return data["weight"], data["domain"]


def generate_transfer_function(
    source_microphone: str,
    target_microphone: str,
    method: str = "amplitude_measurements",
    frequency_step: int = 1,
) -> Tuple[np.ndarray, np.ndarray]:
    """Get transfer function from a source to a reference microphone.

    Generate an estimate of the transfer function (in the frequency domain) from
    `source_microphone` spectrum magnitude to `target_microphone` spectrum magnitude.

    Args:
        source_microphone: source microphone name.
        target_microphone: target microphone name.
        method: name of method used for transfer function parameters computation.
        frequency_step: frequency step of the reconstructed spectrum.

    Returns:
        2-tuple containing

        - A shape-(n,) array containing the transfer function spectrum magnitude
          estimation.
        - A shape-(n,) array containing the frequency vector, whose step is
          frequency_step.

    """
    # Load weights and validity domain
    weight, domain = transfer_function_coefficients(
        source_microphone, target_microphone, method
    )
    f_H = np.arange(domain[0], domain[1], frequency_step)
    # Create polynomial model
    model = np.poly1d(weight)
    H = model(f_H)
    return H, f_H


def transform_LEQ_value(
    source_microphone: str,
    target_microphone: str,
    band_freq: list,
    leq_val: float,
    method: str = "amplitude_measurements",
    tol: float = 0.01,
    clip: Tuple[float, float] = (0, np.inf),
) -> float:
    """Transform a single L_eq band value.

    Transform an L_eq value using an estimation of the transfer function.

    Args:
        source_microphone: source microphone name.
        target_microphone: target microphone name.
        band_freq: boundaries of frequency band.
        leq_val: input pressure L_eq level, in dB.
        method: name of method used for transfer function parameters computation.
        tol: tolerance for transfer function validity range, expressed as a percentage
            of this range boundaries.
        clip: transfer function values will be clipped if they fall outside of this
            range.

    Returns:
        The transformed LEQ value

    """
    # Load transfer function
    H, f_H = generate_transfer_function(source_microphone, target_microphone, method)
    # validity domain
    begin_val = f_H[0]
    end_val = f_H[-1]

    if (band_freq[0] <= begin_val * (1 - tol)) or (band_freq[1] >= end_val * (1 + tol)):
        return np.NaN
    else:
        f_min = int(np.abs(f_H - band_freq[0]).argmin())
        f_max = int(np.abs(f_H - band_freq[1]).argmin())

        leq_factor = np.mean(np.clip(H[f_min:f_max], clip[0], clip[1]))
        return leq_val + 20 * np.log10(leq_factor)


# TODO: Delete this function if unused ?
def transform_LEQ_band(
    source_microphone: str,
    target_microphone: str,
    band_freq: list,
    feats: dict,
    db: bool = True,
    method: str = "amplitude_measurements",
) -> Dict[str, np.ndarray]:
    """Post process LEQ bands with an approximation of the transfer function.

    Make as if the LEQ bands were computed on a transformed spectrum. This
    approach leads to errors. Non transformed bands are set to NaN.

    Args:
        source_microphone: source microphone name.
        target_microphone: target microphone name.
        band_freq: frequency bands of the LEQ bands. Example : [1, 2] means
            [0, 1] Hz and [1, 2] Hz bands.
        feats: feature dictionary output by audiofeature.
        db: if True, return result in log (dB) scale, in linear scale otherwise.
        method: name of method used for transfer function parameters computation.

    Returns:
        The feature dictionary with additional key "transformed_bandleq"
        containing transformed LEQ bands

    """
    # Load transfer function
    H, f_H = generate_transfer_function(source_microphone, target_microphone, method)
    H = H / max(H)  # normalized transfer function
    # validity domain
    begin_val = f_H[0]
    end_val = f_H[-1]

    feature = copy.deepcopy(feats["bandleq"])

    for count, bound in enumerate(band_freq):
        # Add the implicit 0 Hz band beginning
        if count == 0:
            boundaries = [0, bound]
        else:
            boundaries = [band_freq[count - 1], bound]

        # transform LEQ bands based on transfer function
        if (boundaries[0] <= begin_val * 0.98) or boundaries[1] >= end_val * 1.01:
            # Add nan
            feature[:, count] = np.NaN * np.zeros(feature[:, count].shape)
        else:
            # Find boundaries of transfer function fitting the LEQ band
            fb = int(np.abs(f_H - boundaries[0]).argmin() + 1)
            ffin = int(np.abs(f_H - boundaries[1]).argmin() - 1)
            # computation of leq factor to apply
            leq_factor = np.abs(H[fb:ffin]) ** 2
            # (Factor squared because used with LEQ, which are power not amplitude)
            # Transform LEQ value
            if db:
                feature[:, count] = feature[:, count] + 10 * np.log10(
                    np.mean(leq_factor)
                )
            else:
                feature[:, count] = feature[:, count] * np.mean(leq_factor)
    # Add transformed LEQ values to feature dictionary
    feats["transformed_bandleq"] = feature
    return feats


def filter_frequency_domain(
    signal: np.ndarray, sample_rate: float, frequency_response: np.ndarray
) -> np.ndarray:
    """Filter a signal in the frequency domain.

    Args:
        signal: the signal to filter.
        sample_rate: the sampling frequency.
        frequency_response: the frequency response used to filter the signal.

    Returns:
        The filtered signal.

    """
    block_size = len(frequency_response)
    overlap_ratio = 0.5
    blocks = helpers.split_signal(
        signal=signal,
        rate=sample_rate,
        block_duration=block_size / sample_rate,
        overlap_ratio=overlap_ratio,
        zero_pad=True,
    )
    window = np.sqrt(np.hanning(block_size))[None, :]
    n_blocks = len(blocks)
    blocks = blocks * np.tile(window, (n_blocks, 1))

    filtered_blocks = np.zeros(blocks.shape)
    for k, block in enumerate(blocks):
        X = np.fft.fft(block, block_size)
        Y = X * frequency_response
        filtered_blocks[k, :] = np.real(np.fft.ifft(Y, block_size))
    filtered_blocks = filtered_blocks * np.tile(window, (n_blocks, 1))
    filtered_signal = helpers.add_and_merge_blocks(
        blocks=filtered_blocks, overlap_ratio=overlap_ratio
    )

    # crop signal due to zero padding when splitting the signal into blocks
    filtered_signal.resize(signal.shape, refcheck=False)

    return filtered_signal


def symmetrize_spectrum(spectrum: np.ndarray) -> np.ndarray:
    """Symmetrize the input spectrum.

    Args:
        spectrum: the input spectrum to symmetrize.

    Returns:
        The symmetrized spectrum.

    """
    return np.hstack([spectrum[:-1], spectrum[::-1]])


def _make_inverse_response(frequency_response: np.ndarray) -> np.ndarray:
    """Inverse a frequency response.

    Args:
        frequency_response: the input frequency response to inverse.

    Returns:
        The inverted frequency response.

    """
    h_db = converters.lin2db(frequency_response)
    mean_h = h_db.mean()
    inv_h_db = 2 * mean_h - h_db
    inv_frequency_response = converters.db2lin(inv_h_db)
    return inv_frequency_response


def _load_frequency_response(
    source_microphone: str, target_microphone: str, freq_max: float
) -> Tuple[np.ndarray, np.ndarray]:
    """Load a frequency response and make the inverse if needed.

    Args:
        source_microphone: the source microphone from which filter the signal.
        target_microphone: the target microphone to which filter the signal.
        freq_max: upper bound of the frequency domain.

    Returns:
        2-tuple containing

        - A shape-(n,) array containing the frequency response.
        - A shape-(2,) array containing its domain.

    """
    if source_microphone != "BK_4939" and target_microphone != "BK_4939":
        filt_source_bk, domain_source_bk = generate_transfer_function(
            source_microphone, "BK_4939", frequency_step=10
        )
        filt_bk_target, domain_bk_target = generate_transfer_function(
            target_microphone, "BK_4939", frequency_step=10
        )
        filt_bk_target = _make_inverse_response(filt_bk_target)

        fmax = min([domain_source_bk.max(), domain_bk_target.max()])
        shape = (max([filt_source_bk.shape[0], filt_bk_target.shape[0]]),)
        domain = (
            domain_bk_target
            if domain_bk_target.max() > domain_source_bk.max()
            else domain_source_bk
        )
        transfer_function = np.ones(shape)
        transfer_function[domain <= fmax] = (
            filt_source_bk[domain_source_bk <= fmax]
            * filt_bk_target[domain_bk_target <= fmax]
        )
    else:
        try:
            transfer_function, domain = generate_transfer_function(
                source_microphone, target_microphone, frequency_step=10
            )
        except KeyError:
            transfer_function, domain = generate_transfer_function(
                target_microphone, source_microphone, frequency_step=10
            )
            transfer_function = _make_inverse_response(transfer_function)

    transfer_function = transfer_function[domain <= freq_max]
    frequency_response = symmetrize_spectrum(transfer_function)

    return frequency_response, domain


def transfer_recording_microphone(
    signal: np.ndarray,
    sample_rate: float,
    source_microphone: str,
    target_microphone: str,
) -> np.ndarray:
    """Filter an audio signal by applying a transfer function.

    The signal is filtered so as to change its record microphone.

    Args:
        signal: the input audio signal.
        sample_rate: the sampling frequency.
        source_microphone: the source microphone from which filter the signal.
        target_microphone: the target microphone to which filter the signal.

    Returns:
        the filtered signal.

    """

    def compute_spl(x: np.ndarray) -> float:
        """Compute the SPL of an input signal.

        Args:
            x: the input signal.

        Returns:
            The SPL of the input signal.

        """
        return acousticfeatures.spl(temporalfeatures.rms(temporalfeatures.power(x)))

    freq_max = sample_rate // 2
    frequency_response, _ = _load_frequency_response(
        source_microphone, target_microphone, freq_max
    )

    signal_mean = signal.mean()
    signal -= signal_mean
    spl_signal = compute_spl(signal)
    filtered_signal = filter_frequency_domain(signal, sample_rate, frequency_response)
    filtered_signal = converters.compensate_spl(filtered_signal, spl_signal)
    filtered_signal = converters.compensate_mean(filtered_signal, signal_mean)

    return filtered_signal
