from typing import Dict, Tuple

import numpy as np

# Note: VESPER_VM1000_A and VESPER_VM1000_B are mean VESPER frequency
# responses computed using microphone evaluation campaign data
# (see audio-signal-processing-experiments notebook).
tf_coefficients_file = {
    ("AVISOFT_40008", "BK_4939"): "coef_avisoft_to_bk",
    ("BEYERDYNAMIC_MM1", "BK_4939"): "coef_beyerdynamic_to_bk",
    ("KNOWLES_SPH0645LM4H", "BK_4939"): "coef_I2S_to_bk",
    ("VESPER_VM1000", "BK_4939"): "coef_vesper_to_bk",
    ("VESPER_VM1000_A", "BK_4939"): "coef_vesper_A_to_bk",
    ("VESPER_VM1000_B", "BK_4939"): "coef_vesper_B_to_bk",
}


def compute_fourier_transform(
    signal: np.ndarray, sample_rate: float, dimension: int = 0
) -> Tuple[np.ndarray, np.ndarray]:
    """Compute the fourier transform of a numpy array.

    Args:
        signal: A shape-(n,) array containing the signal in the time domain.
        sample_rate: signal sampling frequency.
        dimension: dimension along which to compute the FFT.

    Return:
        2-tuple containing

        - A shape-(n,) containing the spectrum of signal `signal`.
        - A shape-(n,) containing the associated frequency vector.

    """
    delta_t = 1 / sample_rate
    spectrum = np.fft.fft(signal, axis=dimension)
    freq_vector = np.fft.fftfreq(spectrum.shape[-1], d=delta_t)

    # Shift DFT outputs for visualization.
    s_spectrum = np.fft.fftshift(spectrum)
    s_f = np.fft.fftshift(freq_vector)
    return s_spectrum, s_f


def microphone_model(metadata: Dict[str, dict], filename: str, channel: int) -> str:
    """Get microphone model associated with a recording.

    Args:
        metadata: metadata dictionary.
        filename: name of the associated recording file.
        channel: channel number.

    Returns:
        The microphone model name.

    """
    nb_channel = metadata[filename]["input_settings"]["n_channels"]
    if nb_channel == 1:
        chan = ""
    else:
        chan = channel
    return metadata[filename]["input_settings"]["microphone{}".format(chan)]
