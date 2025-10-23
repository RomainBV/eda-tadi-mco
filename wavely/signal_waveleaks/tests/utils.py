from typing import List

import numpy as np
import pandas as pd

from wavely.signal.features.features import FeaturesComputer
from wavely.signal.preprocessing.preprocessing import calibrate_audio_signal
from wavely.signal.units.helpers import split_signal


def relative_change(reference, observed):
    return np.abs((reference - observed)) / np.abs(reference).max()


def extract_features(
    data: np.ndarray,
    sample_rate: float,
    frame_length: float,
    features: List[str],
    microphone: str,
    preamp_gain: float,
) -> pd.DataFrame:
    """ Extract the selected features for the given audio signal

    Args:
        data: A single channel audio signal.
        sample_rate: The sample rate of the input audio signal.
        frame_length: The length of the frames on which features are computed.
        features: Features to be computed.
        microphone: Microphone type. See calibrate_audio_signal().
        preamp_gain: Gain applied by the microphone preamplifier.
            See calibrate_audio_signal().

    Returns:
        pd.DataFrame: A time indexed DataFrame containing the expected features
            indexed with seconds starting from 0.

    """
    window_length = int(frame_length * sample_rate)
    num_frames = int((len(data) - 1) / (0.5 * window_length)) - 1  # -1 TO avoid error

    blocks = split_signal(data, sample_rate, frame_length, 0.5)

    nfft = int(frame_length * sample_rate)
    features_processor = FeaturesComputer(
        block_size=window_length, rate=sample_rate, nfft=nfft, features=features
    )

    normalized_blocks = calibrate_audio_signal(
        blocks=blocks, preamp_gain=preamp_gain, microphone=microphone
    )
    blocks_features = features_processor.compute(normalized_blocks, flatten=True)

    timestamps = [i * frame_length / 2 for i in range(num_frames)]
    return pd.DataFrame(blocks_features, index=pd.to_timedelta(timestamps, unit="s"))
