# This example program can be used as follows to extract features from a set
# of audio files given as arguments.
#
# For examples :
#   python extract_features_from_audiofiles.py record1.wav record2.wav
#
# This program creates an hdf5 file for each input file.

import argparse

import numpy as np
import pandas as pd
import soundfile as sf

from wavely.signal.features import FeaturesComputer


def data_matrix_from_audio_signal(audio, sr, dt):
    """Extract a data matrix from a given audio signal.

    Args:
        audio (numpy.array): audio signal
        sr (float):          sample rate
        dt (float):          block length in seconds

    Returns:
        pandas.DataFrame: Each row represent a particular block (length dt)
                          and columns are the features computed in
                          the features package.

    """
    block_size = int(dt * sr)
    audio = audio[: -(audio.size % block_size) or None]
    audio = audio.reshape(-1, block_size)

    af = FeaturesComputer(block_size, rate=sr, window=np.hanning)

    features = []
    for i in range(len(audio)):
        block_features = af.compute(audio[i, :], flatten=True)
        features.append(block_features)
    df = pd.DataFrame(features)

    return df


def main(audiofiles):
    dt = 0.02
    for f in audiofiles:
        audio, sr = sf.read(f)
        df = data_matrix_from_audio_signal(audio, sr, dt)
        df.to_hdf(f + ".h5", key="features")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("audiofiles", metavar="filename", type=str, nargs="+")
    args = parser.parse_args()
    main(args.audiofiles)
