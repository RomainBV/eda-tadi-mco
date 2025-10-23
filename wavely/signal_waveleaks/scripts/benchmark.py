"""Audiofeature code profiling.

Generate a sine wave signal and extract features.

"""
import random
import time

import numpy as np

from wavely.signal.features.features import FeaturesComputer

AMPLITUDE = 0.1
NOISE_AMPLITUDE = 0.1
IMPULSE_AMPLITUDE = 0.3
SAMPLE_RATE = 192000
DURATION = 30  # s
FREQUENCY = 23000  # Hz
N = DURATION * SAMPLE_RATE

BLOCK_SIZE = 4096
N_BLOCKS = 1000  # we will process 1000 blocks


def main():

    # Audio signal: sine wave + noise + some random impulses
    ########################################################

    t = np.arange(N) / SAMPLE_RATE
    y = AMPLITUDE * np.sin(2 * np.pi * FREQUENCY * t)
    y += np.random.rand(N) * NOISE_AMPLITUDE
    for _ in range(10):
        y[int(random.uniform(0, DURATION) * SAMPLE_RATE)] += IMPULSE_AMPLITUDE
    y = y[: -(y.size % BLOCK_SIZE) or None]
    # plt.plot(t[:1000], y[:1000])
    # plt.show()

    # compute features
    ##################

    y = y.reshape(-1, BLOCK_SIZE)
    y = y[:N_BLOCKS, :]
    print(
        "Number of blocks: {}. Duration: {}s".format(
            y.shape[0], round(float(y.shape[0] * BLOCK_SIZE) / SAMPLE_RATE)
        )
    )

    af = FeaturesComputer(block_size=BLOCK_SIZE, rate=SAMPLE_RATE)

    t0 = time.time()
    for i in range(N_BLOCKS):
        af.compute(y[i, :], flatten=True)
    t1 = time.time()
    print("Elapsed: {}s".format(t1 - t0))


if __name__ == "__main__":
    main()
