import doctest
from unittest import TestCase, TextTestRunner

import numpy as np
from scipy.signal import welch
from scipy.stats import linregress

from wavely.signal import generators
from wavely.signal.preprocessing.preprocessing import band_indexes


class DoctestTextTestRunner(TextTestRunner):
    def test_doctest(self):
        suite_generators = doctest.DocTestSuite(generators)
        self.run(suite_generators)


class SignalGenerationTestCase(TestCase):
    @classmethod
    def setUpClass(cls):
        cls.n = 10000
        cls.rate = 48000
        cls.nfft = 1024

        cls.array_with_nan = np.array(
            [
                [5, np.nan, np.nan, 7, 2],
                [3, np.nan, 1, 8, np.nan],
                [4, 9, 6, np.nan, np.nan],
            ]
        )
        cls.array_filled_nan = np.array(
            [[5, 5, 5, 7, 2], [3, 3, 1, 8, 8], [4, 9, 6, 6, 6]]
        )

    def test_pink_noise(self):
        np.random.seed(123)
        signal = generators.pink_noise(self.n)
        f, Pxx = welch(signal, fs=self.rate, nfft=self.nfft)
        band_freqs, band_index = band_indexes(self.rate, self.nfft, "third")
        Pxx_octave = np.zeros(band_index.size)
        Pxx_octave[0] = Pxx[0]
        for i in range(1, band_index.size):
            if band_index[i - 1] == band_index[i]:
                Pxx_octave[i] = Pxx[i]
            else:
                Pxx_octave[i] = Pxx[band_index[i - 1]:band_index[i]].mean()
        idx = band_freqs > 500.0
        slope = linregress(np.where(idx)[0], 20 * np.log10(Pxx_octave[idx]))

        # Test good generation of pink noise using its statistical properties
        # 1. Spectral density has merely a linear slope in loglog scale
        self.assertGreaterEqual(slope.rvalue ** 2, 0.99)
        # 2. The fitted hyperplan fairly represents the data
        self.assertLessEqual(slope.pvalue, 0.05)

        # TODO find suitable way to test spectral density decay

    def test_fill_nan(self):
        filled_array = generators.fill_nan(self.array_with_nan)
        np.testing.assert_array_equal(filled_array, self.array_filled_nan)
