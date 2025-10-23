import doctest
import unittest

import numpy as np

from wavely.signal.features import temporalfeatures
from wavely.signal.features.features import FeaturesComputer

# TODO: refactor this, there is no need to go through the FeaturesComputer
# class to test the individual feature functions !


class DoctestTextTestRunner(unittest.TextTestRunner):
    def test_doctest(self):
        suite = doctest.DocTestSuite(temporalfeatures)
        self.run(suite)


class TestTemporalFeatures(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.block_size = 256
        cls.rate = 192000
        # simple sin signal
        cls.f = 750  # Hz (in order to have a integer number of periods)
        cls.n_fft = int(cls.block_size * 3.2)  # 8192
        cls.x = np.sin(
            2.0
            * np.pi
            * cls.f
            * np.arange(0.0, cls.block_size / cls.rate, 1 / cls.rate)
        )
        cls.af = FeaturesComputer(
            block_size=cls.block_size,
            rate=cls.rate,
            n_fft=cls.n_fft,
            window=np.ones,
            features="all",
        )
        cls.feats = cls.af.compute(np.expand_dims(cls.x, 0))

        # long sin signal
        cls.block_size_2 = cls.block_size * 10
        cls.n_fft_2 = int(cls.block_size_2 * 3.2)  # 8192
        cls.x_2 = np.sin(
            2.0
            * np.pi
            * cls.f
            * np.arange(0.0, cls.block_size_2 / cls.rate, 1 / cls.rate)
        )
        cls.af_2 = FeaturesComputer(
            block_size=cls.block_size_2,
            rate=cls.rate,
            n_fft=cls.n_fft_2,
            window=np.ones,
            features="all",
        )
        cls.feats_2 = cls.af_2.compute(np.expand_dims(cls.x_2[:-1], 0))

        cls.blocks = np.expand_dims(cls.x_2[:-1], 0)

    def test_power(self):
        # Pascal scale (linear)
        self.assertEqual(self.af["power"], 1.0 / 2)

    def test_rms(self):
        # Pascal scale (linear)
        self.assertAlmostEqual(self.af["rms"][0], 1.0 / np.sqrt(2))

        with self.assertRaises(ValueError):
            temporalfeatures.rms(self.blocks)

    def test_crest(self):
        # Crest is the sinus maximum
        self.assertAlmostEqual(self.af["crest"][0], 1)

    def test_peak_to_peak(self):
        self.assertAlmostEqual(self.af["peak_to_peak"][0], 2)

    def test_crest_factor(self):
        # Crest factor of a sinus
        self.assertAlmostEqual(self.af["crest_factor"][0], np.sqrt(2))

    def test_temporal_kurtosis(self):
        # Kurtosis of a sinus period
        self.assertAlmostEqual(self.af["temporal_kurtosis"][0], 1.5)

    def test_zero_crossing_rate(self):
        self.assertLessEqual(self.af["zero_crossing_rate"][0], 1e-2)
