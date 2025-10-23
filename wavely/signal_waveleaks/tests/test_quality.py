import unittest

import numpy as np

from wavely.signal.features.acousticfeatures import REF_PRESSURE
from wavely.signal.features.features import FeaturesComputer
from wavely.signal.quality import is_clipped, is_overloaded, is_underloaded
from wavely.signal.units.converters import db2lin
from wavely.signal.units.helpers import split_signal


class TestQualityCheck(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.duration = 5  # seconds
        cls.rate = int(96e3)  # kHz
        cls.signal_size = int(cls.duration * cls.rate)
        cls.t = np.arange(cls.signal_size) / cls.rate
        cls.f1 = 1e3  # Hz
        cls.signal = np.sin(2 * np.pi * cls.f1 * cls.t)
        cls.block_size = int(0.5 * cls.rate)
        # impulse
        cls.impulse = np.zeros(cls.signal_size)
        cls.impulse[10] = 1.2
        cls.clipped_signal = cls.signal + cls.impulse
        cls.clipped_blocks = split_signal(
            cls.clipped_signal, cls.rate, cls.block_size / cls.rate
        )
        # underloaded spl
        cls.underloaded_level = 10  # dB
        cls.underloaded_signal = (
            db2lin(cls.underloaded_level / 2.0) * REF_PRESSURE / 0.7
        ) * cls.signal
        cls.underloaded_blocks = split_signal(
            cls.underloaded_signal, cls.rate, cls.block_size / cls.rate
        )
        # overload spl
        cls.overloaded_level = 140  # dB
        cls.overloaded_signal = (
            db2lin(cls.overloaded_level / 2.0) * REF_PRESSURE / 0.7
        ) * cls.signal
        cls.overloaded_blocks = split_signal(
            cls.overloaded_signal, cls.rate, cls.block_size / cls.rate
        )
        # compute features
        cls.fc = FeaturesComputer(
            block_size=cls.block_size, rate=cls.rate, features=["spl"]
        )
        cls.feats_overloaded = cls.fc.compute(cls.overloaded_blocks)
        cls.feats_underloaded = cls.fc.compute(cls.underloaded_blocks)

    def test_is_clipped(self):
        self.clipped = is_clipped(self.clipped_blocks)
        self.assertEqual(self.clipped, True)

    def test_is_underloaded(self):
        # Unknown microphone
        with self.assertRaises(ValueError):
            is_underloaded(self.feats_underloaded["spl"], "foo")
        # None SNR
        with self.assertRaises(ValueError):
            is_underloaded(self.feats_underloaded["spl"], "BEHRINGER_ECM8000")
        # test underloaded signal
        self.assertEqual(
            is_underloaded(self.feats_underloaded["spl"], "VESPER_VM1000"), True
        )

    def test_is_overloaded(self):
        # Unknown microphone
        with self.assertRaises(ValueError):
            is_overloaded(self.feats_overloaded["spl"], "foo")
        # None AOP
        with self.assertRaises(ValueError):
            is_overloaded(self.feats_overloaded["spl"], "BEHRINGER_ECM8000")
        # test underloaded signal
        self.assertEqual(
            is_overloaded(self.feats_overloaded["spl"], "VESPER_VM1000"), True
        )
