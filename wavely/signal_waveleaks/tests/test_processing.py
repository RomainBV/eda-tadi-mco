import doctest
from unittest import TestCase, TextTestRunner

import numpy as np

from wavely.signal.processing import processing
from wavely.signal.processing.processing import (
    make_fade_in,
    make_fade_in_out,
    make_fade_out,
)


class DoctestTextTestRunner(TextTestRunner):
    def test_doctest(self):
        suite_processing = doctest.DocTestSuite(processing)
        self.run(suite_processing)


class FadeInOutTestCase(TestCase):
    @classmethod
    def setUpClass(cls):
        cls.n = 10
        cls.rate = 20.0
        cls.signal = np.ones(cls.n)

        cls.fade_in_duration_invalid = 0.6
        cls.fade_out_duration_invalid = 0.6
        cls.rate_invalid = np.array([cls.rate])

    def test_fade_in_out(self):
        fade_in_out = make_fade_in_out(self.rate)
        faded_in_out_signal = fade_in_out(self.signal)
        self.assertAlmostEqual(faded_in_out_signal.sum(), 6, places=3)

        with self.assertRaises(ValueError):
            fade_in_out_invalid_fade_in_duration = make_fade_in_out(
                self.rate, fade_in_duration=self.fade_in_duration_invalid
            )
            fade_in_out_invalid_fade_in_duration(self.signal)

        with self.assertRaises(ValueError):
            fade_in_out_invalid_fade_out_duration = make_fade_in_out(
                self.rate, fade_out_duration=self.fade_out_duration_invalid
            )
            fade_in_out_invalid_fade_out_duration(self.signal)

    def test_fade_in(self):
        with self.assertRaises(ValueError):
            fade_in_invalid_rate = make_fade_in(self.rate_invalid)
            fade_in_invalid_rate(self.signal)

    def test_fade_out(self):
        with self.assertRaises(ValueError):
            fade_out_invalid_rate = make_fade_out(self.rate_invalid)
            fade_out_invalid_rate(self.signal)
