import unittest

import numpy as np
from numpy.testing import assert_array_almost_equal

from wavely.signal.features.acousticfeatures import (
    REF_INTENSITY,
    REF_POWER,
    REF_PRESSURE,
    REF_PRESSURE_SQUARED,
    REF_SPECIFIC_ACOUSTIC_IMPEDANCE,
    enbw,
    intensitylevel,
    leq,
    ln,
    sil,
    soundpressurelevel,
    spl,
    spl2,
    swl,
)
from wavely.signal.features.features import FeaturesComputer
from wavely.signal.units.converters import db2lin, lin2db


class TestAcousticFeatures(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.rate = 192000
        cls.REF_LEVEL = 74.0  # dB SPL
        cls.amplitude = np.sqrt(2.0) * np.power(10, cls.REF_LEVEL / 20.0) * REF_PRESSURE

        # Calibration tone
        cls.f = 1e3  # Hz
        cls.block_size = int((cls.rate / cls.f) * 10)  # = 10 periods.
        cls.n_fft = cls.block_size
        cls.x = cls.amplitude * np.sin(
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

        # Ultrasound tone
        cls.f = 30000  # Hz
        cls.block_size = int((cls.rate / cls.f) * 10)  # = 10 periods.
        cls.n_fft = cls.block_size
        cls.u = np.sqrt(2.0) * np.sin(
            2.0
            * np.pi
            * cls.f
            * np.arange(0.0, cls.block_size / cls.rate, 1 / cls.rate)
        )
        cls.uf = FeaturesComputer(
            block_size=cls.block_size,
            rate=cls.rate,
            n_fft=cls.n_fft,
            window=np.ones,
            features="all",
        )
        cls.ufeats = cls.uf.compute(np.expand_dims(cls.u, 0))

        # gaussian random signal with no windowing
        cls.window = np.hanning
        cls.block_size = 8192
        cls.n_fft = cls.block_size
        cls.band_freq = np.array([1e3, 6e3, 11e3, cls.rate / 2.0])
        cls.rx = np.random.randn(cls.block_size)  # mean 0 and variance 1
        cls.raf = FeaturesComputer(
            block_size=cls.block_size,
            rate=cls.rate,
            n_fft=cls.n_fft,
            window=cls.window,
            band_freq=cls.band_freq,
            features="all",
        )
        cls.rfeats = cls.raf.compute(np.expand_dims(cls.rx, 0))

        # null signal
        cls.block_size = 256
        cls.n_fft = cls.block_size
        cls.null_x = np.zeros(cls.block_size)
        cls.null_af = FeaturesComputer(
            block_size=cls.block_size,
            rate=cls.rate,
            n_fft=cls.n_fft,
            window=np.ones,
            features="all",
        )
        cls.null_feats = cls.null_af.compute(np.expand_dims(cls.null_x, 0))

        # dirac signal with no windowing
        cls.dx = np.zeros(cls.block_size)
        cls.dx[10] = 1.0
        cls.daf = FeaturesComputer(
            block_size=cls.block_size,
            rate=cls.rate,
            n_fft=cls.n_fft,
            window=np.ones,
            features="all",
        )
        cls.dfeats = cls.daf.compute(np.expand_dims(cls.dx, 0))

    def test_enbw(self):
        # equivalent noise bandwidth of windows
        N = 1024

        # rectangular
        self.assertAlmostEqual(enbw(np.ones(N)), 1.0)
        # hanning
        self.assertAlmostEqual(enbw(np.hanning(N)), 1.50, places=2)
        # hamming
        self.assertAlmostEqual(enbw(np.hamming(N)), 1.36, places=2)
        # blackman
        self.assertAlmostEqual(enbw(np.blackman(N)), 1.73, places=2)

    def test_spl(self):
        # The Sound Pressure Level of the calibration tone (1 Pa RMS) is 94 dB.
        self.assertAlmostEqual(spl(REF_PRESSURE), 0.0)
        self.assertEqual(self.af["spl"][0], self.REF_LEVEL)

    def test_spl2(self):
        # The Sound Pressure Level of the calibration tone (1 Pa RMS) is 94 dB.
        self.assertAlmostEqual(spl2(REF_PRESSURE_SQUARED), 0.0)
        self.assertEqual(spl2(self.af["rms"][0] ** 2), self.REF_LEVEL)

    def test_swl(self):
        self.assertAlmostEqual(swl(REF_POWER), 0.0)

    def test_sil(self):
        # The Sound Intensity Level equals the SPL for plane waves.
        self.assertAlmostEqual(sil(REF_INTENSITY), 0.0)
        self.assertAlmostEqual(
            sil(self.af["rms"][0] ** 2 / REF_SPECIFIC_ACOUSTIC_IMPEDANCE),
            self.af["spl"][0],
            0,
        )

    def test_intensitylevel(self):
        self.assertEqual(
            intensitylevel(np.array([1])),
            sil(np.array([1]) / REF_SPECIFIC_ACOUSTIC_IMPEDANCE),
        )

    def test_soundpressurelevel(self):
        self.assertEqual(soundpressurelevel(np.array([1])), spl(np.array([1])))

    def test_ultrasoundlevel(self):
        self.assertEqual(self.uf["ultrasoundlevel"][0], self.uf["spl"][0])
        self.assertAlmostEqual(
            self.af["ultrasoundlevel"][0], self.null_af["ultrasoundlevel"][0], 3
        )

    def test_audiblelevel(self):
        self.assertEqual(self.af["audiblelevel"][0], self.af["spl"][0])
        self.assertAlmostEqual(
            self.uf["audiblelevel"][0], self.null_af["audiblelevel"][0], 3
        )

    def test_band_periodogram(self):
        self.assertEqual(len(self.raf["band_periodogram"]), len(self.band_freq))

    def test_leq(self):
        assert_array_almost_equal(
            leq(np.asarray([0.0, 0.0, 0.0, 0.0])), np.asarray([0.0])
        )
        assert_array_almost_equal(
            leq(np.array([0.0, 10.0])), lin2db(np.asarray(11.0 / 2.0))
        )
        levels = np.array([12.1, 0.3, 32.32, -2.2, 12.1, 32.3])
        self.assertEqual(
            leq(levels, int_window=3).tolist(), [leq(levels[0:3]), leq(levels[3:])]
        )

    def test_ln(self):
        levels = [0.32, 1.21, -9.32, 34.23]
        self.assertEqual(ln(np.array([levels]), 100), 34.23)
        self.assertEqual(ln(np.array([levels]), 0), -9.32)

    def test_bandleq(self):
        # Returns the correct number of `bandleq` based on the input parameter
        # `band_freq`
        self.assertEqual(len(self.raf["bandleq"][0]), len(self.raf.params["band_freq"]))
        # the sum of the energy in all bands should be equal to the total energy
        assert_array_almost_equal(
            spl2(self.raf["periodogram"][0].sum()),
            lin2db(db2lin(self.raf["bandleq"][0]).sum()),
            decimal=2,
        )

    def test_bandflatness(self):
        # Returns the correct number of `bandleq` based on the input parameter
        # `band_freq`
        self.assertEqual(
            len(self.raf["bandflatness"][0]), len(self.raf.params["band_freq"])
        )
        # A null signal has a flat PSD.
        self.assertAlmostEqual(self.null_feats["bandflatness"].mean(), 1)
        # Dirac signal PSD should be almost flat : bandflatness equals the number of
        # bands.
        self.assertAlmostEqual(
            self.daf["bandflatness"].sum(), len(self.daf.params["band_freq"]), places=-1
        )

    def test_bandcrest(self):
        # Returns the correct number of `bandleq` based on the input parameter
        # `band_freq`
        self.assertEqual(
            len(self.raf["bandcrest"][0]), len(self.raf.params["band_freq"])
        )
        # A null signal has a crest value of 0 in every band
        self.assertAlmostEqual(self.null_feats["bandcrest"].sum(), 0)
        # Dirac signal PSD should be almost flat : bands max values equals bands mean
        # values.
        self.assertAlmostEqual(
            self.daf["bandcrest"].sum(), len(self.daf.params["band_freq"]), places=-1
        )
