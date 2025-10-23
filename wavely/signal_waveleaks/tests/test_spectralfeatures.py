import unittest

import numpy as np
from scipy.stats import kurtosis, skew

from wavely.signal.features import acousticfeatures, temporalfeatures
from wavely.signal.features.features import FeaturesComputer
from wavely.signal.features.spectralfeatures import compensate_dft
from wavely.signal.preprocessing import preprocessing
from wavely.signal.units import converters, helpers

# TODO: refactor this, there is no need to go through the FeaturesComputer
# class to test the individual feature functions !
#  use for example a data driven test approach with inputs like:
#  [
#      {"signal": np.array(...), "frequency": ..., "rate": ...},
#      { ...},
#      ...
#  ]
#  Then loop over the above dictionaries to run the tests. Would be much
# simpler to understand and less verbose.


class TestSpectralFeatures(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.block_size = 256
        cls.rate = 192000
        cls.block_size_long = int(cls.rate / 10)

        # Signal equal to one everywhere and n_fft lower than N
        cls.n_fft = cls.block_size // 2
        cls.x = np.ones(cls.block_size)
        cls.oaf = FeaturesComputer(
            block_size=cls.block_size,
            rate=cls.rate,
            n_fft=cls.n_fft,
            window=np.ones,
            features="all",
        )

        cls.ofeats = cls.oaf.compute(np.expand_dims(cls.x, 0))

        # Signal equal to one everywhere
        cls.block_size_1 = 3840
        cls.n_fft = cls.block_size_1
        cls.f = 100
        cls.x = np.sin(
            2
            * np.pi
            * cls.f
            * np.arange(0.0, cls.block_size_1 / cls.rate, 1 / cls.rate)
        )
        cls.laf = FeaturesComputer(
            block_size=cls.block_size_1,
            rate=cls.rate,
            n_fft=cls.n_fft,
            window=np.ones,
            features="all",
        )
        cls.lfeats = cls.laf.compute(np.expand_dims(cls.x, 0))

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

        # same but no window
        cls.naf = FeaturesComputer(
            block_size=cls.block_size,
            rate=cls.rate,
            n_fft=cls.block_size,
            window=np.ones,
            features="all",
        )
        cls.nfeats = cls.naf.compute(np.expand_dims(cls.x, 0))

        # multiple sin signal
        cls.mf = np.array([20, 40, 63, 93, 102, 67]).reshape(-1, 1)
        cls.mx = np.sin(
            2.0
            * np.pi
            * (
                cls.mf * np.arange(0.0, 1.0, 1 / cls.block_size)
                + np.random.rand(*cls.mf.shape)
            )
        ).sum(axis=0)
        cls.maf = FeaturesComputer(
            block_size=cls.block_size,
            rate=cls.rate,
            n_fft=cls.block_size,
            window=np.ones,
            features="all",
        )
        cls.mfeats = cls.maf.compute(np.expand_dims(cls.mx, 0))

        # gaussian random signal with no windowing
        cls.rx = 0.5 * np.random.randn(cls.block_size)
        cls.raf = FeaturesComputer(
            block_size=cls.block_size,
            rate=cls.rate,
            n_fft=cls.n_fft,
            window=np.ones,
            features="all",
        )
        cls.rfeats = cls.raf.compute(np.expand_dims(cls.rx, 0))

        # simple sine signal + gaussian random signal with no windowing
        cls.srx = np.sin(
            2.0
            * np.pi
            * cls.f
            * np.arange(0.0, cls.block_size / cls.rate, 1 / cls.rate)
        ) + 0.1 * np.random.randn(cls.block_size)
        cls.sraf = FeaturesComputer(
            block_size=cls.block_size,
            rate=cls.rate,
            n_fft=cls.block_size,
            window=np.ones,
        )
        cls.srfeats = cls.sraf.compute(np.expand_dims(cls.srx, 0))

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

        # null signal
        cls.null_x = np.zeros(cls.block_size)
        cls.null_af = FeaturesComputer(
            block_size=cls.block_size,
            rate=cls.rate,
            n_fft=cls.n_fft,
            window=np.ones,
            features="all",
        )
        cls.null_feats = cls.null_af.compute(np.expand_dims(cls.null_x, 0))

        # long gaussian random signal + sine
        sine = (1.0 / np.sqrt(2)) * np.sin(
            2.0
            * np.pi
            * cls.f
            * np.arange(0.0, cls.block_size_long / cls.rate, 1 / cls.rate)
        )
        # generate a gaussian white noise whose SPL is 40dB below the sine one.
        sine_spl = acousticfeatures.spl(
            temporalfeatures.rms(temporalfeatures.power(sine))
        )
        noise = np.random.randn(10 * cls.block_size_long)
        noise = converters.compensate_spl(noise, sine_spl - 40.0)

        noise_fc = FeaturesComputer(
            block_size=cls.block_size_long,
            rate=cls.rate,
            n_fft=cls.block_size_long,
            window=np.ones,
        )
        _ = noise_fc.compute(
            helpers.split_signal(noise, cls.rate, cls.block_size_long / cls.rate)
        )
        cls.background_noise = preprocessing.make_background_spectrum(
            noise_fc["periodogram"], estimator="mean"
        )

        cls.emergence_fc = FeaturesComputer(
            block_size=cls.block_size_long,
            rate=cls.rate,
            n_fft=cls.block_size_long,
            window=np.ones,
        )
        signal = sine + helpers.split_signal(
            noise, cls.rate, cls.block_size_long / cls.rate
        ).mean(0)
        cls.emergence_feats = cls.emergence_fc.compute(
            np.expand_dims(signal, 0), background_spectrum=cls.background_noise
        )

        cls.sine_fc = FeaturesComputer(
            block_size=cls.block_size_long,
            rate=cls.rate,
            n_fft=cls.block_size_long,
            window=np.ones,
        )
        cls.sine_feats = cls.sine_fc.compute(np.expand_dims(sine, 0))

    def test_compensate_dft(self):
        dft = self.oaf["dft"]
        copy_norm_dft = compensate_dft(
            dft, self.n_fft, np.ones(self.block_size), inplace=False
        )
        self.assertTrue(id(dft) != id(copy_norm_dft))

        inplace_norm_dft = compensate_dft(
            dft, self.n_fft, np.ones(self.block_size), inplace=True
        )
        self.assertTrue(np.allclose(copy_norm_dft, dft))
        self.assertTrue(inplace_norm_dft is None)

    def test_periodogram(self):
        self.assertAlmostEqual(self.af["power"][0], 1.0 / 2)
        self.assertAlmostEqual(self.af["rms"][0], 1.0 / np.sqrt(2))

        # Frequency analysis
        periodogram = self.af["periodogram"]
        rperiodogram = self.raf["periodogram"]

        # total power is 1/2 (equivalent to `power`)
        self.assertAlmostEqual(periodogram.sum(), 1.0 / 2.0)

        # peak freq should be at bin f
        self.assertEqual(
            periodogram.argmax(), np.round(self.f * self.n_fft / self.rate)
        )

        # spectral power should be the same as time-based power (Parseval's identity)
        self.assertAlmostEqual(rperiodogram.sum(), self.raf["power"][0])
        self.assertEqual(self.naf["periodogram"].sum(), self.naf["power"])

    def test_spectrum(self):
        spectrum = self.oaf["spectrum"]
        self.assertEqual(spectrum[0, 0], 2 / self.block_size)

    def test_melspectrogram(self):
        melspectrogram = self.laf["melspectrogram"]
        self.assertEqual(melspectrogram.shape[1], 128)

    def test_spectralcentroid(self):

        # spectral centroid should be the mean of all pure sin frequencies
        self.assertAlmostEqual(
            self.maf["spectralcentroid"][0], self.mf.mean() / self.block_size * 192000
        )
        self.assertEqual(self.null_feats["spectralcentroid"], 0.0)

    def test_spectralspread(self):

        # spectral spread should be the variance of all pure sin frequencies
        self.assertAlmostEqual(
            self.maf["spectralspread"][0],
            self.mf.std() * (192000 / self.block_size),
            places=5,
        )
        self.assertEqual(self.null_feats["spectralspread"][0], 0.0)

    def test_spectralskewness(self):

        self.assertAlmostEqual(
            self.maf["spectralskewness"][0],
            skew(self.mf * (self.rate / self.block_size))[0],
        )

        self.assertEqual(0, self.null_feats["spectralskewness"])

    def test_spectralkurtosis(self):

        self.assertAlmostEqual(
            self.maf["spectralkurtosis"][0],
            kurtosis(self.mf * (self.rate / self.block_size), fisher=False)[0],
        )

        self.assertEqual(0, self.null_feats["spectralkurtosis"])

    def test_spectralflatness(self):

        # dirac signal DSP should be flat.
        self.assertAlmostEqual(self.daf["spectralflatness"][0], 1.0, places=2)
        self.assertAlmostEqual(self.null_feats["spectralflatness"][0], 1)

    def test_crestfactor(self):

        # dirac signal DSP should have no crest, that is, a max value equal to its
        # mean value.
        self.assertAlmostEqual(self.daf["spectralcrest"][0], 1.0, places=2)
        self.assertAlmostEqual(self.null_feats["spectralcrest"][0], 0.0)

    def test_spectralentroy(self):

        # one peak spectrum should have entropy 0
        self.assertAlmostEqual(self.naf["spectralentropy"][0], 0.0)
        self.assertEqual(self.null_feats["spectralentropy"], 0.0)

    def test_spectralflux(self):

        # Spectral flux of a sine signal at time 0 should be equal to the sum of its
        # (constant over time) DSP values, that is, 1/2 for the dirac at the sine freq.
        self.assertAlmostEqual(self.naf["spectralflux"][0], 1 / 2.0)
        self.assertEqual(self.null_feats["spectralflux"], 0.0)

    def test_peakfreq(self):

        self.assertEqual(
            self.af["peakfreq"][0],
            round(self.f / self.rate * self.n_fft) * self.rate / self.n_fft,
        )
        self.assertEqual(self.null_feats["peakfreq"], 0.0)

    def test_spectralirregularity(self):
        # The sine should produce a dirac dent of 1/2 on top of the flat gaussian noise
        # DSP, that is, an irregularity of 2 * 1/2 = 1.
        self.assertAlmostEqual(self.sraf["spectralirregularity"][0], 1.0, places=1)
        # Null signal
        self.assertEqual(self.null_feats["spectralirregularity"], 0.0)

    def test_highfrequencycontent(self):

        self.assertAlmostEqual(
            np.sqrt(self.af["highfrequencycontent"][0]), self.f, places=-2
        )
        self.assertEqual(self.null_feats["highfrequencycontent"], 0.0)

    def test_spectralrolloff(self):
        # 95% of spectral power should be contained within the first 95% frequency bins
        # of a signal with flat PSD.
        rolloff_bin = 0.95 * self.daf.n_fft // 2
        rolloff_freq = (self.daf.rate / self.daf.n_fft) * rolloff_bin
        self.assertEqual(self.daf["spectralrolloff"], rolloff_freq)

    def test_amplitude_envelope(self):
        # Rounding is necessary. We take the first element because occasionally
        # there is also a zero.
        self.assertEqual(np.unique(np.round(self.af["amplitude_envelope"][0]), 6)[0], 1)

    def test_instantaneous_frequency(self):
        # Rounding is necessary. We take the first element because occasionally
        # there is also a zero.
        self.assertEqual(
            np.unique(np.round(self.af["instantaneous_frequency"][0]), 0), 750
        )

    def test_spectral_emergence(self):
        self.assertEqual(
            self.emergence_fc["periodogram"].argmax(),
            self.sine_fc["periodogram"].argmax(),
        )
