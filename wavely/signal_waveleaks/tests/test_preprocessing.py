import doctest
from typing import Callable
from unittest import TestCase, TextTestRunner

import numpy as np

from wavely.signal.features.features import FeaturesComputer
from wavely.signal.preprocessing import preprocessing
from wavely.signal.preprocessing.preprocessing import (
    OverlapAndAddFilter,
    _is_outlier,
    apply_gain,
    band_indexes,
    calibrate_audio_signal,
    compensate_mems_resonance,
    fft_frequencies,
    frequency_filters,
    hz_to_mel,
    make_a_filtering,
    make_butter_filter,
    make_butter_highpass_filter,
    make_c_filtering,
    make_dc_offset,
    make_declipper,
    make_linear_filter,
    make_overlap_and_add_filter,
    max_abs_scaler,
    mel,
    mel_frequencies,
    mel_to_hz,
    min_max_scaler,
    octave_bands_frequencies,
    simulate_aging_mic_response,
    simulate_resonance_peak,
)
from wavely.signal.units import converters, helpers
from wavely.signal.units.helpers import SignalSplitter

from tests.utils import relative_change


class DoctestTextTestRunner(TextTestRunner):
    def test_doctest(self):
        suite = doctest.DocTestSuite(preprocessing)
        self.run(suite)


class AudioSignalCalibrationTestCase(TestCase):
    def test_empty_input(self):
        shape = (0,)
        result = calibrate_audio_signal(np.ndarray(shape), microphone="Builtin")
        np.testing.assert_array_equal(result, np.ndarray(shape))

    def test_unknown_mic(self):
        y = np.ndarray((10,))
        with self.assertRaises(ValueError):
            calibrate_audio_signal(y, microphone="foo")
        with self.assertRaises(ValueError):
            calibrate_audio_signal(y, microphone=None)

    def test_default_parameters(self):
        # preamp_gain and rss_calibration default to 0, so using the default mic,
        # the signal does not change

        shape = (10,)
        y = np.ndarray(shape)
        result = calibrate_audio_signal(y, microphone="Builtin")
        np.testing.assert_array_equal(result, y)

    def test_2d_input(self):
        shape = (10, 2)
        y = np.ndarray(shape)
        result = calibrate_audio_signal(y, microphone="Builtin")
        np.testing.assert_array_equal(result, y)


# The two following tests are used to remove an offset
# But the first one uses make_butter_highpass_filter
# And the second uses make_dc_offset
class DcOffsetTestCase(TestCase):
    def test_dc_offset(self):
        cutoff = 15
        rate = 1024
        f = 200
        x = np.sin(2 * np.pi * f * np.arange(rate) / rate)
        x_with_offset = x + 0.5
        dc_offset = make_dc_offset(rate, cutoff)
        change = relative_change(reference=x, observed=dc_offset(x_with_offset))
        # Currently the best performance
        self.assertLessEqual(change.mean(), 0.136)
        self.assertLessEqual(change.max(), 0.431)

    def test_dc_offset_no_cutoff(self):
        rate = 1024
        f = 200
        x = np.sin(2 * np.pi * f * np.arange(rate) / rate)
        x_with_offset = x + 0.5
        dc_offset = make_dc_offset(rate)
        change = relative_change(reference=x, observed=dc_offset(x_with_offset))
        # Currently the best performance
        self.assertLessEqual(change.mean(), 0.136)
        self.assertLessEqual(change.max(), 0.431)


class LinearFilterTestCase(TestCase):
    def test_iir_fir_filter(self):
        rate = 1024
        f = 200
        x = np.sin(2 * np.pi * f * np.arange(rate) / rate)
        x_with_offset = x + 0.5

        b_coeffs = [0.30185684375464916, 0.26025809766724706, 0.21865935577079654]
        a_coeffs = [1, -0.2602580976672470, 0.479483800008893]

        linear_filter = make_linear_filter(b_coeffs=b_coeffs, a_coeffs=a_coeffs)

        filtered_signal = linear_filter(x_with_offset)
        self.assertIsNotNone(filtered_signal)

        fir_filter = make_linear_filter(np.array([0.1, 0.5, 0.9]), np.array([1]))
        filtered_signal = fir_filter(x_with_offset)
        self.assertIsNotNone(filtered_signal)

    def test_filter_warning(self):
        with self.assertWarns(Warning):
            _ = make_linear_filter([0.1, 0.5, 0.9, 0.5], [0.5, 0.5, 0.5])

    def test_linear_filter_error(self):
        with self.assertRaises(ValueError):
            _ = make_linear_filter([2, 5, 0.9, 0.5], [0, 0.5, 0.5, 4])


class AudioSignalNormalisationTestCase(TestCase):
    def test_apply_gain_empty_input(self):
        shape = (0,)
        result = apply_gain(np.ndarray(shape))
        np.testing.assert_array_equal(result, np.ndarray(shape))

    def test_apply_gain_1d_input(self):
        # preamp_gain and rss_calibration default to 0, so using the default mic,
        # the signal does not change

        shape = (10,)
        y = np.ndarray(shape)
        result = apply_gain(y)
        self.assertIsInstance(result, np.ndarray)
        self.assertEqual(result.shape, y.shape)

    def test_apply_gain_2d_input(self):
        shape = (10, 2)
        y = np.ndarray(shape)
        result = apply_gain(y)
        self.assertIsInstance(result, np.ndarray)
        self.assertEqual(result.shape, y.shape)

    def test_min_max_scaler(self):
        np.random.seed(123)
        x = np.random.randn(5, 10)
        y = min_max_scaler(x)
        self.assertEqual(y.min(), 0.0)
        self.assertEqual(y.max(), 1.0)

    def test_max_abs_scaler(self):
        np.random.seed(123)
        rate = 16000.0
        x = np.random.randn(5, 10)
        y = max_abs_scaler(x, rate)
        self.assertTrue(y.min() == -1 or y.max() == 1)


class AFilteringTestCase(TestCase):
    def test_a_filtering_attenuation(self):
        rate = 96000
        duration = 10
        t = np.linspace(0, 10, duration * rate)
        a_filter = make_a_filtering(rate)
        frequencies = [125, 250, 500, 1000]
        true_attenuations = [16.1, 8.6, 3.2, 0]
        for frequency, true_attenuation in zip(frequencies, true_attenuations):
            signal = np.sin(2 * np.pi * frequency * t)
            filtered_signal = a_filter(signal)
            attenuation = converters.lin2db(np.sum(signal**2)) - converters.lin2db(
                np.sum(filtered_signal**2)
            )
            self.assertAlmostEqual(attenuation, true_attenuation, places=0)


class CFilteringTestCase(TestCase):
    def test_c_filtering_attenuation(self):
        rate = 96000
        duration = 10
        t = np.linspace(0, 10, duration * rate)
        c_filter = make_c_filtering(rate)
        frequencies = [125, 250, 500, 10000]
        true_attenuations = [0.2, 0, 0, 4.4]
        for frequency, true_attenuation in zip(frequencies, true_attenuations):
            signal = np.sin(2 * np.pi * frequency * t)
            filtered_signal = c_filter(signal)
            attenuation = converters.lin2db(np.sum(signal**2)) - converters.lin2db(
                np.sum(filtered_signal**2)
            )
            self.assertAlmostEqual(attenuation, true_attenuation, places=0)


class HighpassFilteringTestCase(TestCase):
    def test_highpass_filtering(self):
        rate = 16e3
        cutoff = 20e3

        with self.assertRaises(ValueError):
            make_butter_highpass_filter(cutoff, rate)


class FilteringTestCase(TestCase):
    @classmethod
    def setUpClass(cls):
        cls.rate = 96e3
        cls.sample_short = int(cls.rate // 50)
        t_short = np.arange(cls.sample_short) / cls.rate
        cls.f_short = cls.rate * np.arange(cls.sample_short) / cls.sample_short
        cls.f = [100, 10000, 30000]
        cls.signal_short = (
            np.sin(2 * np.pi * cls.f[0] * t_short)
            + np.sin(2 * np.pi * cls.f[1] * t_short)
            + np.sin(2 * np.pi * cls.f[2] * t_short)
        )

        cls.filters_parameters = {
            "lowpass": {"cutoff": 5000, "target_f": cls.f[0]},
            "bandpass": {"cutoff": [5000, 15000], "target_f": cls.f[1]},
            "highpass": {"cutoff": [15000], "target_f": cls.f[2]},
        }

        cls.sample_long = int(cls.rate * 30)
        t_long = np.arange(cls.sample_long) / cls.rate
        cls.f_long = cls.rate * np.arange(cls.sample_long) / cls.sample_long
        cls.signal_long = (
            np.sin(2 * np.pi * cls.f[0] * t_long)
            + np.sin(2 * np.pi * cls.f[1] * t_long)
            + np.sin(2 * np.pi * cls.f[2] * t_long)
        )

        cls.block_duration = 0.04
        cls.blocks_long_without_overlap = helpers.split_signal(
            cls.signal_long,
            rate=cls.rate,
            block_duration=cls.block_duration,
            overlap_ratio=0,
        )
        cls.overlap_1 = 0.5
        cls.blocks_long_with_overlap_1 = helpers.split_signal(
            cls.signal_long,
            rate=cls.rate,
            block_duration=cls.block_duration,
            overlap_ratio=cls.overlap_1,
        )
        cls.overlap_2 = 0.75
        cls.blocks_long_with_overlap_2 = helpers.split_signal(
            cls.signal_long,
            rate=cls.rate,
            block_duration=cls.block_duration,
            overlap_ratio=cls.overlap_2,
        )

    def test_butter_filter(self):
        rate_invalid = 16e3
        cutoff_invalid = 20e3

        with self.assertRaises(ValueError):
            make_butter_filter(cutoff_invalid, rate_invalid)

        for filter_type, parameters in self.filters_parameters.items():
            butter_filter = make_butter_filter(
                parameters["cutoff"], self.rate, filter_type, 10
            )
            spectrum = np.fft.fft(butter_filter(self.signal_short))[
                : self.sample_short // 2
            ]
            max_spectrum = np.abs(spectrum).max()
            self.assertEqual(
                self.f_short[np.where(np.abs(spectrum) > max_spectrum // 2)[0][0]],
                parameters["target_f"],
            )

    def test_overlap_add_butter_filter(self):
        # Test for overlap add filtering
        for filter_type, parameters in self.filters_parameters.items():
            overlap_and_add_filter = make_overlap_and_add_filter(
                rate=self.rate,
                filter_callable=make_butter_filter(
                    rate=self.rate,
                    cutoff=parameters["cutoff"],
                    filter_type=filter_type,
                    order=10,
                ),
            )
            spectrum = np.fft.fft(overlap_and_add_filter(self.signal_long))[
                : self.sample_long // 2
            ]
            max_spectrum = np.abs(spectrum).max()
            self.assertEqual(
                self.f_long[np.where(np.abs(spectrum) > max_spectrum // 2)[0][0]],
                parameters["target_f"],
            )

    def _filter_signal(
        self,
        butter_filter: Callable[[np.ndarray], np.ndarray],
        blocks: np.ndarray,
        are_blocks_with_overlap: bool,
        overlap_ratio: float,
        overlap_reconstruction: float,
    ):
        """Filter a signal with the corresponding filter.

        Args:
            butter_filter: A Callable used to filter a signal.
            blocks: A shape-(n,k) array containing the blocks to filter.
            are_blocks_with_overlap: If False, apply an overlap of overlap_ratio to
                the blocks.
            overlap_ratio: The overlap ratio.
            overlap_reconstruction: The overlap used for reconstruction.

        """
        n_blocks = 100
        filtered_blocks = np.zeros_like(blocks)
        n_batches = int(blocks.shape[0] / n_blocks)
        overlap_and_add_filter = OverlapAndAddFilter(
            filter_func=butter_filter,
            block_duration=self.block_duration,
            rate=self.rate,
            are_blocks_with_overlap=are_blocks_with_overlap,
            overlap_ratio=overlap_ratio,
        )
        for batch in range(n_batches):
            overlap_and_add_filter.push(
                blocks[slice(batch * n_blocks, (batch + 1) * n_blocks), :]
            )
            filtered_blocks[
                slice(batch * n_blocks, (batch + 1) * n_blocks)
            ] = overlap_and_add_filter.pop_blocks().squeeze()
        overlap_and_add_filter.push(
            blocks[slice(n_batches * n_blocks, blocks.shape[0]), :]
        )
        filtered_blocks[
            slice(n_batches * n_blocks, blocks.shape[0])
        ] = overlap_and_add_filter.pop_blocks().squeeze()

        filtered_signal = helpers.unsplit_signal(
            blocks=filtered_blocks, overlap_ratio=overlap_reconstruction
        )
        return filtered_signal

    def test_OverlapAndAddFilter_without_overlap(self):
        for filter_type, parameters in self.filters_parameters.items():
            butter_filter = make_butter_filter(
                cutoff=parameters["cutoff"],
                rate=self.rate,
                filter_type=filter_type,
                order=10,
            )
            filtered_signal = self._filter_signal(
                butter_filter=butter_filter,
                blocks=self.blocks_long_without_overlap,
                are_blocks_with_overlap=False,
                overlap_ratio=self.overlap_1,
                overlap_reconstruction=0,
            )
            spectrum = np.fft.fft(butter_filter(filtered_signal))[
                : self.sample_long // 2
            ]
            max_spectrum = np.abs(spectrum).max()
            self.assertEqual(
                self.f_long[np.where(np.abs(spectrum) > max_spectrum // 2)[0][0]],
                parameters["target_f"],
            )

    def test_OverlapAndAddFilter_with_overlap(self):
        for filter_type, parameters in self.filters_parameters.items():
            butter_filter = make_butter_filter(
                cutoff=parameters["cutoff"],
                rate=self.rate,
                filter_type=filter_type,
                order=10,
            )
            filtered_signal_1 = self._filter_signal(
                butter_filter=butter_filter,
                blocks=self.blocks_long_with_overlap_1,
                are_blocks_with_overlap=True,
                overlap_ratio=self.overlap_1,
                overlap_reconstruction=self.overlap_1,
            )
            spectrum_1 = np.fft.fft(butter_filter(filtered_signal_1))[
                : self.sample_long // 2
            ]
            max_spectrum_1 = np.abs(spectrum_1).max()
            self.assertEqual(
                self.f_long[np.where(np.abs(spectrum_1) > max_spectrum_1 // 2)[0][0]],
                parameters["target_f"],
            )

            filtered_signal_2 = self._filter_signal(
                butter_filter=butter_filter,
                blocks=self.blocks_long_with_overlap_2,
                are_blocks_with_overlap=True,
                overlap_ratio=self.overlap_2,
                overlap_reconstruction=self.overlap_2,
            )
            spectrum_2 = np.fft.fft(butter_filter(filtered_signal_2))[
                : self.sample_long // 2
            ]
            max_spectrum_2 = np.abs(spectrum_2).max()
            self.assertEqual(
                self.f_long[np.where(np.abs(spectrum_2) > max_spectrum_2 // 2)[0][0]],
                parameters["target_f"],
            )

    @classmethod
    def tearDownClass(cls):
        pass


class FrequencyFiltersCase(TestCase):
    def test_frequency_filters(self):
        cutoff = 20e3
        frequency_filters_no_sample_rates_larger = frequency_filters(cutoff=cutoff)
        for filter_type in ["lowpass", "highpass"]:
            self.assertIsNone(
                frequency_filters_no_sample_rates_larger[filter_type][16e3]
            )

        sample_rates = [48e3, 96e3]
        frequency_filters_sample_rates = frequency_filters(
            cutoff=cutoff, sample_rates=sample_rates
        )
        for filter_type in ["lowpass", "highpass"]:
            self.assertListEqual(
                list(frequency_filters_sample_rates[filter_type].keys()), sample_rates
            )


class FrequencyBandsTestCase(TestCase):
    def test_band_indexes(self):
        rate = 44100
        window = np.hanning
        N = int(rate * 0.04)
        n_fft = N
        band_freq = "third"
        freqs = 10.0
        args = {
            "block_size": N,
            "rate": rate,
            "n_fft": n_fft,
            "window": window,
            "band_freq": band_freq,
            "features": "all",
        }
        low_samplerate = 6e3
        default_freqs = 5e3

        # This exception is raised when the frequency resolution is not enough in the
        # low frequencies.
        # In this case there are duplicated indexes which result in empty slices when
        # applying np.split on the periodogram.
        with self.assertRaises(ValueError):
            FeaturesComputer(**args)

        # This exception is raised when the frequency bandwidth is lower than the
        # frequency resolution.
        with self.assertRaises(ValueError):
            band_indexes(rate=rate, n_fft=n_fft, freqs=freqs)

        # This exception is raised when the freqs argument is not properly set
        with self.assertRaises(TypeError):
            band_indexes(rate=rate, n_fft=n_fft, freqs=None)

        # Simple test on the expected result of the function
        expected_band_freqs = [20.0, 40.0]
        expected_band_ix = [2, 4]
        band_freqs, band_ix = band_indexes(rate=100.0, n_fft=10, freqs=20.0)
        self.assertEqual(band_freqs.tolist(), expected_band_freqs)
        self.assertEqual(band_ix.tolist(), expected_band_ix)

        with self.assertRaises(ValueError):
            band_indexes(rate=low_samplerate, n_fft=n_fft, freqs=default_freqs)

        # Test for low value of sampling frequency
        low_rate = 11e3
        octave_freqs, _ = band_indexes(rate=low_rate, n_fft=n_fft, freqs="octave")
        self.assertLessEqual(octave_freqs[-1], low_rate)

    def test_octave_bands_frequencies(self):
        band_type = None
        # This exception is raised when the band_type is not properly set
        with self.assertRaises(ValueError):
            octave_bands_frequencies(band_type)


class MelCoefficientsTestCase(TestCase):
    @classmethod
    def setUpClass(cls):
        cls.hz_f = 1000
        cls.hz_f_htk = 6300
        cls.hz_f_v = [110, 220, 440]
        cls.hz_f_o = 2800 / 3
        cls.mel_f = 15
        cls.mel_f_htk = 2595
        cls.mel_f_v = [1.65, 3.3, 6.6]
        cls.mel_f_o = 14
        cls.rate = 32
        cls.n_fft = 4
        cls.n_mels = 2
        cls.norm_error = 2
        cls.fmax = cls.rate / 2
        cls.norm_none = None

    def test_hz_to_mel(self):
        mel_f = hz_to_mel(self.hz_f)
        self.assertAlmostEqual(mel_f, self.mel_f)

        mel_f_htk = hz_to_mel(self.hz_f_htk, True)
        self.assertAlmostEqual(mel_f_htk, self.mel_f_htk)

        mel_f_v = hz_to_mel(self.hz_f_v)
        self.assertTrue((mel_f_v == self.mel_f_v).all())

    def test_mel_to_hz(self):
        hz_f = mel_to_hz(self.mel_f)
        self.assertAlmostEqual(hz_f, self.hz_f)

        hz_f_htk = mel_to_hz(self.mel_f_htk, True)
        self.assertAlmostEqual(hz_f_htk, self.hz_f_htk)

        hz_f_v = mel_to_hz(self.mel_f_v)
        self.assertTrue((hz_f_v == self.hz_f_v).all())

        hz_f_o = mel_to_hz(self.mel_f_o)
        self.assertAlmostEqual(hz_f_o, self.hz_f_o)

    def test_fft_frequencies(self):
        freqs = fft_frequencies(self.rate, self.n_fft)
        self.assertTrue((freqs == [0, 8, 16]).all())

    def test_mel_frequencies(self):
        mel_freqs = mel_frequencies(self.n_mels)
        self.assertTrue((mel_freqs == [0, 11025]).all())

    def test_mel(self):
        mfcc = mel(sr=self.rate, n_fft=self.n_fft, fmax=self.fmax, n_mels=self.n_mels)
        self.assertEqual(mfcc.shape, (3, self.n_mels))

        mfcc_norm = mel(
            sr=self.rate,
            n_fft=self.n_fft,
            fmax=self.fmax,
            n_mels=self.n_mels,
            norm=self.norm_none,
        )
        self.assertAlmostEqual(mfcc_norm.sum(), 1)

        with self.assertRaises(ValueError):
            mel(sr=self.rate, n_fft=self.n_fft, norm=self.norm_error)


class DeclipperTestCase(TestCase):
    @classmethod
    def setUpClass(cls):
        cls.n_1 = 10
        cls.idx_1 = 5
        cls.signal_1 = np.random.rand(cls.n_1)
        cls.signal_1[cls.idx_1] = cls.n_1
        cls.threshold = 10

        np.random.seed(123)
        cls.n_2 = 100
        cls.signal_2 = 0.01 * np.random.randn(cls.n_2)
        cls.idx_2 = range(cls.n_2 // 2 - 5, cls.n_2 // 2 + 5)
        cls.signal_2[cls.idx_2] *= 100

        cls.signal_3 = np.ones(cls.n_1)

    def test_is_outlier(self):
        mask = _is_outlier(self.signal_1, self.threshold)
        self.assertTrue(mask[self.idx_1])

        mask = _is_outlier(self.signal_1[:, None], self.threshold)
        self.assertTrue(mask[self.idx_1])

    def test_declipper(self):
        declipper = make_declipper()
        declipped_signal = declipper(self.signal_2)
        self.assertTrue(
            (declipped_signal[self.idx_2] <= self.signal_2[self.idx_2]).all()
        )

        declipped_signal_no_std = declipper(self.signal_1)
        self.assertTrue((declipped_signal_no_std == self.signal_1).all())

        declipped_signal_no_outlier = declipper(self.signal_3)
        self.assertTrue((declipped_signal_no_outlier == self.signal_3).all())


class MicSimulationsTestCase(TestCase):
    @classmethod
    def setUpClass(cls):
        np.random.seed(123)
        signal_length = 5000
        cls.raw_signal = np.random.randn(signal_length)
        cls.sample_rate = 96000.0
        # Features parameters
        block_duration = 0.04
        overlap_ratio = 0.0

        cls.block_size = int(block_duration * cls.sample_rate)

        features = ["spectrum"]

        cls.signal_splitter = SignalSplitter(
            rate=cls.sample_rate,
            block_duration=block_duration,
            overlap_ratio=overlap_ratio,
            zero_pad=False,
        )
        cls.features_computer = FeaturesComputer(
            rate=cls.sample_rate,
            block_size=cls.block_size,
            features=features,
        )

    def get_frequency_intensity(
        self, signal: np.ndarray, frequency: np.ndarray
    ) -> float:
        """Get frequency intensity.

        Compute spectrum of a signal and return the frequency intensity.

        Args:
            signal: An audio time series.
            frequency: Frequency in Hz.

        Returns:
            The frequency intensity.

        """
        signal_blocks = self.signal_splitter.fit_transform(signal)
        signal_spectrum = self.features_computer.compute(signal_blocks)["spectrum"]
        frequency_index = int(frequency * self.block_size / self.sample_rate)
        frequency_intensity = np.abs(signal_spectrum[0, frequency_index])
        return frequency_intensity

    def test_compensate_mems_resonance(self):
        mems_filter = compensate_mems_resonance(
            rate=self.sample_rate,
        )

        filtered_signal = mems_filter(self.raw_signal)
        mems_resonance_frequency = 14000.0  # Hz

        raw_signal_frequency_intensity = self.get_frequency_intensity(
            self.raw_signal, mems_resonance_frequency
        )

        filtered_signal_frequency_intensity = self.get_frequency_intensity(
            filtered_signal, mems_resonance_frequency
        )

        has_decreased = (
            filtered_signal_frequency_intensity < raw_signal_frequency_intensity
        )
        self.assertTrue(has_decreased)

    def test_simulate_aging_mic_response(self):
        filtered_signal = simulate_aging_mic_response(
            signal=self.raw_signal,
            sample_rate=self.sample_rate,
        )
        low_frequency_to_check = 250.0  # Hz
        raw_low_frequency_intensity = self.get_frequency_intensity(
            self.raw_signal, low_frequency_to_check
        )
        filtered_low_frequency_intensity = self.get_frequency_intensity(
            filtered_signal, low_frequency_to_check
        )
        low_frequency_has_decreased = (
            filtered_low_frequency_intensity < raw_low_frequency_intensity
        )

        high_frequency_to_check = 7000.0  # Hz
        raw_high_frequency_intensity = self.get_frequency_intensity(
            self.raw_signal, high_frequency_to_check
        )
        filtered_high_frequency_intensity = self.get_frequency_intensity(
            filtered_signal, high_frequency_to_check
        )
        high_frequency_has_decreased = (
            filtered_high_frequency_intensity < raw_high_frequency_intensity
        )

        self.assertTrue(low_frequency_has_decreased)
        self.assertTrue(high_frequency_has_decreased)

    def test_simulate_resonance_peak(self):
        frequency_resonance_peak = 33000.0
        filtered_signal = simulate_resonance_peak(
            signal=self.raw_signal,
            sample_rate=self.sample_rate,
            peak_frequency=frequency_resonance_peak,
            lp_cutoff_frequency=30000.0,
            hp_cutoff_frequency=38000.0,
            Q=20,
            G=10,
        )

        filtered_resonance_frequency_intensity = self.get_frequency_intensity(
            filtered_signal, frequency_resonance_peak
        )

        raw_resonance_frequency_intensity = self.get_frequency_intensity(
            self.raw_signal, frequency_resonance_peak
        )

        has_increased = (
            filtered_resonance_frequency_intensity > raw_resonance_frequency_intensity
        )

        self.assertTrue(has_increased)
