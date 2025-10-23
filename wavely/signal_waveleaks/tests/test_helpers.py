import unittest

import numpy as np

from wavely.signal.preprocessing.preprocessing import overlap_add_window
from wavely.signal.units.helpers import (
    SignalSplitter,
    add_and_merge_blocks,
    compute_hysteresis,
    split_signal,
    unsplit_signal,
)


class TestHelpers(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.rate = 1024
        n = 1024
        cls.block_duration_1 = 0.3
        cls.block_duration_2 = 0.5
        cls.overlap_ratio_1 = 0.1
        cls.overlap_ratio_2 = 0.2
        f1 = 20
        t = np.arange(0, n) / cls.rate
        # Generate a sin signal
        cls.signal = np.sin(2.0 * np.pi * f1 * t)

        cls.blocks_1_shape = (3, 307)
        cls.blocks_2_shape = (2, 512)

    def test_SignalSplitter(self):
        signal_splitter = SignalSplitter(
            rate=self.rate,
            block_duration=self.block_duration_1,
            overlap_ratio=self.overlap_ratio_1,
            memory_efficiency=False,
        )

        with self.assertRaises(ValueError):
            signal_splitter.transform(self.signal)

        blocks = signal_splitter.fit_transform(self.signal)
        self.assertEqual(blocks.shape, self.blocks_1_shape)
        for attribute in ["n_blocks_", "shape_", "block_size_", "step_"]:
            self.assertTrue(hasattr(signal_splitter, attribute))

        signal = signal_splitter.inverse_transform(blocks)
        self.assertEqual(signal.shape, self.signal.shape)

    def test_split_signal(self):
        blocks_1 = split_signal(
            signal=self.signal,
            rate=self.rate,
            block_duration=self.block_duration_1,
            overlap_ratio=self.overlap_ratio_1,
            memory_efficiency=False,
        )
        self.assertEqual(blocks_1.shape, self.blocks_1_shape)
        blocks_2 = split_signal(
            signal=self.signal,
            rate=self.rate,
            block_duration=self.block_duration_2,
            overlap_ratio=self.overlap_ratio_2,
            memory_efficiency=False,
        )
        self.assertEqual(blocks_2.shape, self.blocks_2_shape)
        with self.assertRaises(ValueError):
            split_signal(
                signal=self.signal,
                rate=self.rate,
                block_duration=self.block_duration_1,
                overlap_ratio=2,
                memory_efficiency=False,
            )
        with self.assertRaises(ValueError):
            split_signal(
                signal=self.signal,
                rate=self.rate,
                block_duration=0,
                overlap_ratio=self.overlap_ratio_1,
            )
        self.assertEqual(
            split_signal(
                signal=self.signal, rate=self.rate, block_duration=self.block_duration_1
            ).shape,
            (3, 307),
        )

        # Linear function
        fs = 1
        slope_1 = 12
        slope_2 = 13
        ramp_1 = np.arange(0, slope_1)
        block_duration = 6
        first_block = np.arange(0, 6)
        second_block = np.arange(6, 12, fs)
        blocks = split_signal(signal=ramp_1, rate=fs, block_duration=block_duration)
        self.assertTrue((blocks[0, :] == first_block).all())
        self.assertTrue((blocks[1, :] == second_block).all())

        ramp_2 = np.arange(0, slope_2)
        with self.assertWarns(Warning):
            split_signal(
                signal=ramp_2,
                rate=fs,
                block_duration=block_duration,
                memory_efficiency=False,
            )

    def test_unsplit_signal(self):
        rate = 1

        signal_odd = np.arange(9)
        for block_duration, overlap_ratio in [(3, 0.5), (5, 0.75)]:
            blocks = split_signal(
                signal=signal_odd,
                rate=rate,
                block_duration=block_duration,
                overlap_ratio=overlap_ratio,
            )
            signal_from_blocks = unsplit_signal(
                blocks=blocks, overlap_ratio=overlap_ratio
            )
            self.assertTrue(np.allclose(signal_odd, signal_from_blocks))

        signal_event = np.arange(20)
        for block_duration, overlap_ratio in [(4, 0.5), (4, 0.75), (5, 0.4)]:
            blocks = split_signal(
                signal=signal_event,
                rate=rate,
                block_duration=block_duration,
                overlap_ratio=overlap_ratio,
            )
            signal_from_blocks = unsplit_signal(
                blocks=blocks, overlap_ratio=overlap_ratio
            )
            self.assertTrue(np.allclose(signal_event, signal_from_blocks))

    def test_add_and_merge_blocks(self):
        rate = 100
        block_duration = 0.10
        overlap_ratio = 0.5
        signal_duration = 10
        signal_size = 100
        t = np.linspace(0, signal_duration, signal_size)
        block_size = int(block_duration * rate)
        block_overlap = int(overlap_ratio * block_size)
        block_step = block_size - block_overlap
        f = 50
        signal = np.sin(2 * np.pi * f * t)
        padded_signal = np.hstack([np.zeros(block_step), signal, np.zeros(block_step)])

        blocks = split_signal(
            signal=padded_signal,
            rate=rate,
            block_duration=block_duration,
            overlap_ratio=overlap_ratio,
            zero_pad=True,
        )

        window = overlap_add_window(block_size, block_step)[None, :]

        merged_signal = add_and_merge_blocks(
            blocks=blocks * window, overlap_ratio=overlap_ratio
        )
        # Remove padding
        merged_signal = merged_signal[block_step:-block_step]

        self.assertTrue(np.allclose(signal, merged_signal))

    def test_compute_hysteresis(self):
        threshold = 25
        hysteresis = 5
        signal1 = np.asarray([0, 1, 2, -40, 12, 31, 25, 38, 22, 19, 24])
        expected_result1 = [
            False,
            False,
            False,
            False,
            False,
            True,
            True,
            True,
            True,
            False,
            False,
        ]
        result = compute_hysteresis(signal1, threshold, hysteresis, upward=True)
        np.testing.assert_array_equal(result, expected_result1)

        signal2 = np.asarray([20, 22, 28, -10, 31, 38, 28, 31, 22, 28, 37])
        expected_result2 = [
            True,
            True,
            True,
            True,
            False,
            False,
            False,
            False,
            True,
            True,
            False,
        ]
        result = compute_hysteresis(signal2, threshold, hysteresis, upward=False)
        np.testing.assert_array_equal(result, expected_result2)

        signal3 = np.asarray([33, 28, 37, 29, 21, 27, 10, 32, 28])
        expected_result3 = [False, False, False, False, True, True, True, False, False]
        result = compute_hysteresis(signal3, threshold, hysteresis, upward=False)
        np.testing.assert_array_equal(result, expected_result3)

        signal4 = np.asarray([31, 29, 37, 35, 29, 34, 27, 10, 32, 45])
        threshold4 = np.asarray([32, 33, 32, 33, 32, 32, 34, 11, 31, 46])
        hysteresis = 5
        expected_result4 = [
            False,
            False,
            True,
            True,
            True,
            True,
            False,
            False,
            True,
            True,
        ]
        result = compute_hysteresis(signal4, threshold4, hysteresis, upward=True)
        np.testing.assert_array_equal(result, expected_result4)
