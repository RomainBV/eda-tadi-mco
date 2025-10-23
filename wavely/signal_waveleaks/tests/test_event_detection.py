import unittest

import numpy as np
from scipy import signal as scipy_signal
from tqdm import tqdm

from wavely.signal.processing.event_detection.event_detection import (
    Event,
    TemporalEventDetector,
    ThresholdEventDetector,
)

from tests import utils


class TestEventDetection(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.input_settings = {"microphone": "Digital", "preamp_gain": 0}

        cls.expected_events = [
            Event(16.6775, 18.75),
            Event(25.6, 35.55),
            Event(54.825, 58.825),
            Event(66.625, 72.625),
            Event(85.725, 93.7),
            Event(96.875, 97.875),
            Event(104.825, 114.825),
            Event(122.275, 128.225),
            Event(134.425, 137.4),
            Event(155.725, 165.7),
            Event(177.125, 185.125),
            Event(197.125, 203.125),
            Event(214.3, 218.3),
            Event(226.5, 228.475),
            Event(263.475, 279.275),
        ]

        total_length = 316.67
        rate = 48000
        data = 1e-3 * (2 * np.random.rand(int(total_length * rate)) - 1)
        for event in cls.expected_events:
            start_idx = int(event.start_time * rate)
            end_idx = int(event.end_time * rate)
            data[start_idx:end_idx] *= 8e2
        cls.block_duration = 0.05
        cls.features = utils.extract_features(
            data,
            rate,
            frame_length=cls.block_duration,
            features=["rms"],
            **cls.input_settings
        )

        cls.expected_threshold_events = [
            Event(0.625, 0.75),
            Event(1.425, 1.8),
            Event(3.025, 3.65),
        ]

        cls.expected_fusion_small_events = [
            Event(16.65, 35.525),
            Event(54.80, 72.60),
            Event(85.70, 97.875),
            Event(104.80, 137.375),
            Event(155.70, 165.675),
            Event(214.275, 228.45),
            Event(263.45, 279.25),
        ]

        # Data for Threshold Event Detector
        rate_threshold = 16000
        f1 = 200
        f2 = 7000
        f3 = 4500
        t_1s = np.arange(0, 0.8 * rate_threshold - 1) / rate_threshold

        # Multiples chirp signals used to construct the signal for spectral
        # centroid variation. The ThresholdEventDetector test is done with
        # spectral centroid feature.
        signal_chirp1 = scipy_signal.chirp(t_1s, 0, 1, f2, method="linear") * 0.5
        signal_chirp2 = scipy_signal.chirp(t_1s, f1, 1, f2, method="linear") * 0.5
        signal_chirp3 = scipy_signal.chirp(t_1s, f3, 1, f1, method="linear") * 0.5
        signal_chirp4 = scipy_signal.chirp(t_1s, f1, 1, f2, method="linear") * 0.5
        signal_chirp5 = scipy_signal.chirp(t_1s, f2, 1, 0, method="linear") * 0.5

        data_threshold = np.concatenate(
            [signal_chirp1, signal_chirp2, signal_chirp3, signal_chirp4, signal_chirp5]
        )

        cls.features_threshold = utils.extract_features(
            data_threshold,
            rate_threshold,
            frame_length=cls.block_duration,
            features=["spectralcentroid"],
            **cls.input_settings
        )

    @classmethod
    def _number_of_events(cls, min_length: float):
        """Compute the number of events with duration greater or equal to min_length."""
        return sum(
            1
            for event in cls.expected_events
            if event.end_time - event.start_time >= min_length
        )

    @classmethod
    def _online_process(cls, detector):
        events = []
        for index, feature in cls.features.iterrows():
            feature = dict(feature)
            feature["time"] = index
            detector.push(feature)
            event = detector.pop_event()
            if event is not None:
                events.append(event)
        last_events = detector.flush()
        valid_events = events + last_events

        return valid_events

    def test_TemporalEventDetector(self):
        # Min length test argument
        min_length = 1
        detector = TemporalEventDetector(
            block_duration=self.block_duration, min_length=min_length
        )
        events = self._online_process(detector)
        self.assertEqual(len(events), self._number_of_events(min_length))

        # Min length test argument
        min_length = 0.15
        detector = TemporalEventDetector(
            block_duration=self.block_duration, min_length=min_length
        )
        events = self._online_process(detector)
        self.assertEqual(len(events), self._number_of_events(min_length))

        for event, expected_event in zip(events, self.expected_events):
            self.assertAlmostEqual(
                (event.start_time).total_seconds(),
                (expected_event.start_time),
                delta=min_length,
            )
            self.assertAlmostEqual(
                (event.end_time).total_seconds(),
                expected_event.end_time,
                delta=min_length,
            )

        # Min length test argument
        min_length = 50
        detector = TemporalEventDetector(
            block_duration=self.block_duration, min_length=min_length
        )
        events = self._online_process(detector)
        self.assertEqual(len(events), 0)

        # Max length test argument
        min_length = 0.1
        max_length = 60
        detector = TemporalEventDetector(
            block_duration=self.block_duration,
            min_length=min_length,
            max_length=max_length,
        )
        events = self._online_process(detector)
        for event, expected_event in zip(events, self.expected_events):
            if expected_event.end_time - expected_event.start_time > max_length:
                expected_event = Event(
                    expected_event.start_time, expected_event.start_time + max_length
                )
            self.assertAlmostEqual(
                (event.start_time).total_seconds(),
                expected_event.start_time,
                delta=min_length,
            )
            self.assertAlmostEqual(
                (event.end_time).total_seconds(),
                expected_event.end_time,
                delta=min_length,
            )

        # Lowpass window and min inter event time test arguments
        min_length = 9
        max_length = 60
        lowpass_window_length = 2
        min_inter_event_time = 9

        delta_error = 1
        detector = TemporalEventDetector(
            block_duration=self.block_duration,
            min_length=min_length,
            max_length=max_length,
            lowpass_window_length=lowpass_window_length,
            min_inter_event_time=min_inter_event_time,
        )
        events = self._online_process(detector)
        for event, expected_event in zip(events, self.expected_fusion_small_events):
            if expected_event.end_time - expected_event.start_time > max_length:
                expected_event = Event(
                    expected_event.start_time, expected_event.start_time + max_length
                )
            self.assertAlmostEqual(
                (event.start_time).total_seconds(),
                expected_event.start_time,
                delta=delta_error,
            )
            self.assertAlmostEqual(
                (event.end_time).total_seconds(),
                expected_event.end_time,
                delta=delta_error,
            )

    @classmethod
    def _number_of_threshold_events(cls, min_length: float):
        """Compute the number of events with duration greater or equal to min_length."""
        return sum(
            1
            for event in cls.expected_threshold_events
            if event.end_time - event.start_time >= min_length
        )

    @classmethod
    def _online_process_threshold(cls, detector):
        events = []
        for index, feature in tqdm(cls.features_threshold.iterrows()):
            feature = dict(feature)
            feature["time"] = index
            detector.push(feature)
            event = detector.pop_event()
            if event is not None:
                events.append(event)
        last_events = detector.flush()
        events = events + last_events
        valid_events = [
            (event[0].total_seconds(), event[1].total_seconds()) for event in events
        ]
        return valid_events

    def test_ThresholdEventDetector(self):
        min_length = 0.5
        max_length = 2
        threshold = 4500
        hysteresis = 0
        min_inter_event_time = 0.0
        feature = "spectralcentroid"

        detector = ThresholdEventDetector(
            block_duration=self.block_duration,
            min_length=min_length,
            min_inter_event_time=min_inter_event_time,
            max_length=max_length,
            detection_above_threshold=True,
            feature=feature,
            hysteresis=hysteresis,
            min_level=threshold,
        )

        events = self._online_process_threshold(detector)
        self.assertEqual(len(events), self._number_of_threshold_events(min_length))

        min_length = 0.1
        max_length = 2
        threshold = 4500
        hysteresis = 1000
        min_inter_event_time = 0.0
        feature = "spectralcentroid"

        detector = ThresholdEventDetector(
            block_duration=self.block_duration,
            min_length=min_length,
            min_inter_event_time=min_inter_event_time,
            max_length=max_length,
            detection_above_threshold=True,
            feature=feature,
            hysteresis=hysteresis,
            min_level=threshold,
        )

        events = self._online_process_threshold(detector)
        self.assertEqual(len(events), self._number_of_threshold_events(min_length))

        min_length = 2
        max_length = 3
        threshold = 4500
        hysteresis = 1000
        min_inter_event_time = 0.0
        feature = "spectralcentroid"

        detector = ThresholdEventDetector(
            block_duration=self.block_duration,
            min_length=min_length,
            min_inter_event_time=min_inter_event_time,
            max_length=max_length,
            detection_above_threshold=True,
            feature=feature,
            hysteresis=hysteresis,
            min_level=threshold,
        )

        events = self._online_process_threshold(detector)
        self.assertEqual(len(events), 0)


if __name__ == "__main__":
    unittest.main()
