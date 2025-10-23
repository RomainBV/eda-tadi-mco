import datetime as dt
import warnings
from typing import Any, List, NamedTuple, Optional, Sequence

import numpy as np
import pandas as pd

from wavely.signal.features.acousticfeatures import spl
from wavely.signal.units.helpers import compute_hysteresis


class Event(NamedTuple):
    start_time: pd.Timestamp
    end_time: pd.Timestamp


class RingBuffer:
    """Class RingBuffer that implements a circular buffer."""

    def __init__(self, length: int):
        self._buffer = list()
        self._length = length

    def append(self, x: Any):
        if len(self._buffer) == self._length:
            self._buffer.pop(0)
        self._buffer.append(x)

        return self

    def __getitem__(self, key):
        return self._buffer[key]

    def __repr__(self):
        return self._buffer.__repr__()

    def __len__(self):
        return self._buffer.__len__()


class ThresholdEventDetector:
    """Implement threshold event detection."""

    def __init__(
        self,
        block_duration: float,
        min_length: float = 1.0,
        max_length: float = 60.0,
        min_level: float = 2.0,
        hysteresis: float = 0.0,
        detection_above_threshold: bool = True,
        min_inter_event_time: float = 1.0,
        feature: str = "rms",
        lowpass_window_length: Optional[float] = None,
    ):
        """Initialize the ThresholdEventDetector.

        Args:
            block_duration: Audio frame duration in seconds.
            feature: Feature used to trigger the threshold event detector.
            min_length: Minimal duration of an event to be detected (in seconds).
            max_length: Maximal duration of an event in seconds.
                If a detected event is longer, then returns the first part of this
                event of length equal to max_length.
            min_level: Minimum event level in unit of feature.
            hysteresis: Hysteresis value in unit of feature.
            min_inter_event_time: Minimum time between two events with no merge.
            lowpass_window_length: Length of the lowpass window used for convolution
                (in seconds). Default value is set to min_length duration.

        """
        self._block_duration = block_duration
        self.feature = feature
        self._min_length = min_length
        self._max_length = max_length
        self._min_level = min_level
        self._hysteresis = hysteresis
        self._min_inter_event_time = min_inter_event_time
        self._detection_above_threshold = detection_above_threshold
        self._min_frames = int(self._min_length / self._block_duration)
        self._max_frames = int(self._max_length / self._block_duration)

        # Boolean used to determine if last event detected is longer than
        # max_length (recording condition) and still ongoing.
        self._max_sized_event_detected_still_ongoing = False

        # Default value for lowpass window length.
        if lowpass_window_length is None:
            self._n_frames_lowpass_window = int(min_length / block_duration)
        # Lowpass window length in range of [block_duration,min_length].
        elif (lowpass_window_length >= block_duration) and (
            lowpass_window_length <= min_length
        ):
            self._n_frames_lowpass_window = int(lowpass_window_length / block_duration)
        else:
            self._n_frames_lowpass_window = int(min_length / block_duration)
            warnings.warn(
                "The lowpass window length should be in range of [block_duration, "
                "min_length] to remove small events. The lowpass window "
                f"length is set to min_length = {self._min_length}."
            )
        self._lowpass_window = np.ones(self._n_frames_lowpass_window)

        self._time_index = RingBuffer(self._max_frames)
        self._detection_data = np.zeros((1, self._max_frames))

        self._events = []
        self._last_event = []

    def push(self, features: dict):
        """Push features to detection_data RingBuffer.

        Args:
            features: Features with a time field.

        """
        if not (self.feature in features):
            raise ValueError(
                "threshold feature argument is unexpected," "got {}".format(features)
            )

        # Get the next buffer.
        self._time_index.append(features["time"])
        self._detection_data = np.roll(self._detection_data, -1)
        self._detection_data[0, -1] = features[self.feature]
        # Predict the events.
        self._events += self._next_events()

    def pop_event(self) -> Optional[Event]:
        """Return the last detected event or None if there is no detected event."""
        if not self._events:
            return None
        else:
            return self._events.pop(0)

    def _next_events(self) -> List[Event]:
        """Return a list of events in the current window."""

        # Skip test until a new event happened after the previous one saved.
        if len(self._last_event) > 0 and self._time_index[0] <= self._last_event[-1]:
            return []

        events = self._detect_events()
        events = merge_close_events(events, self._min_inter_event_time)
        # Remove events too short.
        events = [
            event
            for event in events
            if (event.end_time - event.start_time).total_seconds() >= self._min_length
        ]
        if len(events) > 0:
            self._last_event = events[-1]
        else:
            self._last_event = []

        return events

    def _detect_events(self) -> List[Event]:
        """Event detection routine.

        Returns:
            A list of event containing for each the start and end time index.

        """
        threshold_crossed_array = compute_hysteresis(
            self._detection_data[0, :],
            self._min_level,
            self._hysteresis,
            self._detection_above_threshold,
        )

        # Avoid detect false event at the begining of signal.
        if len(self._time_index) < self._time_index._length:
            threshold_crossed_array[: -len(self._time_index)] = 0

        # The whole buffer is above threshold.
        if np.min(threshold_crossed_array == 1):
            # Record event if it's not a still ongoing long event.
            if self._max_sized_event_detected_still_ongoing is False:
                offset_index = np.asarray([self._max_frames - 1])
                onset_index = np.asarray([0])
                self._max_sized_event_detected_still_ongoing = True
            else:
                # TODO: Event detection periodic fallback here
                # Wait for the end of the long ongoing event.
                return []
        else:
            # Last event was max buffer, wait for the end of the event.
            if self._max_sized_event_detected_still_ongoing:
                if threshold_crossed_array[0] != 0:
                    return []
                # End of last event > max_length
                self._max_sized_event_detected_still_ongoing = False

            threshold_crossed_array = lowpass_detection(
                input=threshold_crossed_array, window=self._lowpass_window
            )

            diff_event_detected = np.diff(threshold_crossed_array)
            # Take group delay into account for onset index.
            onset_index = (
                np.nonzero(diff_event_detected == 1)[0]
                - self._n_frames_lowpass_window
                + 1
            )
            offset_index = np.nonzero(diff_event_detected == -1)[0]

            # Get only the event (delay caused by the np.diff function).
            if len(onset_index) != 0:
                onset_index += 1

            # Start a new detection only after the crossing of threshold and hysteresis.
        return self._infer_events(self._time_index, onset_index, offset_index)

    def _infer_events(
        self,
        time: Sequence[dt.datetime],
        onset_index: np.ndarray,
        offset_index: np.ndarray,
    ) -> List[Event]:
        """Compute events based on onset and offset detections.

        Args:
            time: An array containing the timestamps of the considered
                frames.
            onset_index: Indices related to the time array where an
                onset is detected.
            offset_index: Indices related to the time array where an
                offset is detected.

        Returns:
            A list of event containing for each the start and end time index.

        """
        if len(onset_index) == 0 or len(offset_index) == 0:
            return []

        # Begin with an event.
        if onset_index[0] > offset_index[0]:
            onset_index = np.insert(
                onset_index, 0, 0
            )  # First frame is the beginning of an event.
        events = []

        # For all detected ONSETS/OFFSETS, crop to max_length if longer.
        for detection in np.arange(0, np.minimum(len(onset_index), len(offset_index))):
            event = Event(
                time[onset_index[detection] - self._max_frames + len(self._time_index)],
                time[
                    offset_index[detection] - self._max_frames + len(self._time_index)
                ],
            )
            if (event.end_time - event.start_time).total_seconds() > self._max_length:
                event = Event(
                    event.start_time,
                    event.start_time + dt.timedelta(seconds=self._max_length),
                )
            events.append(event)

        return events

    def flush(self) -> List[Event]:
        """Return a list of events in the last window.

        Returns:
            A list containing the events from the last window computation. This
            function is used when an event is saved and there is less than one window
            till the end of wave file or event detection usage.

        """
        # Do one more test over the buffer.
        events = self._detect_events()

        # Take only events after the last one saved.
        if len(events) > 0 and len(self._last_event) != 0:
            events = [
                event for event in events if (event.start_time > self._last_event[-1])
            ]

        events = merge_close_events(events, self._min_inter_event_time)
        events = [
            event
            for event in events
            if (event.end_time - event.start_time).total_seconds() >= self._min_length
        ]

        if len(events) > 0:
            self._last_event = events[-1]
        else:
            self._last_event = []

        return events


class TemporalEventDetector:
    """Class TemporalEventDetector that implement real-time event detection."""

    def __init__(
        self,
        block_duration: float,
        min_length: float = 1.0,
        max_length: float = 60.0,
        min_level: float = 2.0,
        hysteresis: float = 0.0,
        detection_above_threshold: bool = True,
        min_inter_event_time: float = 1.0,
        integration_window: float = 80.0,
        lowpass_window_length: Optional[float] = None,
    ):
        """Initialize the TemporalEventDetector.

        Args:
            block_duration: Audio frame duration in seconds.
            min_length: Minimal duration of an event to be
                detected (in seconds).
            max_length: Maximal duration of an event (in seconds).
                If a detected event is longer, then returns the first part of this
                event of length equal to max_length.
            min_level: Minimum event level (in dB).
            hysteresis: Hysteresis level value (in dB).
            detection_above_threshold: Boolean to determine if the detection is above
                or below the threshold level.
            min_inter_event_time: Minimum time between two events with no merge
                (in seconds).
            integration_window: Length of the integration window (in seconds).
            lowpass_window_length: Length of the lowpass window used for convolution
                (in seconds). Default value is set to min_length duration.

        """
        self._block_duration = block_duration
        self._min_length = min_length
        self._max_length = max_length
        self._min_level = min_level
        self._hysteresis = hysteresis
        self._detection_above_threshold = detection_above_threshold
        self._min_inter_event_time = min_inter_event_time
        self._max_sized_event_detected_still_ongoing = False
        self._integration_window = integration_window

        self._integration_frames = int(self._integration_window / self._block_duration)
        self._min_frames = int(self._min_length / self._block_duration)
        self._max_frames = int(self._max_length / self._block_duration)
        self._max_buffer_frames = max(self._integration_frames, self._max_frames)

        # Default value for lowpass window length.
        if lowpass_window_length is None:
            self._n_frames_lowpass_window = int(min_length / block_duration)
        # Lowpass window length in range of [block_duration,min_length].
        elif (lowpass_window_length >= block_duration) and (
            lowpass_window_length <= min_length
        ):
            self._n_frames_lowpass_window = int(lowpass_window_length / block_duration)
        else:
            self._n_frames_lowpass_window = int(min_length / block_duration)
            warnings.warn(
                "The lowpass window length should be in range of [block_duration, "
                "min_length] to remove small events. The lowpass window "
                f"length is set to min_length = {self._min_length}."
            )
        self._lowpass_window = np.ones(self._n_frames_lowpass_window)

        self._time_index = RingBuffer(self._max_buffer_frames)
        # first row: rms
        # second row: threshold
        # third row: sound pressure level
        self._detection_data = np.zeros((3, self._max_buffer_frames))

        self._events = []
        self._last_event = []

    def push(self, features: dict):
        """Push features to detection data RingBuffer.

        Args:
            features: Features with a time field.

        """
        # Get the next buffer.
        self._time_index.append(features["time"])
        self._detection_data = np.roll(self._detection_data, -1)
        self._detection_data[0, -1] = features["rms"]
        # RMS moving average.
        # First elements of detection_data when RingBuffer is not full.
        if len(self._time_index) < self._integration_frames:
            threshold = spl(
                self._detection_data[
                    0,
                    slice(
                        self._max_buffer_frames - len(self._time_index),
                        self._max_buffer_frames,
                    ),
                ].mean()
            )
        # Threshold computation when RingBuffer is full.
        else:
            threshold = spl(
                self._detection_data[
                    0,
                    slice(
                        self._max_buffer_frames - self._integration_frames,
                        self._max_buffer_frames,
                    ),
                ].mean()
            )
        # Sound pressure level (SPL).
        sound_level = spl(features["rms"])

        self._detection_data[1, -1] = threshold + self._min_level
        self._detection_data[2, -1] = sound_level
        # Predict the events.
        self._events += self._next_events()

    def pop_event(self) -> Optional[Event]:
        """Return the last detected event or None if there is no detected event."""
        if not self._events:
            return None
        else:
            return self._events.pop(0)

    def _next_events(self) -> List[Event]:
        """Return a list of events in the current window."""
        # Wait until a new event occure after the previous one.
        if len(self._last_event) > 0 and self._time_index[0] <= self._last_event[-1]:
            return []

        events = self._detect_events()
        events = merge_close_events(events, self._min_inter_event_time)
        events = [
            event
            for event in events
            if (event.end_time - event.start_time).total_seconds() >= self._min_length
        ]

        if len(events) > 0:
            self._last_event = events[-1]
        else:
            self._last_event = []

        return events

    def _detect_events(self) -> List[Event]:
        """Event detection routine.

        Returns:
            A list of event containing for each the start and end time index.

        """
        threshold_crossed_array = compute_hysteresis(
            self._detection_data[2, :],
            self._detection_data[1, :],
            self._hysteresis,
            self._detection_above_threshold,
        )

        # Avoid detect false event at the begining of signal.
        if len(self._time_index) < self._time_index._length:
            threshold_crossed_array[: -len(self._time_index)] = 0

        # The whole buffer is above threshold.
        if np.min(threshold_crossed_array == 1):
            # Record event if it's not a still ongoing long event.
            if self._max_sized_event_detected_still_ongoing is False:
                offset_index = np.asarray([self._max_frames - 1])
                onset_index = np.asarray([0])
                self._max_sized_event_detected_still_ongoing = True
            else:
                # TODO: Event detection periodic fallback here
                # Wait for the end of the long ongoing event.
                return []
        else:
            # Last event was max buffer, wait for the end of the event.
            if self._max_sized_event_detected_still_ongoing:
                if threshold_crossed_array[0] != 0:
                    return []
                # End of last event > max_length.
                self._max_sized_event_detected_still_ongoing = False

            threshold_crossed_array = lowpass_detection(
                input=threshold_crossed_array, window=self._lowpass_window
            )

            diff_event_detected = np.diff(threshold_crossed_array)
            # Take group delay into account for onset index.
            onset_index = (
                np.nonzero(diff_event_detected == 1)[0]
                - self._n_frames_lowpass_window
                + 1
            )
            offset_index = np.nonzero(diff_event_detected == -1)[0]

            # Get only the event (delay caused by the np.diff function).
            if len(onset_index) != 0:
                onset_index += 1

        return self._infer_events(self._time_index, onset_index, offset_index)

    def _infer_events(
        self,
        time: Sequence[dt.datetime],
        onset_index: np.ndarray,
        offset_index: np.ndarray,
    ) -> List[Event]:
        """Compute events based on onset and offset detections.

        Args:
            time: An array containing the timestamps of the considered
                frames.
            onset_index: Indices related to the time array where an
                onset is detected.
            offset_index: Indices related to the time array where an
                offset is detected.

        Returns:
            A list of event containing for each the start and end time index.
        """
        if len(onset_index) == 0 or len(offset_index) == 0:
            return []

        # Begin with an event
        if onset_index[0] > offset_index[0]:
            onset_index = np.insert(
                onset_index, 0, 0
            )  # First frame of the buffer is the beginning of an event.

        events = []

        # For all detected ONSETS/OFFSETS
        for detection in np.arange(0, np.minimum(len(onset_index), len(offset_index))):
            event = Event(
                time[
                    onset_index[detection]
                    - self._max_buffer_frames
                    + len(self._time_index)
                ],
                time[
                    offset_index[detection]
                    - self._max_buffer_frames
                    + len(self._time_index)
                ],
            )
            # Test if the event is longer than max_length parameter.
            if (event.end_time - event.start_time).total_seconds() > self._max_length:
                event = Event(
                    event.start_time,
                    event.start_time + dt.timedelta(seconds=self._max_length),
                )
            events.append(event)

        return events

    def flush(self) -> List[Event]:
        """Return a list of events in the last window.

        Returns:
            A list containing the events from the last window computation. This
            function is used when an event is saved and there is less than one window
            till the end of wave file or event detection usage.

        """
        # Do one more test over the buffer.
        events = self._detect_events()

        # Take only events after the last one saved.
        if len(events) > 0 and len(self._last_event) != 0:
            events = [
                event for event in events if (event.start_time > self._last_event[-1])
            ]

        events = merge_close_events(events, self._min_inter_event_time)
        events = [
            event
            for event in events
            if (event.end_time - event.start_time).total_seconds() >= self._min_length
        ]

        if len(events) > 0:
            self._last_event = events[-1]
        else:
            self._last_event = []

        return events


def merge_close_events(
    ordered_events: List[Event],
    min_inter_event_time: float,
) -> List[Event]:
    """Merge close events.

    Merge events such as the resulting list of events contains only pair of events
    such as the time difference between the end time of the first one and the start
    time of the second one is greater or equal to `min_inter_event_time`.

    Args:
        ordered_events: List of ordered events such as the one
            returned by `detect_events`.
        min_inter_event_time: Minimum time between two events before merging them
            into one (in secondes).

    Returns:
        An ordered list of events.

    """
    if not ordered_events:
        return []
    events = []
    event_start_time = ordered_events[0].start_time
    event_end_time = ordered_events[0].end_time

    for next_event in ordered_events[1:]:
        if (
            next_event.start_time - event_end_time
        ).total_seconds() < min_inter_event_time:
            event_end_time = next_event.end_time
        else:
            new_event = Event(event_start_time, event_end_time)
            events.append(new_event)
            event_start_time = next_event.start_time
            event_end_time = next_event.end_time
    last_event = Event(event_start_time, event_end_time)
    events.append(last_event)

    return events


def lowpass_detection(input: np.ndarray, window: np.ndarray) -> np.ndarray:
    """Convolve the input signal with a window and get the position of the
    result with window length (above or below) as a vector.

    Args:
        input: Signal to convolve and get info about short events.
        window: Window to convolve used to set the size of the events to be detected.

    Returns:
        The threshold crossed array with window convolution with group delay.

    """

    if len(window) == 1:
        return input

    # Apply a convolution between input and window.
    filtred_crossed_array = np.convolve(window, input)
    # Delete the end of the result and remove the window's length to center the signal.
    filtred_crossed_array = filtred_crossed_array[: -len(window) + 1] - len(window) + 1
    # Threshold crossed array with group delay.
    return filtred_crossed_array.clip(0, 1)
