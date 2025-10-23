import unittest

import numpy as np

from wavely.signal import settings
from wavely.signal.transfer_function.impulse_response import (
    extract_impulse_response,
    select_impulse_response_over_window,
)
from wavely.signal.transfer_function.transfer_function import (
    transfer_recording_microphone,
    transform_LEQ_band,
    transform_LEQ_value,
)


class TestTransferFunction(unittest.TestCase):
    @classmethod
    def setUp(cls):
        cls.source_microphone = "VESPER_VM1000"
        cls.target_microphone = "BK_4939"
        cls.method = "amplitude_measurements"
        transfer_parameters = settings.settings.TRANSFER_PARAMETERS[cls.method][
            "coef_vesper_to_bk"
        ]
        cls.weight = transfer_parameters["weight"]
        cls.domain = transfer_parameters["domain"]
        cls.band_freq = [100, 1000, 5000, 6000]
        cls.incorrect_band_freq = [0, 96000]

        cls.feats = {"bandleq": np.array([[50.0, 60.0, 70.0, 50.0]])}
        cls.db = True
        cls.transformed_bandleq = np.array(
            [[np.nan, 47.89751829, 58.51016994, 41.57036833]]
        )

        cls.sample_rate = 96000
        cls.signal = 2 * np.random.rand(10 * cls.sample_rate) - 1

    def test_transform_LEQ_value(self):
        transformed_leq_val = transform_LEQ_value(
            self.source_microphone,
            self.target_microphone,
            self.band_freq[-2:],
            self.feats["bandleq"][0, 2],
            self.method,
        )
        self.assertAlmostEqual(transformed_leq_val, 73.929994462)

        transformed_leq_val = transform_LEQ_value(
            self.source_microphone,
            self.target_microphone,
            self.incorrect_band_freq,
            self.feats["bandleq"][0, 2],
            self.method,
        )
        self.assertTrue(np.isnan(transformed_leq_val))

    def test_transform_LEQ_band(self):
        transformed_features = transform_LEQ_band(
            self.source_microphone,
            self.target_microphone,
            self.band_freq,
            self.feats,
            self.db,
            self.method,
        )
        self.assertIn("transformed_bandleq", transformed_features)
        self.assertTrue(np.isnan(transformed_features["transformed_bandleq"][0, 0]))
        for computed_val, true_val in zip(
            transformed_features["transformed_bandleq"][0, 1:],
            self.transformed_bandleq[0, 1:],
        ):
            self.assertAlmostEqual(computed_val, true_val)

    def test_transfer_recording_microphone(self):
        filtered_signal = transfer_recording_microphone(
            self.signal,
            self.sample_rate,
            self.source_microphone,
            self.target_microphone,
        )
        self.assertEqual(len(filtered_signal), len(self.signal))

    def tearDown(self):
        pass


class TestImpulseResponse(unittest.TestCase):
    @classmethod
    def setUp(cls):
        pass

    def test_extract_impulse_response(self):
        pass

    def test_select_impulse_response_over_window(self):
        pass

    def tearDown(self):
        pass
