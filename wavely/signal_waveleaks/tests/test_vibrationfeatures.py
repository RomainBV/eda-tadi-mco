import unittest

import numpy as np

from wavely.signal.features.features import FeaturesComputer
from wavely.signal.features.vibrationfeatures import (
    REF_ACCELERATION,
    REF_VELOCITY,
    acceleration_level,
    gl_filter,
    velocity_level,
)
from wavely.signal.units.helpers import split_signal


class TestVibrationFeatures(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.rate = 192e3
        cls.N = 4096
        cls.f0 = 5e3
        cls.t = np.arange(cls.N) / cls.rate
        cls.x_lin = (1 / 9.81) * cls.t  # linear acceleration
        cls.x_sin = np.sin(2 * np.pi * cls.f0 * cls.t)  # sinusoidal acceleration
        cls.block_size = 512
        cls.blocks_lin = split_signal(cls.x_lin, cls.rate, cls.block_size / cls.rate)
        cls.blocks_sin = split_signal(cls.x_sin, cls.rate, cls.block_size / cls.rate)
        cls.f1 = 100
        cls.f2 = 2e3
        cls.order = 5
        cls.Wn = np.array([cls.f1, cls.f2])
        cls.fc = FeaturesComputer(
            block_size=cls.block_size,
            rate=cls.rate,
            features=[
                "acceleration",
                "velocity",
                "displacement",
                "gl_acceleration",
                "gl_velocity",
            ],
        )
        cls.feats_lin = cls.fc.compute(cls.blocks_lin)
        cls.feats_sin = cls.fc.compute(cls.blocks_sin)

    def test_gl_filter(self):
        with self.assertRaises(ValueError):
            gl_filter(blocks=self.blocks_sin, rate=self.rate, Wn=0, order=self.order)
        with self.assertRaises(ValueError):
            gl_filter(
                blocks=self.blocks_sin,
                rate=self.rate,
                Wn=self.Wn,
                order=self.order,
                filter_output="sos",
            )

    def test_acceleration(self):
        true_acceleration = self.t
        self.assertAlmostEqual(
            true_acceleration.mean(), self.feats_lin["acceleration"].flatten("C").mean()
        )

    def test_velocity(self):
        true_velocity = (1 / 2) * self.t ** 2
        self.assertAlmostEqual(
            true_velocity.mean(), self.feats_lin["velocity"].flatten("C").mean()
        )

    def test_displacement(self):
        true_displacement = (1 / 6) * self.t ** 3
        self.assertAlmostEqual(
            true_displacement.mean(), self.feats_lin["displacement"].flatten("C").mean()
        )

    def test_gl_acceleration(self):
        self.assertAlmostEqual(
            self.feats_sin["gl_acceleration"].mean(), (np.sqrt(2) / 2), places=1
        )

    def test_acceleration_level(self):
        self.assertAlmostEqual(
            acceleration_level(REF_ACCELERATION * np.ones(self.block_size)), 0.0
        )

    def test_velocity_level(self):
        self.assertAlmostEqual(
            velocity_level(REF_VELOCITY * np.ones(self.block_size)), 0.0
        )

    # def test_gl_velocity(self):
    #    self.assertAlmostEqual(self.af["gl_velocity"][0], 2.84184478e+08)
    # TODO: Think about a suitable test
