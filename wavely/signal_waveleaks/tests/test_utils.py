import unittest

import numpy as np

from wavely.signal.units.converters import db2lin, g_to_ms2, lin2db, ms2_to_g


class TestUtils(unittest.TestCase):
    def test_lin_db(self):
        self.assertAlmostEqual(lin2db(1.0), 0.0)
        self.assertAlmostEqual(lin2db(0.0), -156.5355977)
        self.assertEqual(lin2db(10.0), 10.0)

        self.assertEqual(db2lin(0.0), 1.0)
        self.assertEqual(db2lin(-np.inf), 0.0)
        self.assertEqual(db2lin(10.0), 10.0)

        self.assertAlmostEqual(lin2db(db2lin(43.21)), 43.21)

        self.assertAlmostEqual(g_to_ms2(1.0), 9.81)
        self.assertAlmostEqual(ms2_to_g(9.81), 1)
        self.assertAlmostEqual(g_to_ms2(ms2_to_g(9.81)), 9.81)
