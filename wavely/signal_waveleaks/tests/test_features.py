import unittest

import numpy as np
from numpy.testing import assert_array_almost_equal, assert_array_equal

from wavely.signal.features.features import FeaturesComputer


class TestFeaturesComputer(unittest.TestCase):
    @classmethod
    def setUpClass(cls):

        cls.rate = 192000
        cls.window = np.hanning

        # simple sin signal with noise
        cls.block_size = 512
        cls.f = 15
        cls.n_fft = int(cls.block_size * 3.2)
        cls.t = np.arange(0.0, 1.0, 1 / cls.block_size)
        cls.x = np.sin(2.0 * np.pi * cls.f * cls.t)
        cls.x += np.random.randn(cls.block_size) * 0.1

        cls.af = FeaturesComputer(
            block_size=cls.block_size,
            rate=cls.rate,
            n_fft=cls.n_fft,
            window=cls.window,
            features="all",
        )
        cls.features = cls.af.compute(np.expand_dims(cls.x, 0))

        cls.aff = FeaturesComputer(
            block_size=cls.block_size,
            rate=cls.rate,
            n_fft=cls.n_fft,
            window=cls.window,
            features=["spl"],
        )

    def test_gl_filter_bands_value(self):
        Wn_gla = np.array([0, 1])
        Wn_glv = np.array([0, 1])
        af = FeaturesComputer(
            block_size=self.block_size, rate=self.rate, Wn_gla=Wn_gla, Wn_glv=Wn_glv
        )
        self.assertTrue((af.params["Wn_gla"] == Wn_gla).all())
        self.assertTrue((af.params["Wn_glv"] == Wn_glv).all())

    def test_only_compute_default_feature(self):

        af = FeaturesComputer(
            block_size=self.block_size,
            rate=self.rate,
            n_fft=self.n_fft,
            window=np.hanning,
        )
        spl = af.compute(np.expand_dims(self.x, 0))
        self.assertIs(len(spl.keys()), 1)
        self.assertIn("spl", spl)

    def test_all_features_registered_for_export_are_computed(self):

        self.assertSetEqual(
            FeaturesComputer._exported_features, set(self.features.keys())
        )

    def test_add_feature(self):

        af = FeaturesComputer(
            block_size=self.block_size,
            rate=self.rate,
            n_fft=self.n_fft,
            window=np.hanning,
        )

        @af.feature
        def testfeature(blocks: np.ndarray) -> float:
            return blocks.mean(axis=-1)

        features = af.compute(np.expand_dims(self.x, 0))

        self.assertIn("testfeature", features.keys())
        self.assertIn("testfeature", af._features_function.keys())
        self.assertIn("testfeature", af._exported_features)

        # new feature is registered only on the class instance above
        self.assertNotIn("testfeature", self.af._exported_features)
        self.assertNotIn("testfeature", self.af._features_function.keys())

    def test_store_periodogram(self):

        af = FeaturesComputer(
            block_size=self.block_size,
            rate=self.rate,
            n_fft=self.n_fft,
            window=np.hanning,
        )
        af.compute(np.expand_dims(self.x, 0))
        previous_value = af["periodogram"]
        af.compute(np.expand_dims(self.x, 0))
        assert_array_equal(af["prev_periodogram"], previous_value)

    def test_batch_process(self):
        # processing as a single batch or as multiple batches containing
        # a single block leads to the same result

        nblocks = 8
        x = self.x.reshape(nblocks, -1)
        block_size = x.shape[1]

        af1 = FeaturesComputer(
            block_size=block_size, rate=self.rate, n_fft=self.n_fft, window=np.hanning
        )
        af2 = FeaturesComputer(
            block_size=block_size, rate=self.rate, n_fft=self.n_fft, window=np.hanning
        )

        # process all blocks at once
        features_batch = af1.compute(x)
        # process block by block
        features = [af2.compute(x[[i], :]) for i in range(nblocks)]

        for name in features_batch.keys():
            assert_array_almost_equal(
                features_batch[name], [features[i][name][0] for i in range(nblocks)]
            )

    def test_no_flatten(self):

        features = self.af.compute(np.expand_dims(self.x, 0))
        self.assertIn("bandleq", features.keys())
        self.assertIsInstance(features["bandleq"], np.ndarray)

    def test_flatten(self):
        features = self.af.compute(np.expand_dims(self.x, 0), flatten=True)
        self.assertNotIn("bandleq", features.keys())
        self.assertIn("bandleq1", features.keys())
        self.assertIsInstance(features["bandleq1"], np.ndarray)

    def test_flatten_with_batch(self):

        nblocks = 8
        x = self.x.reshape(nblocks, -1)
        block_size = x.shape[1]

        af1 = FeaturesComputer(
            block_size=block_size, rate=self.rate, n_fft=self.n_fft, window=np.hanning
        )
        af2 = FeaturesComputer(
            block_size=block_size, rate=self.rate, n_fft=self.n_fft, window=np.hanning
        )

        # process all blocks at once
        features_batch = af1.compute(x, flatten=True)
        # process block by block
        features = [af2.compute(x[[i], :], flatten=True) for i in range(nblocks)]

        for name in features_batch.keys():
            assert_array_almost_equal(
                features_batch[name], [features[i][name][0] for i in range(nblocks)]
            )

    def test_only_compute_subset_of_features(self):
        target_features = ["spectralcentroid"]
        af = FeaturesComputer(
            block_size=self.block_size,
            rate=self.rate,
            n_fft=self.n_fft,
            window=np.hanning,
            features=target_features,
        )
        res = af.compute(np.expand_dims(self.x, 0))
        self.assertEqual(list(res.keys()), target_features)

    def test_invalid_features_argument(self):
        target_features = ["foo"]
        with self.assertRaises(ValueError):
            FeaturesComputer(
                block_size=self.block_size,
                rate=self.rate,
                n_fft=self.n_fft,
                window=np.hanning,
                features=target_features,
            )

        target_features = []
        with self.assertRaises(ValueError):
            FeaturesComputer(
                block_size=self.block_size,
                rate=self.rate,
                n_fft=self.n_fft,
                window=np.hanning,
                features=target_features,
            )

    def test_features_selection_with_dims(self):
        def test_dim(dim):
            af = FeaturesComputer(
                block_size=self.block_size,
                rate=self.rate,
                n_fft=self.n_fft,
                window=np.hanning,
                features="all",
                dims=dim,
            )
            features = af.compute(np.expand_dims(self.x, 0))

            for f in features.keys():
                f_dim = len(FeaturesComputer._features_dimensions[f])
                self.assertEqual(f_dim, int(dim[0][0]))

        dims_to_test = [["1d"], ["2d"]]
        for dim in dims_to_test:
            test_dim(dim)

    def test_features_selection_with_tags(self):
        def test_tag(tag):
            af = FeaturesComputer(
                block_size=self.block_size,
                rate=self.rate,
                n_fft=self.n_fft,
                window=np.hanning,
                features="all",
                tags=[tag],
            )
            features = af.compute(np.expand_dims(self.x, 0))

            self.assertCountEqual(FeaturesComputer._tag_features[tag], features.keys())
            self.assertEqual(
                set(FeaturesComputer._tag_features[tag]), set(features.keys())
            )
            for f in features:
                if f in FeaturesComputer._tag_features["2d"]:
                    self.assertTrue(np.allclose(features[f], self.features[f]))
                else:
                    self.assertEqual(features[f], self.features[f])

        tags_to_test = ["1d", "temporal", "vibration", "spectral", "acoustic", "2d"]
        for tag in tags_to_test:
            test_tag(tag)

    def test_tag_does_not_exist(self):
        tag = "a_tag_that_should_never_happend_or_it_would_be_strange"
        with self.assertRaises(ValueError):
            FeaturesComputer(
                block_size=self.block_size,
                rate=self.rate,
                n_fft=self.n_fft,
                window=np.hanning,
                features="all",
                tags=[tag],
            )
