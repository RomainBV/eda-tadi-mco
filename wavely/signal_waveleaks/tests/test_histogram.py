import unittest

import numpy as np
from numpy.testing import assert_array_almost_equal, assert_array_equal

from wavely.signal.processing.histogram import (
    FadingHistogram,
    HistogramSpecification,
    MergedHistogram,
    SlidingHistogram,
    find_min_bin,
    find_sorted_index,
    merge_bins,
)


class TestSlidingHistogram(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.n_bins = 11
        cls.max_value = 100
        cls.min_value1 = 0
        cls.min_value2 = 10

        cls.expected_bins = np.asarray([0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100])

    def test_class_creation(self):
        histogram_specification = HistogramSpecification(
            n_bins=self.n_bins, max_value=self.max_value, min_value=self.min_value1
        )
        histogram_strategy = SlidingHistogram(specification=histogram_specification)
        sliding_window_histogram1 = histogram_strategy.create_online_histogram()

        self.assertEqual(sliding_window_histogram1._min_value, self.min_value1)
        self.assertEqual(sliding_window_histogram1._max_value, self.max_value)
        assert_array_equal(sliding_window_histogram1.get_bins(), self.expected_bins)

        sliding_window_histogram1.reset()

        histogram_specification = HistogramSpecification(
            n_bins=self.n_bins - 1, max_value=self.max_value, min_value=self.min_value2
        )
        histogram_strategy = SlidingHistogram(specification=histogram_specification)
        sliding_window_histogram2 = histogram_strategy.create_online_histogram()

        assert_array_equal(sliding_window_histogram2.get_bins(), self.expected_bins[1:])

    def test_insert(self):
        histogram_specification = HistogramSpecification(
            n_bins=self.n_bins, max_value=self.max_value
        )
        histogram_strategy = SlidingHistogram(specification=histogram_specification)
        sliding_window_histogram = histogram_strategy.create_online_histogram()

        input_data1 = np.asarray([10, 11, 12])
        expected_frequencies = np.asarray([0, 3, 0, 2])

        sliding_window_histogram.insert(input_data1)
        test_frequency = sliding_window_histogram.get_frequencies()
        assert_array_equal(test_frequency[:1], expected_frequencies[:1])

        input_data2 = np.asarray([26, 28.5])
        sliding_window_histogram.insert(input_data2)
        test_frequency = sliding_window_histogram.get_frequencies()
        self.assertEqual(test_frequency[3], expected_frequencies[3])

        sliding_window_histogram.reset()
        zero_array = np.zeros(self.n_bins)
        assert_array_equal(sliding_window_histogram.get_frequencies(), zero_array)


class TestFadingHistogram(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.n_bins = 5
        cls.max_value = 3000
        cls.input_bins1 = [
            10,
            20,
            100,
            2000,
            1000,
            200,
            30,
            300,
            3000,
        ]
        cls.alpha = 0.9

        cls.histogram_specification = HistogramSpecification(
            n_bins=cls.n_bins, max_value=cls.max_value
        )
        cls.histogram_strategy = FadingHistogram(
            specification=cls.histogram_specification, alpha=cls.alpha
        )

        cls.default_min_value = 0
        cls.expected_frequencies_fading_array = np.array([3.88, 0.66, 0, 0.59, 1])

    def test_class_creation(self):
        online_fading_histogram = self.histogram_strategy.create_online_histogram()

        self.assertEqual(online_fading_histogram._n_bins, self.n_bins)
        self.assertEqual(online_fading_histogram._max_value, self.max_value)
        self.assertEqual(online_fading_histogram._histogram_strategy._alpha, self.alpha)
        self.assertEqual(
            online_fading_histogram._min_value,
            self.default_min_value,
        )

    def test_insert(self):
        online_fading_histogram = self.histogram_strategy.create_online_histogram()
        online_fading_histogram.insert(self.input_bins1)

        assert_array_almost_equal(
            online_fading_histogram.get_frequencies(),
            self.expected_frequencies_fading_array,
            decimal=2,
        )

        online_fading_histogram.reset()
        zero_array = np.zeros(self.n_bins)
        assert_array_equal(online_fading_histogram.get_frequencies(), zero_array)


class TestMergedHistogram(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.n_bins = 5
        cls.input_bins = [10, 100, 1000, 20, 200, 2000, 30, 300, 3000]
        cls.histogram_specification = HistogramSpecification(n_bins=cls.n_bins)
        cls.histogram_strategy = MergedHistogram(
            specification=cls.histogram_specification
        )

        cls.expected_merged_bins = [40, 250, 1000, 2000, 3000]
        cls.expected_merged_frequencies = [4, 2, 1, 1, 1]
        cls.expected_centroids = 740
        cls.quantil50 = 92.5
        cls.quantil10 = 40

    def test_class_creation(self):
        online_merged_histogram = self.histogram_strategy.create_online_histogram()

        assert_array_equal(
            online_merged_histogram.get_bins(), np.array([], dtype=np.float)
        )

    def test_insert(self):
        online_merged_histogram = self.histogram_strategy.create_online_histogram()
        online_merged_histogram.insert(self.input_bins[0:4])

        assert_array_equal(
            online_merged_histogram.get_bins(), np.sort(self.input_bins[0:4])
        )

        online_merged_histogram.reset()

        zero_array = np.array([])

        assert_array_equal(online_merged_histogram.get_frequencies(), zero_array)
        assert_array_equal(online_merged_histogram.get_bins(), zero_array)

    def test_insert_large_dataset(self):
        online_merged_histogram = self.histogram_strategy.create_online_histogram()
        online_merged_histogram.insert(self.input_bins)

        assert_array_equal(
            online_merged_histogram.get_bins(), self.expected_merged_bins
        )
        assert_array_equal(
            online_merged_histogram.get_frequencies(), self.expected_merged_frequencies
        )


class TestHistogramTools(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.n_bins = 5

        cls.histogram_specification = HistogramSpecification(n_bins=cls.n_bins)
        cls.histogram_strategy = MergedHistogram(
            specification=cls.histogram_specification
        )

        cls.input_bins = [10, 200, 1000, 20, 100, 2000, 3000, 300, 30]
        cls.frequencies = [1, 1, 1, 1, 1, 1, 1, 1, 1]

        cls.expected_merged_bins = [40, 250, 1000, 2000, 3000]
        cls.expected_merged_frequencies = [4, 2, 1, 1, 1]

        cls.expected_centroid = [740.0]
        cls.expected_two_centroids = [237.143, 2500]

        cls.quantil50 = 92.5
        cls.quantil10 = 40

    def test_histogram_strategy_error(self):
        histogram_specification_zero_error = HistogramSpecification(n_bins=0)
        histogram_specification_no_width = HistogramSpecification(
            n_bins=10, max_value=100, min_value=100
        )

        with self.assertRaises(ValueError):
            MergedHistogram(specification=histogram_specification_zero_error)
            MergedHistogram(specification=histogram_specification_no_width)

    def test_get_centroid(self):
        online_merged_histogram = self.histogram_strategy.create_online_histogram()
        online_merged_histogram.insert(self.input_bins)

        centroid = online_merged_histogram.get_centroids(1)
        two_centroids = online_merged_histogram.get_centroids(2)

        self.assertEqual(self.expected_centroid, centroid)
        assert_array_almost_equal(self.expected_two_centroids, two_centroids, 2)

    def test_get_quantile(self):
        online_merged_histogram = self.histogram_strategy.create_online_histogram()
        online_merged_histogram.insert(self.input_bins)

        quantil50 = online_merged_histogram.get_quantile(0.5)
        quantil10 = online_merged_histogram.get_quantile(0.1)

        self.assertEqual(quantil10, self.quantil10)
        self.assertEqual(quantil50, self.quantil50)

    def test_merge_bins(self):
        input_bins = [0.0, 20.0, 40.0, 100.0, 160.0, 1000.0]
        frequencies = [1.0, 4.0, 77.0, 1.0, 0.0, 0.0]

        expected_bins = [0, 20, 40.77, 160, 1000]
        expected_frequencies = [1, 4, 78, 0, 0]

        merged_bins, merged_frequencies = merge_bins(input_bins, frequencies, 2)

        assert_array_almost_equal(expected_bins, merged_bins, 2)
        assert_array_equal(expected_frequencies, merged_frequencies)

    def test_find_min_bin(self):
        input_bins = [0, 7000, 8000, 10000, 15000, 15999]

        expected_min_bins = 4

        min_bin = find_min_bin(input_bins)

        self.assertEqual(expected_min_bins, min_bin)

    def test_find_sorted_index(self):
        input_bins = [0.0, 16.5, 28.0, 49.0, 78.4, 85.0]
        point = 48.5
        expected_index = 3

        index = find_sorted_index(input_bins, point)

        self.assertEqual(expected_index, index)


if __name__ == "__main__":
    unittest.main()
