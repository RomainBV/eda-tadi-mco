import dataclasses
import enum
import random
import warnings
from abc import ABC, abstractmethod
from typing import List, Optional, Tuple, Union

import numpy as np

# Alpha coeffcient for FadingHistogram.
# This value is described in the following paper: Raquel Sebastião, João Gama and
# Teresa Mendonça, "Constructing Fading Histograms from Data Streams",
# Progress in Artificial Intelligence 3(1):15-28
DEFAULT_ALPHA_COEFFICIENT = 0.997


class HistogramType(enum.Enum):
    fading = "fading"
    merged = "merged"
    sliding = "sliding"


@dataclasses.dataclass
class HistogramSpecification:
    n_bins: int
    max_value: Optional[float] = None
    min_value: Optional[float] = 0.0


class OnlineHistogram:
    """An online approximation histogram builder.

    +------------------+        Uses     +--------------------+
    | Online histogram |─ ─ ─ ─ ─ ─ ─ ─ ─> Histogram strategy |
    +------------------+                 +----------+---------+
                                                    |
                                    +---------------+-------------+
                                    |                             |
                            +-------v--------+           +--------v--------+
                            | FixedHistogram |           | MergedHistogram |
                            +-------+--------+           +-----------------+
                                    |
                        +-----------+------------+
                        |                        |
              +---------v--------+      +--------v--------+
              | SlidingHistogram |      | FadingHistogram |
              +------------------+      +-----------------+

    """

    def __init__(
        self,
        histogram_strategy: "HistogramStrategy",
        bins: np.array,
        frequencies: np.array,
        specification: HistogramSpecification,
    ):
        """Initialize the OnlineHistogram class.

        Args:
            histogram_strategy: The histogram strategy to build.
            bins: The array of bins of the histogram.
            frequencies: The array of frequencies of the histogram.
            specification: Histogram specification.

        """
        self._n_bins = specification.n_bins
        self._max_value = specification.max_value
        self._min_value = specification.min_value
        self._histogram_strategy = histogram_strategy
        self.bins = bins
        self.frequencies = frequencies

    def insert(self, points: Union[int, float, List[Union[int, float]]]):
        """Insert points into an histogram.

        Args:
            points: Value(s) to insert in the histogram.

        """
        self._histogram_strategy.insert(self, points)

    def reset(self):
        """Reset frequencies of the histogram."""
        self._histogram_strategy.reset(self)

    def get_centroids(self, n_clusters: int) -> List[float]:
        """Get the centroids of the histogram, knowing the number of clusters.

        Args:
            n_clusters: Number of cluster to apply before computing the spectral
                centroids of bins merged.

        Return:
            The list of centroids (with n_clusters size) after the merged of the input
                bins.

        """
        if n_clusters > len(self.bins):
            raise ValueError(
                f"Unexpected number of cluster {n_clusters} "
                f"for an histogram of size {len(self.bins)}."
            )
        bins_cluster = self.bins.copy()
        frequencies_cluster = self.frequencies.copy()

        # Merge bins according to number of cluster with float frequencies.
        while len(bins_cluster) > n_clusters:
            (bins_cluster, frequencies_cluster) = merge_bins(
                bins=bins_cluster,
                frequencies=frequencies_cluster,
                index=find_min_bin(bins_cluster),
                round_frequencies=False,
            )
        return [bin for bin in bins_cluster]

    def get_quantile(self, q: float):
        """Compute approximation of the quantile value for the current histogram.

        Args:
            q: The quantile in the range (0, 1).

        Retruns:
            The quantile value.

        """
        if not 0 < q < 1:
            raise ValueError(f"Unexpected quantile {q} : valid values are in (0, 1).")

        total_frequency = np.sum(self.frequencies)

        cumulative_frequency = 0.0
        quantile_value = 0.0
        for i, _ in enumerate(self.bins):
            cumulative_frequency += self.frequencies[i]
            if cumulative_frequency / total_frequency < q:
                continue

            if i == 0:
                quantile_value = self.bins[0]
                break

            cumulative_frequency -= self.frequencies[i]
            quantile_value = self.bins[i - 1] + (
                q * total_frequency - cumulative_frequency
            ) * ((self.bins[i] - self.bins[i - 1]) / self.frequencies[i])
            break

        return quantile_value

    def get_bins(self) -> np.ndarray:
        """Get all the bins of the histogram.

        Return:
            A shape-(n,) array containing the bins of the histogram.

        """
        return self.bins

    def get_frequencies(self) -> np.ndarray:
        """Get all the frequencies of the histogram.

        Return:
            A shape-(n,) array containing the frequencies of the histogram.

        """
        return self.frequencies


class HistogramStrategy(ABC):
    """An online approximation histogram strategy."""

    @abstractmethod
    def insert(
        self,
        online_histogram: OnlineHistogram,
        points: Union[int, float, List[Union[int, float]]],
    ):
        """Insert points into a fading histogram.

        Args:
            online_histogram: The OnlineHistogram to insert points.
            points: Value(s) to insert in the histogram.

        """
        raise NotImplementedError

    @abstractmethod
    def reset(
        self,
        online_histogram: OnlineHistogram,
    ):
        """Reset frequencies in the histogram.

        Args:
            online_histogram: The OnlineHistogram to reset.
        """
        raise NotImplementedError


class FixedHistogram(HistogramStrategy):
    """An histogram strategy with fixed bin width."""

    def reset(self, online_histogram: OnlineHistogram):
        """Reset frequencies of the histogram.

        Args:
            online_histogram: The OnlineHistogram to reset.

        """
        online_histogram.frequencies[:] = 0


class FadingHistogram(FixedHistogram):
    """
    The algorithm used here is presented in the following paper:
    Raquel Sebastião, João Gama and Teresa Mendonça,
    "Constructing Fading Histograms from Data Streams",
    Progress in Artificial Intelligence 3(1):15-28

    """

    def __init__(
        self,
        specification: HistogramSpecification,
        alpha: float = DEFAULT_ALPHA_COEFFICIENT,
    ):
        """Initialization of the FadingHistogram class.

        Args:
            specification: The histogram specification.
            alpha: The exponential fading factor, such that alpha is in range of ]0,1[.

        """
        if specification.n_bins <= 0:
            raise ValueError(
                "The number of bins in the HistogramSpecification must be greater than"
                " 1."
            )
        else:
            self._specification = specification

        if specification.max_value == specification.min_value:
            raise ValueError(
                "The min and the max value of the HistogramSpecification can't be the"
                " same."
            )
        else:
            self._bin_width = (specification.max_value - specification.min_value) / (
                specification.n_bins - 1
            )

        self._alpha = alpha

    def create_online_histogram(self) -> OnlineHistogram:
        """Initialize and create the OnlineHistogram object."""

        bins = (
            np.arange(self._specification.n_bins) * self._bin_width
            + self._specification.min_value
        )
        frequencies = np.zeros(self._specification.n_bins)
        return OnlineHistogram(
            histogram_strategy=self,
            specification=self._specification,
            bins=bins,
            frequencies=frequencies,
        )

    def insert(
        self,
        online_histogram: OnlineHistogram,
        points: Union[int, float, List[Union[int, float]]],
    ):
        """Insert points into a fading histogram.

        Args:
            online_histogram: The OnlineHistogram to insert points.
            points: Value(s) to insert in the histogram.

        """
        if isinstance(points, (np.ndarray, List)):
            for point in points:
                online_histogram.insert(point)
        else:
            if points > online_histogram._max_value:
                points = online_histogram._max_value
                warnings.warn("point added to the last bin.")
            elif points < online_histogram._min_value:
                points = online_histogram._min_value
                warnings.warn("point added to the first bin.")
            index_point = int(
                np.round((points - online_histogram._min_value) / self._bin_width)
            )
            online_histogram.frequencies = online_histogram.frequencies * self._alpha
            online_histogram.frequencies[index_point] += 1.0


class SlidingHistogram(FixedHistogram):
    """
    The algorithm used here is the classic representation of an histogram over a sliding
    window with fixed bin width.

    """

    def __init__(self, specification: HistogramSpecification):
        """Initialization of the SlidingHistogram class.

        Args:
            specification: The histogram specification.

        """
        if specification.n_bins <= 0:
            raise ValueError(
                "The number of bins in the HistogramSpecification must be greater than"
                " 1."
            )
        else:
            self._specification = specification

        if specification.max_value == specification.min_value:
            raise ValueError(
                "The min and the max value of the HistogramSpecification can't be the"
                " same."
            )
        else:
            self._bin_width = (specification.max_value - specification.min_value) / (
                specification.n_bins - 1
            )

        self._specification = specification

    def create_online_histogram(self) -> OnlineHistogram:
        """Initialize and create the OnlineHistogram object."""

        bins = (
            np.arange(self._specification.n_bins) * self._bin_width
            + self._specification.min_value
        )
        frequencies = np.zeros(self._specification.n_bins)
        return OnlineHistogram(
            histogram_strategy=self,
            specification=self._specification,
            bins=bins,
            frequencies=frequencies,
        )

    def insert(
        self,
        online_histogram: OnlineHistogram,
        points: Union[int, float, List[Union[int, float]]],
    ):
        """Insert points into a sliding histogram.

        Args:
            online_histogram: The OnlineHistogram to insert points.
            points: Value(s) to insert in the histogram.

        """
        if isinstance(points, (np.ndarray, List)):
            for point in points:
                online_histogram.insert(point)
        else:
            if points > online_histogram._max_value:
                points = online_histogram._max_value
                warnings.warn("point added to the last bin.")
            elif points < online_histogram._min_value:
                points = online_histogram._min_value
                warnings.warn("point added to the first bin.")
            index_point = int(
                np.round((points - online_histogram._min_value) / self._bin_width)
            )
            online_histogram.frequencies[index_point] += 1.0


class MergedHistogram(HistogramStrategy):
    """
    This strategy allows to build an histogram iteratively based on an online
    algorithm. This algorithm is presented in the following paper:
    Yael Ben-Haim and Elad Tom-Tov, "A streaming parallel decision tree algorithm",
    J. Machine Learning Research 11 (2010), pp. 849--872.

    """

    def __init__(self, specification: HistogramSpecification):
        """Initialize the MergedHistogram class.

        Args:
            specification: The histogram specification.

        """
        if specification.n_bins <= 0:
            raise ValueError(
                "The number of bins in the HistogramSpecification must be greater than"
                " 1."
            )
        else:
            self._specification = specification

    def create_online_histogram(self) -> OnlineHistogram:
        """Initialize and create the OnlineHistogram object."""

        bins = np.array([], dtype=np.float)
        frequencies = np.array([], dtype=np.int)
        return OnlineHistogram(
            histogram_strategy=self,
            specification=self._specification,
            bins=bins,
            frequencies=frequencies,
        )

    def insert(
        self,
        online_histogram: OnlineHistogram,
        points: Union[float, int, List[Union[int, float]]],
    ):
        """Insert points into a merged histogram.

        Args:
            online_histogram: The OnlineHistogram to insert points.
            points: Value(s) to insert in the histogram.

        """
        if isinstance(points, (np.ndarray, List)):
            for point in points:
                online_histogram.insert(point)
        else:
            if len(online_histogram.bins) == 0:
                online_histogram.bins = np.append(online_histogram.bins, points)
                online_histogram.frequencies = np.append(
                    online_histogram.frequencies, 1
                )
                return

            i = find_sorted_index(online_histogram.bins, points)
            if i < len(online_histogram.bins) and online_histogram.bins[i] == points:
                online_histogram.frequencies[i] += 1.0
            else:
                online_histogram.bins = np.insert(online_histogram.bins, i, points)
                online_histogram.frequencies = np.insert(
                    online_histogram.frequencies, i, 1
                )

            if len(online_histogram.bins) > online_histogram._n_bins:
                (online_histogram.bins, online_histogram.frequencies) = merge_bins(
                    bins=online_histogram.bins,
                    frequencies=online_histogram.frequencies,
                    index=find_min_bin(online_histogram.bins),
                    round_frequencies=True,
                )

    def reset(self, online_histogram):
        """Reset frequencies of the histogram.
        Args:
            online_histogram: The OnlineHistogram to reset.

        """
        online_histogram.bins = np.array([], dtype=np.float)
        online_histogram.frequencies = np.array([], dtype=np.int)


def merge_bins(
    bins: np.ndarray,
    frequencies: np.ndarray,
    index: int,
    round_frequencies: bool = True,
) -> Tuple[np.ndarray, np.ndarray]:
    """Merge bins at index `i` and `i`+1.

    Args:
        bins: The list of the bins where the merge bins should be done.
        frequencies: The list of the frequencies associated of bins.
        index: Index of the bin where the merge is done and store.
        round_frequencies: Boolean to indicate if the merged frequency is rounded.
            Default value is True.

    Returns:
        2-tuple containing:

        - A shape-(n,) containing the bins with the merged bins at index `i` and `i+1`.
        - A shape-(n,) containing the associated frequencies of bins.

    """
    bin_low = bins[index]
    frequency_low = frequencies[index]
    bin_high = bins[index + 1]
    frequency_high = frequencies[index + 1]
    bins = np.delete(bins, index + 1)
    frequencies = np.delete(frequencies, index + 1)

    frequency_merged = frequency_low + frequency_high
    bin_merged = (
        frequency_low * bin_low + frequency_high * bin_high
    ) / frequency_merged
    bins[index] = float(bin_merged)
    # Frequencies should be always integer.
    if round_frequencies:
        frequencies[index] = np.round(frequency_merged)
    else:
        frequencies[index] = frequency_merged

    return bins, frequencies


def find_min_bin(bins: np.ndarray) -> int:
    """Find minimal size bin of a given list of bins.

    Note: In case of equality, the bin is partly random selected, the last bins are
        more likely to be chosen than the first ones because of the sequential trial.

    Args:
        bins: List of bins where to find the smallest bin width.

    Return:
        The index of the smallest bin width.

    """
    min_i = 0
    min_size = bins[-1] - bins[0]
    min_count = 1
    for i, (bin_low, bin_high) in enumerate(zip(bins[:-1], bins[1:])):
        bin_size = bin_high - bin_low
        if bin_size < min_size:
            min_size = bin_size
            min_i = i
            min_count = 1
        elif bin_size == min_size:
            min_count += 1
            if random.random() <= (1.0 / min_count):
                min_i = i

    return min_i


def find_sorted_index(bins: np.ndarray, point: float) -> int:
    """Find index in the list of bins where should be inserted the given point.

    Note: This function performs a binary search to find the location.

    Args:
        bins: The list of bins to find the index.
        point: Point to search in the list of bins.

    Return:
        The best index for the input point to insert.

    """
    index = 0
    low_limit = 0
    high_limit = len(bins)
    while low_limit < high_limit:
        index = (low_limit + high_limit) // 2
        if bins[index] > point:
            high_limit = index
        elif bins[index] < point:
            index += 1
            low_limit = index
        else:
            break
    return index
