import math
import warnings
from typing import Callable, Optional, Union

import numpy as np


def compute_overlap_size_from_ratio(
    overlap_ratio: float, block_size: int, round_fn: Callable[[float], int] = math.floor
) -> int:
    """Compute the overlap size from the overlap ratio.

    Args:
        overlap_ratio: The overlap ratio.
        block_size: The block size.
        round_fn: The function used to round the overlap size.

    Returns:
        The overlap size.

    """
    return round_fn(block_size * overlap_ratio)


def split_signal(
    signal: np.ndarray,
    rate: float,
    block_duration: float,
    overlap_ratio: float = 0,
    memory_efficiency: bool = True,
    zero_pad: bool = False,
    round_fn: Callable[[float], int] = math.floor,
) -> np.ndarray:
    """Split a raw signal into blocks.

    Args:
        signal: A shape-(n,) array signal to split, where n is the signal length.
        rate: Sampling rate.
        block_duration: Block duration in seconds.
        overlap_ratio: Overlapping ratio (in [0,1)).
        memory_efficiency: If set to True, an efficient implementation is used to
            split the signal but the returned array will always be read only.
            Otherwise, it will be writable if the original array was, the downside
            here is the implementation is less efficient.
        zero_pad: When the signal length does not match the total length of the
            overlapped blocks and this boolean is set to True, then the signal
            is zero-padded. Otherwise, the signal is cropped.
        round_fn: The function used to round the overlap size.

    Returns:
        A shape-(n, k) array which holds the new sliced signal, where n is the number
            of blocks and k the block size.


    """
    return SignalSplitter(
        rate=rate,
        block_duration=block_duration,
        overlap_ratio=overlap_ratio,
        memory_efficiency=memory_efficiency,
        zero_pad=zero_pad,
        round_fn=round_fn,
    ).fit_transform(signal)


def next_power_of_two(x: int) -> int:
    """Compute the next power of 2 of a given integer.

    Args:
        x: the input integer.

    Returns:
        the next power of 2 of x.

    """
    return int(2 ** np.ceil(np.log2(np.abs(x))))


def compute_hysteresis(
    signal: np.ndarray,
    threshold: Union[float, np.ndarray],
    hysteresis: float,
    upward: bool = True,
) -> np.array:
    """Compute hysteresis of input numpy array.

    Arg:
        signal: input vector to compute hysteresis.
        threshold: Threshold value to compare to input signal.
        hysteresis: Hysteresis value to add (or substract) to threshold
            to compare to input signal.
        upward: Boolean used to choose if the hysteresis is used with ascending
            threshold (default) or descending threshold.

    Returns:
        A shape-(n,) array of zeros and ones contanining the information about
        hysteresis condition. If upward is True, the signal needs to cross the
        threshold_high and be above to return 1. Then the signal needs to cross the
        threshold_low and be below to return 0. If upward is False, the signal needs
        to cross the threshold_low and be below to return 1. Then the signal needs
        to cross the threshold_high and be above to return 0.

    """
    # Define the position of input signal compare to threshold.
    if upward:
        above = signal >= threshold
        threshold_low = threshold - hysteresis
        outside_element = (signal <= threshold_low) | above
    else:
        below = signal <= threshold
        threshold_high = threshold + hysteresis
        outside_element = (signal >= threshold_high) | below

    index = np.nonzero(outside_element)[0]  # Index for elements outside of hysteresis.
    if not index.size:  # prevent index error if index vector is empty
        return np.zeros_like(signal, dtype=int)

    count = np.cumsum(outside_element)
    if upward:
        return np.where(count, above[index[count - 1]], False).clip(0, 1)

    return np.where(count, below[index[count - 1]], False).clip(0, 1)


def get_remaining_signal(
    signal: np.ndarray,
    blocks: np.ndarray,
    overlap_ratio: float,
    round_fn: Callable[[float], int] = math.floor,
) -> Optional[np.ndarray]:
    """Get the remaining signal from a split signal.

    Args:
        signal: A shape-(n,) array signal to split, where n is the signal length.
        blocks: A shape-(n,k) array containing the split signal.
        overlap_ratio: Overlapping ratio (in [0,1)).
        round_fn: The function used to round the overlap size.

    Returns:
        A shape-(n,) array containing the remaining signal.

    """
    n_blocks, block_size = blocks.shape
    overlap = compute_overlap_size_from_ratio(
        overlap_ratio=overlap_ratio, block_size=block_size, round_fn=round_fn
    )
    remaining_signal_size = signal.size - (
        n_blocks * block_size - (n_blocks - 1) * overlap
    )
    if remaining_signal_size == 0:
        return None
    else:
        return signal[-remaining_signal_size:]


def add_and_merge_blocks(
    blocks: np.ndarray,
    overlap_ratio: float = 0,
    remaining_signal: Optional[np.ndarray] = None,
    round_fn: Callable[[float], int] = math.floor,
) -> np.ndarray:
    """Add and merge blocks into a signal.

    Args:
        blocks: A shape-(n,k) array blocks of signal, where k is the block_size.
        overlap_ratio: Overlapping ratio (in [0,1)).
        remaining_signal: Last part of a signal if cropped by `split_signal`.
        round_fn: The function used to round the overlap size.

    Returns:
        A shape-(n,) array containing the reconstructed signal.

    """

    def slice_length(slice_: slice):
        return slice_.stop - slice_.start

    if remaining_signal is None:
        remaining_signal = 0
        remaining_signal_size = 0
    else:
        remaining_signal_size = len(remaining_signal)

    n_blocks, block_size = blocks.shape
    block_overlap = compute_overlap_size_from_ratio(
        overlap_ratio=overlap_ratio, block_size=block_size, round_fn=round_fn
    )
    block_step = block_size - block_overlap

    signal_size = n_blocks * block_step + block_overlap + remaining_signal_size

    signal = np.zeros(signal_size)
    signal[:block_size] = blocks[0, :]
    for i in range(1, n_blocks):
        fill_slice = slice(
            i * block_step, min(i * block_step + block_size, signal_size)
        )
        signal[fill_slice] += blocks[i, : slice_length(fill_slice)]
    signal[-remaining_signal_size:] += remaining_signal

    return signal


def unsplit_signal(
    blocks: np.ndarray,
    overlap_ratio: float,
    remaining_signal: Optional[np.ndarray] = None,
    round_fn: Callable[[float], int] = math.floor,
) -> np.ndarray:
    """Unsplit blocks into a signal.

    Args:
        blocks: A shape-(n,k) array containing the blocks to unsplit, where
            n is the number of blocks and k is the block size.
        overlap_ratio: The overlap ratio.
        remaining_signal: The last part of the signal, discarded by
            `split_signal` in case the length of the signal is not a
            multiple of the step.
        round_fn: The function used to round the overlap size.

    Returns:
        A shape-(n,) array containing the signal.

    """
    block_size = blocks.shape[-1]
    block_overlap = compute_overlap_size_from_ratio(
        overlap_ratio=overlap_ratio, block_size=block_size, round_fn=round_fn
    )
    block_step = block_size - block_overlap

    if overlap_ratio == 0:
        blocks_signal = blocks
    else:
        blocks_signal = blocks[:, :block_step]

    signal = np.hstack([np.ravel(blocks_signal, order="C"), blocks[-1, block_step:]])
    if remaining_signal is None:
        return signal
    else:
        return np.hstack([signal, remaining_signal])


class SignalSplitter:
    def __init__(
        self,
        rate: float,
        block_duration: float,
        overlap_ratio: float = 0.0,
        memory_efficiency: bool = True,
        zero_pad: bool = False,
        round_fn: Callable[[float], int] = math.floor,
    ):
        """Instantiate a SignalSplitter object.

        Args:
            rate: Sampling rate.
            block_duration: Block duration in seconds.
            overlap_ratio: Overlapping ratio (in [0,1)).
            memory_efficiency: If set to True, an efficient implementation is used to
                split the signal but the returned array will always be read only.
                Otherwise, it will be writable if the original array was, the downside
                here is the implementation is less efficient.
            zero_pad: When the signal length does not match the total length of the
                overlapped blocks and this boolean is set to True, then the signal
                is zero-padded. Otherwise, the signal is cropped.
            round_fn: The function used to round the overlap size.

        """
        if not (0 <= overlap_ratio < 1):
            raise ValueError("Overlapping ratio must be in [0,1).")
        if block_duration == 0:
            raise ValueError("block duration too small.")

        self.rate = rate
        self.block_duration = block_duration
        self.overlap_ratio = overlap_ratio
        self.memory_efficiency = memory_efficiency
        self.zero_pad = zero_pad
        self.round_fn = round_fn

    def _raise_on_unfitted(self):
        for attr in ["n_blocks_", "shape_", "block_size_", "step_"]:
            if attr not in self.__dict__:
                raise ValueError("SignalSplitter has not been fitted yet.")

    def fit(self, signal: np.ndarray):
        """Fit the blocks shape to split the signal.

        Args:
            signal: A shape-(n,) array signal to split, where n is the signal length.

        """
        self.block_size_ = int(self.block_duration * self.rate)

        overlap = compute_overlap_size_from_ratio(
            overlap_ratio=self.overlap_ratio,
            block_size=self.block_size_,
            round_fn=self.round_fn,
        )
        self.step_ = self.block_size_ - overlap

        n_blocks = (signal.shape[-1] - overlap) / self.step_
        if not (n_blocks.is_integer() or self.zero_pad):
            warnings.warn(
                "Since the overlapped blocks total length does not match the "
                "length of the signal, `split_signal` will crop the final "
                "element of the new sliced signal."
            )
        if self.zero_pad:
            n_blocks = math.ceil(n_blocks)
            # Zero pad the signal to have a multiple of the step
            self.zero_pad_length_ = (n_blocks * self.step_) - (
                signal.shape[-1] - overlap
            )
        self.n_blocks_ = int(n_blocks)

        self.shape_ = (self.n_blocks_, self.block_size_)

        if self.memory_efficiency is True:
            self.strides_ = (self.step_ * signal.strides[-1], signal.strides[-1])

        self.remaining_signal_ = get_remaining_signal(
            signal, np.empty(self.shape_), self.overlap_ratio, self.round_fn
        )

    def transform(self, signal: np.ndarray) -> np.ndarray:
        """Split a raw signal into blocks.

        Args:
            signal: A shape-(n,) array signal to split, where n is the signal length.

        Returns:
            A shape-(n, k) array which holds the new sliced signal, where n is the
            number of blocks and k the block size.

        """
        self._raise_on_unfitted()
        signal_to_split = signal
        if self.zero_pad:
            signal_to_split = np.concatenate(
                (signal_to_split, np.zeros(self.zero_pad_length_))
            )

        if self.memory_efficiency:
            blocks = np.lib.stride_tricks.as_strided(
                signal_to_split,
                shape=self.shape_,
                strides=self.strides_,
                writeable=False,
            )
        else:
            blocks = [
                signal_to_split[
                    slice(i * self.step_, i * self.step_ + self.block_size_)
                ]
                for i in range(self.n_blocks_)
            ]
            blocks = np.array(blocks[: self.shape_[0]]).reshape(self.shape_)

        return blocks

    def fit_transform(self, signal: np.ndarray) -> np.ndarray:
        """Fit and transform the signal.

        Args:
            signal: A shape-(n,) array containing the signal to normalize.

        Returns:
            A shape-(n, k) array which holds the new sliced signal, where n is the
            number of blocks and k the block size.

        """
        self.fit(signal)
        return self.transform(signal)

    def inverse_transform(self, blocks: np.ndarray) -> np.ndarray:
        """Unsplit blocks into a signal.

        Args:
            blocks: A shape-(n,k) array containing the blocks to unsplit, where
                n is the number of blocks and k is the block size.

        Returns:
            A shape-(n,) array containing the signal.

        """
        self._raise_on_unfitted()
        if blocks.shape[-1] != self.block_size_:
            raise ValueError(
                "The blocks must have split using a block size of {}".format(
                    self.block_size_
                )
            )
        if self.overlap_ratio == 0:
            blocks_signal = blocks
        else:
            blocks_signal = blocks[:, : self.step_]

        signal = np.hstack(
            [np.ravel(blocks_signal, order="C"), blocks[-1, self.step_ :]]
        )
        if self.remaining_signal_ is not None:
            signal = np.hstack([signal, self.remaining_signal_])

        return signal
