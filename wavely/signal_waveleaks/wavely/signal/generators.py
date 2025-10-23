from typing import Optional

import numpy as np
from numpy import random


def fill_nan(array: np.ndarray) -> np.ndarray:
    """Fill NaN in array by propagating last valid value.

    Args:
        array: An array with NaN values to fill.

    Returns:
        The filled array.

    """
    mask = np.isnan(array)
    idx = np.where(~mask, np.arange(mask.shape[1]), 0)
    np.maximum.accumulate(idx, axis=1, out=idx)
    out = array[np.arange(idx.shape[0])[:, None], idx]
    return out


def pink_noise(nrows: int, ncols: Optional[int] = 16) -> np.ndarray:
    """Generate pink noise using the Voss-McCartney algorithm.

    More information about the Voss-McCartney algorithm can be found
    in: http://www.firstpr.com.au/dsp/pink-noise/#Voss-McCartney.

    Args:
        nrows (int): number of values to generate.
        ncols (int): number of random sources to add.

    Returns:
        np.ndarray: the generated pink noise.

    Examples:
    >>> nrows, ncols = 10, 1
    >>> random.seed(123)
    >>> pn = pink_noise(nrows, ncols)
    >>> pn
    array([0.29371405, 0.42583029, 0.89338916, 0.43370117, 0.43086276,
           0.31226122, 0.42635131, 0.63097612, 0.39211752, 0.09210494])

    """
    array = np.empty((nrows, ncols))
    array.fill(np.nan)
    array[0, :] = random.random(ncols)
    array[:, 0] = random.random(nrows)

    # the total number of changes is nrows
    n = nrows
    cols = random.geometric(0.5, n)
    cols[cols >= ncols] = 0
    rows = random.randint(nrows, size=n)
    array[rows, cols] = random.random(n)

    array = fill_nan(array.T).T.sum(1)

    return array
