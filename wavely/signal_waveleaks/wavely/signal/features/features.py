import inspect
from collections import defaultdict
from typing import Callable, Dict, List, Optional, Set

import numpy as np

from wavely.signal.models import FilterOutput
from wavely.signal.preprocessing.preprocessing import band_indexes, mel
from wavely.signal.units.helpers import next_power_of_two


class FeaturesComputer:
    ###################
    # CLASS VARIABLES #
    ###################
    _exported_features = set()
    _features_dimensions = {}
    _features_function = {}
    _tag_features = defaultdict(list)

    default_init_kwargs = {
        "n_fft": None,
        "window": np.hanning,
        "band_freq": 5e3,
        "features": ["spl"],
        "n_mels": 128,
        "order": 5,
        "filter_output": FilterOutput.sos,
        "Wn_gla": np.array([1e3, 10e3]),
        "Wn_glv": np.array([10, 1e3]),
        "tags": None,
        "dims": None,
    }

    def features(self):  # pragma: no cover
        return self._exported_features

    def allfeatures(self):  # pragma: no cover
        return self._features_function.keys()

    @property
    def exported_features(self) -> Set[str]:
        return self._exported_features

    @property
    def features_dimensions(self) -> Dict[str, List[str]]:
        return self._features_dimensions

    @property
    def features_function(self) -> Dict[str, Callable]:
        return self._features_function

    ##############
    # DECORATORS #
    ##############
    # feature function decorator
    @classmethod
    def globalfeature(
        cls, func, tags: Optional[List[str]] = None, dims: Optional[List[str]] = None
    ):
        name = func.__name__
        # bind feature name to function
        cls._features_function[name] = func
        # set features value to be exported
        cls._exported_features.add(name)

        if dims:
            # set the dimensions of the feature
            cls._features_dimensions[name] = dims
            # add the tag corresponding to the dimension of the feature
            cls._tag_features["{}d".format(len(dims))].append(name)

        if tags is None:
            tags = []
        for tag in tags:
            cls._tag_features[tag].append(name)
        return func

    # do not export feature
    @classmethod
    def globalnoexport(cls, func):
        name = func.__name__
        cls._exported_features.remove(name)
        return func

    # same function but work only on one object and the class
    def feature(self, func):
        # copy to avoid interfering with other objects or class
        self._features_function = self._features_function.copy()
        self._exported_features = self._exported_features.copy()

        name = func.__name__
        # bind feature name to function
        self._features_function[name] = func
        # set features value to be exported
        self._exported_features.add(name)
        return func

    def noexport(self, func):  # pragma: no cover
        name = func.__name__
        self._exported_features = self._exported_features.copy().remove(name)
        return func

    ##########################################

    def __init__(self, block_size: int, rate: int, **kwargs):
        """Initialize the feature extractor.

        Args:
            block_size: Size of the signal blocks.
            rate: Signal sample rate.
            n_fft: size of the FFT, see `numpy.fft.rfft`. If ``n_fft < n``, the
                blocks will be truncated, if ``n_fft > n``, the blocks will be
                zero-padded. If not specified, defaults to ``n``.
            window: A function to generate the window to be applied before
                computing the FFT. If not specified, defaults to ``numpy.hanning``.
            band_freq: Specifies frequencies for per-band
                computations.
                If it is a float, it defines the bandwidth of bands. If the value is
                    larger than the Nyquist frequency, band_freq is forced to be equal
                    to rate / 4.
                If it is an array, it defines the upper frequencies of bands.
                If it is a str, it should be either 'octave' or 'third' which will
                    generate octave or third octave bands. Defaults to 5 kHz.
            features: The features to be computed. This must be a subset
                of the features present in `FeaturesComputer._exported_features`. If not
                specified, defaults to ``spl`` only. If None, all features will be
                computed including transforms.
            n_mels: number of Mel coefficients.
            order: Filter order. Default value is 5.
            filter_output: A FilterOutput type.
            Wn_gla: A scalar or length-2 sequence giving the critical
                frequencies for computation of global acceleration level. Default
                values are [1kHz - 10kHz] for GLA.
            Wn_glv: A scalar or length-2 sequence giving the critical
                frequencies for computation of global velocity level. Default values
                are [10z - 1kHz] for GLV.
            tags: Select features with associated tags. This filters
                features selected with the `features` argument. In order to compute all
                the tagged features, argument `features` has to be set to "all".
            dims: Select only features that have the specified dimension.
                This list should be like ["1d", "2d"].

        """
        updated_kwargs = self.default_init_kwargs.copy()
        updated_kwargs.update(kwargs)
        tags = updated_kwargs.get("tags")
        if tags:
            self._filter_tagged_features(tags)

        dims = updated_kwargs.get("dims")
        if dims:
            self._filter_tagged_features(dims)

        n_fft = updated_kwargs.get("n_fft")
        band_freq = updated_kwargs.get("band_freq")
        if isinstance(band_freq, float) and band_freq > rate / 2:
            band_freq = rate / 4
        n_fft = n_fft or next_power_of_two(block_size)
        band, band_ix = band_indexes(rate, n_fft, band_freq)
        if (band_ix[0] == 0) or any(np.diff(band_ix) == 0):
            raise ValueError(
                "Input parameters lead to empty frequency bands. For custom frequency \
                bands, make sure only bands upper bounds are input. Else, consider \
                increasing 'n_fft' so as to reach a sufficient frequency resolution."
            )

        # pre-compute window
        window = updated_kwargs.get("window")
        window = window(block_size)

        # scaling factor
        # different window should result in the same energy
        window[:] /= np.sqrt(np.square(window).mean())

        features = updated_kwargs.get("features")
        if features and all(x in self._exported_features for x in features):
            self._exported_features = set(features)
        elif features == "all":
            pass
        else:
            raise ValueError(
                "{} is not a valid 'features' parameter. Allowed values are: {} or \
                'all'".format(
                    features, ", ".join(list(self._exported_features))
                )
            )

        # parameters for feature computation
        self.params = updated_kwargs
        self.params.update(
            {
                "rate": rate,
                "n_fft": n_fft,
                "window": window,
                "band_freq": band,
                "band_index": band_ix,
                "mel_basis": mel(rate, n_fft, updated_kwargs.get("n_mels")),
            }
        )
        self.feats = {}

    def _filter_tagged_features(self, tags: List[str]):
        """Keep only exported features tagged with at least one of the given tags.

        Args:
            tags: List of tags to keep.

        """
        if any(tag not in self._tag_features.keys() for tag in tags):
            raise ValueError(
                "Invalid 'tags' parameter. Allowed values are: {}".format(
                    ", ".join(self._tag_features.keys())
                )
            )
        tagged_features = set().union(*[self._tag_features[tag] for tag in tags])
        self._exported_features = set(self._exported_features).intersection(
            tagged_features
        )

    def compute(
        self,
        blocks: np.ndarray,
        flatten: bool = False,
        background_spectrum: Optional[np.ndarray] = None,
    ) -> dict:
        """Compute the features on signal blocks.

        Args:
            blocks: A shape-(n,k) signal blocks, where `n` is the number
                of blocks and `k` is the window size.
            flatten: flatten the features before export or not.
            background_spectrum: A shape-(k,) array containing the background
                spectrum, where `k` is the window size. If passed, use it to compute
                the spectral emergence.

        Returns:
            The computed features.

        """
        assert (
            len(blocks.shape) == 2
        ), "Blocks must have shape (M, N), with M the number of blocks, and N the \
            number of samples per block. "

        # Get the last computed periodogram
        if "periodogram" in self.feats:
            prev_periodogram = self.feats["periodogram"][[-1], :]
        else:
            prev_periodogram = np.zeros((1, self.params["n_fft"] // 2 + 1))

        self.feats = self.params.copy()
        self.feats["blocks"] = blocks
        self.feats["prev_periodogram"] = prev_periodogram
        self.feats["background_spectrum"] = background_spectrum

        # get every features to be exported
        _exported_features = {
            feat: self.getfeature(feat) for feat in self._exported_features
        }

        if flatten:
            _exported_features = self.flatten(_exported_features)

        return _exported_features

    def getfeature(self, name):
        if name not in self.feats:
            func = self._features_function[name]
            # compute every input features
            argnames = inspect.getfullargspec(func)[0]
            self.feats[name] = func(**{arg: self.getfeature(arg) for arg in argnames})
        return self.feats.get(name)

    def __getitem__(self, name):
        return self.getfeature(name)

    def __getattr__(self, name):
        return self.getfeature(name)

    def flatten(self, features):
        """Flatten the features before export.

        This is necessary if the features are to be exported in csv or loaded into a
        pandas dataframe. Any feature mapped to an array of more than one dimension,
        such as ``{"some_feature": [[1.], [2.], ...]}``, is flattened into
        ``{"some_feature": [1]., "bandleq2": [2.], ...}``

        .. note:: This currently only support features returning 1D or 2D arrays.

        Args:
            features: A dictionary of features.

        Returns:
            A dictionary of features.

        """
        output = {}
        for key, value in features.items():
            if isinstance(value, np.ndarray) and len(value.shape) == 1:  # 1d array
                output[key] = value
            elif isinstance(value, np.ndarray) and len(value.shape) == 2:  # 2d array
                for i, v in enumerate(value.T):
                    output["{}{}".format(key, i)] = v
            else:
                raise NotImplementedError(
                    "Flatten cannot handle value of type {}".format(type(value))
                )

        return output


def feature(*args, **kwargs):
    """Register a function to exported features.

    One or more tag (str) can be passed to this decorator in order to associate the
    registered function to these tags.

    """
    if len(args) == 1 and callable(args[0]):
        return FeaturesComputer.globalfeature(func=args[0])
    else:

        dims = kwargs["dims"]

        def tagged_feature(func):
            return FeaturesComputer.globalfeature(func, tags=args, dims=dims)

        return tagged_feature


def noexport(func):
    return FeaturesComputer.globalnoexport(func)
