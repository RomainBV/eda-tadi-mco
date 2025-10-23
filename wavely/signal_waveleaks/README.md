# wavely.signal

A python library for signal processing based on the numpy/scipy ecosystem.

## Content

* `wavely/signal/`: python source code
* `docs/`: Sphinx documentation
* `tests/`: Unit tests
* `notebooks/` Jupyter notebook containing code examples

## Setup the development environment

In order to setup the development environment, create a python virtual environment
with `python3 -venv` and install the requirements with `pip install -e .\[dev\]`.

## Setup dvc

In order to have access to remote data using `dvc`, you have to add these following lines to
the file `~/.config/dvc/config`:

```
['remote "s3_git_data_registry_signal"']
    url = s3://git-data-registry/signal
    endpointurl = https://data1.wavely.fr  # for instance
```

The first line declares a remote storage for which the name corresponds to the one in the `core` section of `.dvc/config`.
The url corresponds to the bucket in the remote storage object S3 compatible.
The endpointurl corresponds to the remote storage object S3 compatible.

If you want to run the notebooks, you have to pull some data from the remote storage on MinIO
using `dvc` by running `dvc pull data.dvc`.

## Testing

There is a tox configuration that runs the tests, along with code audit
using pylama and isort. To run tox simply use `tox`.

To run the tests using python3.5 you need to install the following packages:

```shell
sudo apt-get install libhdf5-serial-dev libffi-dev
```

## Build the doc

```shell
cd docs/
make html
```

The doc can then be found under `docs/_build/html`.

## Make a package release

```shell
bump2version --current-version <cur-ver> --new-version <new-ver> minor
git push
git push origin --tags
```

## Dimensions of the features

Dimensions were introduced in version 0.4.0.
When creating a feature with the `@feature` decorator, one can precise
the dimensions of the feature like:
```python3
@feature(dims=["time", "frequency_bands"])
```

A feature dimension can be found in the `_features_dimensions` dict, which is an
attribute of `FeaturesComputer`.

## Tagged features

Tags were introduced in version 0.4.0.
They are useful when one needs only one type of feature to be computed.
You can compute specific tagged features by passing a list of tags to the
`FeaturesComputer`, like:
```python3
FeaturesComputer(
    n=self.N,
    rate=self.rate,
    nfft=self.nfft,
    window=np.ones,
    features="all",
    tags=["1d", "vibration"],
)
```
The code above will compute features tagged "1d" AND features tagged "vibration".

Currently, used tags are:

* `2d`, `1d` (automatically assigned based on the dims argument.)
* `spectral`
* `acoustic`
* `temporal`
* `vibration`

Features can have an unlimited amount of tags.
Features are tagged using the decorator `@feature` like:
```python3
@feature("tag1", "tag2", "tag3")
```

Dimensions of the feature can also be added after the tags, like:

```python3
@feature("tag1", "tag2", "tag3", dims=["dim1", "dim2"])
```
This feature will also automatically be tagged `2d` since `len(dims)` is 2.

## The `dims` argument

You can choose if you want only `1d`, or only `2d` features by passing a list of
str to the `dims` argument of `FeaturesComputer` like:

```python3
FeaturesComputer(
    n=self.N,
    rate=self.rate,
    nfft=self.nfft,
    window=np.ones,
    features="all",
    tags=["vibration"],
    dims=["1d"]
)
```
The code above will compute `1d` features that are tagged '"vibration".

## Add new parameters in settings

The `signal` package comes with a bunch of parameters available in `settings.yaml`. You can have access to them by using the following commands:

```bash
>>> from wavely.signal.settings import settings
>>> # Get microphone names
>>> print(settings.MICROPHONES.keys())
dict_keys(['KNOWLES_SPH0645LM4H', 'KNOWLES_SPH6611LR5H', 'VESPER_VM1000', 'AVISOFT_40008', 'BK_4939', 'DODOTRONIC_Ultramic384k', 'MICW_i436', 'Digital', 'Builtin', 'DODOTRONIC_Ultramic192k', 'KNOWLES_SPH0644LM4H_1', 'BEYERDYNAMIC_MM1', 'BEHRINGER_ECM8000'])
```

If you want to add new microphones in the settings, you can create a `settings.local.yaml` in the root directory of your package and add a new entry corresponding to the new microphone. For instance, if the `settings.yaml` of the `signal` package contains:

```yaml
default:
  microphones:
    BEHRINGER_ECM8000:
      name: "behringer ECM8000"
      sensitivity: -70
      AOP: None
      SNR: None
```

Your `settings.local.yaml` will looks like:

```yaml
default:
  microphones:
    BEHRINGER_ECM8000:
      name: "behringer ECM8000"
      sensitivity: -70
      AOP: None
      SNR: None
    NEW_MIC:
      name: "a cool microphone"
      sensitivity: 0
      AOP: None
      SNR: None
```

The location of the local settings file can be set using environment variable by exporting the value od `SETTINGS_FILE_FOR_DYNACONF`:

```shell
export SETTINGS_FILE_FOR_DYNACONF=<path/to/file>
```

You also have to create a `.env` file which contains the location of the local settings file:

```
SETTINGS_FILE_FOR_DYNACONF=<path/to/file>
```
