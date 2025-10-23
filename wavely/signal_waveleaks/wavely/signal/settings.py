import os

from dynaconf import LazySettings

settings = LazySettings(
    ROOT_PATH_FOR_DYNACONF=os.path.dirname(os.path.realpath(__file__)),
    ENVVAR_FOR_DYNACONF="WAVELY_SIGNAL_SETTINGS",
)
