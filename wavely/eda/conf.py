import os
from pathlib import Path
from typing import Dict, List, Optional,Union

import yaml
from pydantic import BaseModel, BaseSettings, Field

# -----------------------------
# Classes utilitaires
# -----------------------------
class SensorIDDates(BaseModel):
    sensor_id: Optional[str] = None
    dates: List[str] = Field(default_factory=list)
    recording_type: Optional[str] = None
    calibrate: Optional[bool] = True
    time_shift_ms: Optional[float] = None
    device: Optional[str] = None
    rss_calibration: Optional[float] = None
    normalise_gain: Optional[float] = None
    device_name: Optional[str] = None
    channel_prefix: Optional[str] = None
    microphone: Optional[List[str]] = None
    preamp_gain: Optional[List[float]] = None
    mic_calibration: Optional[List[float]] = None


class ExpertModelConfig(BaseModel):
    enabled: bool = True
    min_limit_mean: float = 0.2
    max_limit_std: float = 0.05
    max_limit_spectralflatness: Optional[float] = None
    n_blocks_integration: int = 10
    ultrasound_threshold_db: float = 15.0

class LeakModels(BaseModel):
    expert_model: ExpertModelConfig = ExpertModelConfig()
    ml_model: bool = False

class BaseSensorConfig(BaseModel):
    sensor_ids: List[SensorIDDates] = []

class DataSelectionConfig(BaseModel):
    MAGNETO: Optional[BaseSensorConfig] = None
    ZOOM_F4: Optional[BaseSensorConfig] = None
    OTHER: Optional[BaseSensorConfig] = None

class DataAcquisitionSettings(BaseModel):
    resampling_rate: int = None
    timezone: str = 'Europe/Paris'

class FeaturesComputerKwargs(BaseModel):
    features: List[str]
    band_freq: Union[float, str]

class FeatureSettings(BaseModel):
    block_duration: float = 0.04
    block_overlap: float = 0.0
    aggregation_sliding_window: float = 1.0
    features_computer_kwargs: FeaturesComputerKwargs
    leak_models: LeakModels = LeakModels()

class LabelSettings(BaseModel):
    label_replacements: Dict[str, str] = {}
    label_removals: List[str] = []
    label_background: List[str] = []

class VisualizationSettings(BaseModel):
    frequency_scale: str = 'log'
    base_plot_height: int = 300
    features_count_per_plot: int = 4
    primary_labels: List[str] = []

# -----------------------------
# Fonction pour charger le YAML
# -----------------------------
def yaml_config_settings_source(settings):
    source_file = Path(os.environ.get("EDA_SETTINGS", "wavely/eda/settings.yaml"))
    if source_file.is_file():
        return yaml.safe_load(source_file.read_text("utf-8")) or {}
    return {}

# -----------------------------
# Settings principal (Pydantic v1)
# -----------------------------
class Settings(BaseSettings):
    log_level: str = "INFO"
    campaign: str
    project: str
    source_db: str
    audacity_label_name: str
    root_dir: Optional[Path] = None
    raw_signals_dir: Optional[Path] = None
    datasets_dir: Optional[Path] = None
    max_long_format_duration: Optional[float] = None
    apply_filter_config: Optional[dict] = None
    data_acquisition: Optional[DataAcquisitionSettings] = None
    features: Optional[FeatureSettings] = None
    labels: Optional[LabelSettings] = None
    data_selection: DataSelectionConfig
    visualization: Optional[VisualizationSettings] = None

    class Config:
        @classmethod
        def customise_sources(cls, init_settings, env_settings, file_secret_settings):
            return (
                init_settings,
                env_settings,
                yaml_config_settings_source,
                file_secret_settings
            )

# -----------------------------
# Instanciation
# -----------------------------
settings = Settings()
