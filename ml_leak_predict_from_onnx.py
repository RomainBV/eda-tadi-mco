#%% 
# ==========================
# Imports standard Python
# ==========================
import os
import glob
from pathlib import Path
import logging
import datetime
import warnings
import pytz
import pandas as pd
import re 
from tqdm import tqdm
import numpy as np

# ==========================
# Imports tiers
# ==========================
import streamlit as st
from streamlit_jupyter import StreamlitPatcher
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
import argparse
import soundfile as sf
from typing import Optional, Dict, List, Tuple
from sqlalchemy import create_engine, and_

# ==========================
# Imports du projet / locaux
# ==========================
from wavely.eda.conf import settings
# from wavely.eda import utils
from wavely.edge_metadata.models import Recording
from collections import defaultdict
from wavely.waveleaks.deployment import onnx
from wavely.ml.features import extract_bandleq
from wavely.signal.units.helpers import SignalSplitter
from wavely.signal.features.features import FeaturesComputer

# Ignorer les warnings
warnings.filterwarnings('ignore')

# Logger principal
logger = logging.getLogger(__name__)
logging.getLogger("pydub.converter").setLevel(logging.WARNING)
logging.basicConfig(level=logging.INFO)
logging.basicConfig(level=logging.DEBUG)

classifier = onnx.OnnxLevelOneClassifier(preprocessing_means_gain=50)

# ==========================
# Initialisations
# ==========================

SAMPLE_RATE = 192000
BLOCK_DURATION = 0.04 

DEBUG = False

BAND_FREQ = np.array([
    2000,
    3000,
    4000,
    5000,
    6000,
    7000,
    8000,
    9000,
    10000,
    12000,
    14000,
    15000,
    15500,
    16000,
    16500,
    17000,
    17500,
    18000,
    18500,
    19000,
    19500,
    20000,
    20500,
    21000,
    21500,
    22000,
    22500,
    23000,
    23500,
    24000,
    24500,
    25000,
    25500,
    26000,
    26500,
    27000,
    27500,
    28000,
    28500,
    29000,
    29500,
    30000,
    30500,
    31000,
    31500,
    32000,
    32500,
    33000,
    33500,
    34000,
    34500,
    35000,
    35500,
    36000,
    36500,
    37000,
    37500,
    38000,
    38500,
    39000,
    39500,
    40000,
    45000,
    50000,
    55000,
    60000,
    65000,
    70000,
    75000,
    80000,
    85000,
    90000,
    96000
])



# =====================
# ==== USER INPUTS ====
# =====================

def to_hdf_dataset(
    dataframes: Dict[str, pd.DataFrame],
    root_dirname: str,
    output_name: str = "features.h5",
):
    """Export a dataset in DataFrames to HDF5 file.

    Args:
        dataframes: A Dictionary containing DataFrames to be stored in HDF5 file.
            Keys of the dictionary are the keys of the HDF5 file. Values are the
            corresponding DataFrames.
        root_dirname: The directory path of the HDF file to be stored in.
        output_name: The desired name of the HDF file. Must have the extension ".h5".
            Defaults to "features_csv.h5"

    """
    for key, dataframe in dataframes.items():
        dataframe.to_hdf(
            os.path.join(root_dirname, output_name),
            key=key,
            complevel=0,
            format="table",
            append=False,
        )

def read_hdf_dataset(
    hdf_path: str, hdf_keys: List[str] = None
) -> Dict[str, pd.DataFrame]:
    """ Load a dataset stored in a HDF file.

    Args:
        hdf_path: The path to the HDF file.
        hdf_keys: Identifiers when reading HDF content. Defaults to None where HDF
            file keys are automatically read using pandas.HDFStore.keys. Else return
            only objects corresponding to the entered list of keys.

    Returns:
        A Dictionary which values are the returned DataFrames stored in the HDF file
        corresponding to the entered keys.
    """

    file = pd.HDFStore(hdf_path)
    if hdf_keys is None:
        keys = file.keys()
    else:
        keys = hdf_keys
    dataframes = {os.path.basename(key): file.get(key) for key in keys}
    file.close()
    return dataframes

def predict_from_filepath(filepath,start_time,block_duration):

    signal_splitter = SignalSplitter(rate=SAMPLE_RATE, block_duration=block_duration)
    import scipy.signal as signal

    data, rate = sf.read(filepath, always_2d=True)

    if settings.apply_filter_config['apply_filter']:
        sos = signal.butter(settings.apply_filter_config['order'], settings.apply_filter_config['order'], btype=settings.apply_filter_config['apply_filter'][0], fs=rate, output='sos')
        data = signal.sosfiltfilt(sos, data, axis=0)


    features_computer = FeaturesComputer(
    block_size=int(block_duration*rate),
    rate=rate,
    features=["bandleq"],
    band_freq=BAND_FREQ
    )

    data = data.T
    bandleq = extract_bandleq(
        audio_signal=data,
        features_computer=features_computer,
        signal_splitter=signal_splitter
    )

    bandleq = bandleq[0][:,:,0]

    gain = 50 #rss_calibration = 12 / infineon_sensitivity = 38

    new_bandleq = bandleq + gain
    n_preds = bandleq.shape[0]//50
    for pred in range(n_preds):
        df = pd.DataFrame(new_bandleq[pred*50:(pred+1)*50], columns=[f"bandleq{i}" for i in range(73)])
        data_acquisition_settings = onnx.DataAcquisitionSettings(1, 0.04, 192000)
        event = onnx.ClassifierEvent(
            df,
            blocks=[],
            data_acquisition_settings=data_acquisition_settings
        )
        classifier.push(event)
    pred = classifier.pop_predict_proba()
    index = [start_time + datetime.timedelta(seconds=2*pred) for pred in range(n_preds)]
    pred.index = index
    pred.index.name = "time"

    return pred

def sort_key(path: str) -> Tuple[str, int]:
    """Generate a sorting key from a WAV file path.

    This function extracts a numeric sequence number from the file path 
    if it exists (e.g., `file_01.wav`). It returns a tuple containing:
    
    - The file path prefix without the sequence number.
    - The sequence number as an integer (or `0` if not found).

    Args:
        path (str): Path to the WAV file.

    Returns:
        Tuple[str, int]: A tuple where:
            - The first element is the path without the numeric suffix.
            - The second element is the extracted sequence number,
              or `0` if none is present.

    Example:
        >>> sort_key("recording_12.wav")
        ('recording', 12)

        >>> sort_key("session.wav")
        ('session.wav', 0)
    """
    match = re.search(r'_(\d+)\.wav$', path)
    if match:
        return (path.rsplit('_', 1)[0], int(match.group(1)))
    return (path, 0)

def query_recordings(
    metadata_path: str,
    start: Optional[datetime.datetime] = None,
    end: Optional[datetime.datetime] = None,
    sensor: str = "MAGNETO",
    selected_configuration: Optional[dict] = None,
    time_buffer: int = 0,
) -> Tuple[List[Recording], str, pd.DataFrame]:
    """
    Base function to query recordings from metadata DB with optional filters.

    Args:
        metadata_path (str): Path to metadata SQLite DB.
        start (datetime, optional): Start time.
        end (datetime, optional): End time.
        sensor (str): Sensor type.
        selected_configuration (dict, optional): Config dict.
        time_buffer (int): Minutes buffer to extend time range.

    Returns:
        Tuple[List[Recording], str, pd.DataFrame]:
            - results: list of Recording objects (not filtered for file existence yet)
            - db_url: SQLAlchemy DB URL
            - start_times: DataFrame with filenames as index and start_times
    """
    if not os.path.exists(metadata_path):
        raise FileNotFoundError(f"Non existing path: {metadata_path}")

    db_url = f"sqlite:///{metadata_path}"
    engine = create_engine(db_url)
    Session = sessionmaker(bind=engine)

    with Session() as session:
        # Log interval
        if start is None and end is None:
            logger.info("Getting all recordings")
        elif start is None:
            logger.info(f"Getting recordings up to {end}")
        elif end is None:
            logger.info(f"Getting recordings from {start}")
        else:
            logger.info(f"Getting recordings from {start} to {end}")

        query = session.query(Recording)

        # --- Time filters with buffer ---
        filters = []
        if start is not None:
            filters.append(Recording.start_time >= start - datetime.timedelta(minutes=time_buffer))
        if end is not None:
            filters.append(Recording.start_time <= end + datetime.timedelta(minutes=time_buffer))
        if filters:
            query = query.filter(and_(*filters))

        # --- Sensor filter ---
        if sensor == "MAGNETO" and selected_configuration:
            query = query.filter(Recording.recording_type == selected_configuration["recording_type"])

        results = query.all()

        if not results:
            logger.warning("No recordings found in specified range.")
            return [], db_url, pd.DataFrame(columns=["start_times"])

        # --- Build start_times DF ---
        filenames = [rec.filename for rec in results]
        times = [rec.start_time for rec in results]
        start_times = pd.DataFrame(data={"start_times": times}, index=filenames)

    return results, db_url, start_times

def group_recordings(
    metadata_path: str,
    start: datetime.datetime = None,
    end: datetime.datetime = None,
    sensor: str = "MAGNETO",
    selected_configuration: dict = None,
) -> Tuple[List, str, pd.DataFrame, Dict[str, List[str]]]:
    """
    Retrieve and group recordings from the SQLite metadata database.

    This function queries recordings between `start` and `end`, applies optional
    sensor/channel filters, checks that WAV files exist on disk, and groups 
    them by filename prefix.

    Args:
        metadata_path (str): Path to the SQLite metadata database (metadata.db).
        start (datetime, optional): Start time of the query interval.
        end (datetime, optional): End time of the query interval.
        sensor (str, optional): Sensor type (e.g., 'MAGNETO').
        selected_configuration (dict, optional): Configuration dict containing
            keys such as 'recording_type' and 'channel_prefixe'.

    Returns:
        Tuple[List[Recording], str, pd.DataFrame, Dict[str, List[str]]]:
            - metadata: List of valid Recording objects.
            - db_url: SQLAlchemy database connection string.
            - start_times: DataFrame mapping filenames to start times.
            - file_groups: Dictionary grouping filenames by prefix.

    Raises:
        FileNotFoundError: If `metadata_path` does not exist.
    """
    results, db_url, start_times = query_recordings(
        metadata_path, start, end, sensor, selected_configuration, time_buffer=3
    )

    if not results:
        return [], db_url, start_times, {}

    base_dir = os.path.dirname(metadata_path)
    metadata, valid_filenames = [], []

    for rec in results:
        wav_file_path = os.path.join(base_dir, rec.filename)
        if os.path.exists(wav_file_path):
            if sensor == "MAGNETO":
                metadata.append(rec)
                valid_filenames.append(rec.filename)
            else:
                channel_str = f"_Tr{selected_configuration.get('channel_prefixe', '')}"
                if channel_str in rec.filename:
                    metadata.append(rec)
                    valid_filenames.append(rec.filename)
        else:
            logger.debug(f"File not found: {wav_file_path}")

    start_times = start_times.loc[valid_filenames]

    # Group filenames by prefix
    file_groups = {}
    for filename in valid_filenames:
        prefix = filename.split("_")[0]
        file_groups.setdefault(prefix, []).append(filename)

    for key in file_groups:
        file_groups[key].sort()

    return metadata, db_url, start_times, file_groups

def config_to_dict(config):
    """Transforme ton DataSelectionConfig en dict exploitable pour UI."""
    data = defaultdict(dict)
    for sensor_name, base_cfg in config.__dict__.items():
        if base_cfg and hasattr(base_cfg, "sensor_ids"):
            for sid in base_cfg.sensor_ids:
                data[sensor_name][sid.sensor_id] = sid.dates
    return dict(data) 

def select_configuration(
    config,        
    sensor: str,
    sensor_id: str,
    date: str
) -> Optional[Dict]:
    """Return the full configuration for a given sensor and sensor_id on a specific date.

    Args:
        config: A DataSelectionConfig object containing sensor configurations.
        sensor: The name of the sensor to look up.
        sensor_id: The specific sensor_id to filter by.
        date: The date to filter the configuration.

    Returns:
        A dictionary containing the full configuration if a match is found, 
        otherwise None.
    """
    sensor_config = getattr(config, sensor, None)
    if not sensor_config:
        return None
    
    for sid in sensor_config.sensor_ids:
        if sensor_id and sid.sensor_id and sid.sensor_id != sensor_id:
            continue
        if date and sid.dates and date not in sid.dates:
            continue
        full_config = sid.dict()  
        parent_fields = {k: v for k, v in sensor_config.dict().items() if k != "sensor_ids"}
        full_config.update(parent_fields)
        return full_config
    
    return None



if DEBUG:
    start_time_utc_str = None
    end_time_utc_str = None
    sensor = "MAGNETO"
    sensor_id = "RCA-04"
    date = "2025-10-10"   
    channel = 1 
    config = {
        "data_selection": {
            "MAGNETO": {
                "sensor_ids": [
                    {
                        "sensor_id": "RCA-04",
                        "dates": ["2025-10-10"],
                        "recording_type": "continuous",
                        "calibrate": True,     
                        "time_shift_ms": 0.0,    
                    }
                ]
            }
        }
    }
else:
    StreamlitPatcher().jupyter()

    # --- Titre principal ---
    st.title("Processing : Machine-learning leak model")

    st.markdown(
        f"""
        ## About this page

    

        This page allows you to **compute the  leak detection machine-learning model**.  
        
        Make sure to activate the correct Kernel, for example:

        ```bash
        source <ml_leak_venv_path>/bin/activate
        ```

        **Note:** this Kernel is not compatible with the datasets-analysis package.

        If your goal is to compare leak detection results with acoustic indicators and BandLeq, it is recommended to first run the processing program using the dedicated Kernel:

        ```bash
        streamlit run processing.py
        ```
        Once all processing steps are completed, you can proceed with the machine-learning leak detection calculation:
        ```bash
        streamlit run ml_leak_predict_from_onnx.py
        ```
        Remember to activate the corresponding Kernel for this step as well.


        These computations rely on the settings provided in the `settings.yaml` file, including:

        ### Necessary information
        - **Feature settings**: `block_duration`, `aggregation_sliding_window`  

        ### Optional information
        - **Signal calibration**: `calibrate`  

        The process includes the following steps:

        - **Leak detection computation**  
        - **Interpolation on the aggregated window** defined inthe `settings.yaml`
        - **Export** in `.h5` format in a `results` subdirectory  

        ## User inputs
        """
    )


    # --- Texte explicatif ---
    data = config_to_dict(settings.data_selection)
    # ---- Étape 1 : choix du sensor ----
    sensor = st.selectbox("Choose a sensor :", list(data.keys()))

    # ---- Étape 2 : choix du sensor_id si disponible ----
    sensor_ids = list(data[sensor].keys())
    if any(sensor_ids) and not all(sid == '' for sid in sensor_ids):
        sensor_id = st.selectbox("Choose a sensor ID :", sensor_ids)
    else:
        sensor_id = ''
        st.write("No sensor ID availaible.")

    # ---- Étape 3 : choix de la date si disponible ----
    dates = data[sensor].get(sensor_id, [])
    if dates:
        date = st.selectbox("Choose a date :", dates)
    else:
        date = ''
        if sensor_id != '':
            st.write("No date available for this sensor ID.")
    

    # ---- Étape 4 : choix de la voie d'analyse ----
    channel = st.number_input("channel :", min_value=1, value=1, step=1)
    if channel is None:
        channel = 1

    st.write(f"Selected channel: {channel}")

    # Start time
    start_time_utc_str = st.text_input(
        "Start Time (UTC) – format: YYYY-MM-DD HH:MM:SS (edit only if necessary)", ""
    )

    # End time
    end_time_utc_str = st.text_input(
        "End Time (UTC) – format: YYYY-MM-DD HH:MM:SS (edit only if necessary)", ""
    )

# Conversion simple
start_time_utc = None if not start_time_utc_str or start_time_utc_str.lower() == "none" else start_time_utc_str
end_time_utc = None if not end_time_utc_str or end_time_utc_str.lower() == "none" else end_time_utc_str
selected_configuration = select_configuration(settings.data_selection, sensor, sensor_id, date)


if not DEBUG:
    # --- Conteneur pour afficher les logs ---
    log_container = st.empty()
    # --- Handler Streamlit ---
    class StreamlitHandler(logging.Handler):
        def __init__(self, container):
            super().__init__()
            self.container = container
            self.logs = []

        def emit(self, record):
            msg = self.format(record)
            self.logs.append(msg)
            self.container.text("\n".join(self.logs))

    # Formatter
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')

    # --- Logger principal ---
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)
    if logger.hasHandlers():
        logger.handlers.clear()

    streamlit_handler = StreamlitHandler(log_container)
    streamlit_handler.setFormatter(formatter)
    logger.addHandler(streamlit_handler)

    # --- Logger pour utils ---
    utils_logger = logging.getLogger("wavely.eda.utils")
    utils_logger.setLevel(logging.INFO)
    if utils_logger.hasHandlers():
        utils_logger.handlers.clear()
    utils_logger.addHandler(streamlit_handler)

    
def computation(settings,sensor,sensor_id,date,selected_configuration):

    raw_signals_dir = Path(settings.raw_signals_dir)
    # datasets_dir = Path(settings.datasets_dir)
    root_dir = settings.root_dir
    timezone = settings.data_acquisition.timezone
    aggregation_sliding_window = settings.features.aggregation_sliding_window
    os.chdir(root_dir)
    print(root_dir)
  
  
    data_path = os.path.join(raw_signals_dir,sensor,sensor_id,date)
    metadata_path = os.path.join(data_path,settings.source_db)

    check_existing_csv = glob.glob(os.path.join(os.path.join(data_path, "labels"), "*.csv"))

    use_existing_csv = False
    if check_existing_csv:
        csv_file = check_existing_csv[0] 
        if os.path.getsize(csv_file) > 0: 
            use_existing_csv = True

    if use_existing_csv:
        logger.info(f"Using existing labels CSV: {csv_file}")
        df_labels = pd.read_csv(csv_file)
    else:
        df_labels = None


    time_ranges = [(start_time_utc, end_time_utc)]


    for start_time, end_time in time_ranges:
        if start_time:
            if not date == '':
                start = datetime.datetime.strptime(start_time, '%Y-%m-%d %H:%M:%S').astimezone(pytz.timezone(timezone))
            else:
                start = datetime.datetime.strptime(date+' '+start_time, '%Y-%m-%d %H:%M:%S').astimezone(pytz.timezone(timezone))
        else: 
            start = None
        if end_time:
            if not date == '':
                end = datetime.datetime.strptime(date+' '+end_time, '%Y-%m-%d %H:%M:%S').astimezone(pytz.timezone(timezone))
            else:
                end = datetime.datetime.strptime(end_time,'%Y-%m-%d %H:%M:%S').astimezone(pytz.timezone(timezone))
        else:
            end = None

        metadata,_,start_times,_ = group_recordings(metadata_path,start,end,sensor,selected_configuration)

        recordings = [os.path.join(data_path,meta.filename) for meta in metadata]
        all_files = glob.glob(os.path.join(data_path, "**", "*"), recursive=True)

        # Filtrer uniquement les .wav ou .WAV (hors formatted data)
        wav_files = [
            f for f in all_files
            if f.lower().endswith(".wav") and not "formatted_data" in str(f).lower()
        ]
        recordings = list(set(wav_files).intersection(set(recordings)))
        recordings = sorted(recordings, key=sort_key)

        
        leak_detection = pd.DataFrame()

        for filepath in tqdm(recordings):
            filename = filepath.split('/')[-1]

            # Vérifier que le fichier est dans l'index
            if filename in start_times.index:
                start_time = start_times.loc[filename, 'start_times']
                # Si c'est un pd.Timestamp, convertir en datetime
                print(start_time)
            else:
                print(f"{filename} n'est pas dans start_times")

            df = predict_from_filepath(filepath,start_time,BLOCK_DURATION) 

            df = df.sort_index() 
            df["Label"] = None  
            for _, row in df_labels.iterrows():
                mask = (df.index >= row["Start"]) & (df.index <= row["End"])
                df.loc[mask, "Label"] = row["Label"]
            
            df["Label"].fillna("other", inplace=True)
            leak_detection = pd.concat([leak_detection, df])  
        leak_detection.index.names = ['time']

        logger.info(f"Leak detection from ML model, compiled loaded from : {data_path}")


        if float(aggregation_sliding_window).is_integer():
            freq_str = f"{int(aggregation_sliding_window)}s"
        else:
            freq_str = f"{int(aggregation_sliding_window * 1000)}ms"

        num_cols = leak_detection.select_dtypes(include=["number"]).columns
        str_cols = leak_detection.select_dtypes(exclude=["number"]).columns

        df_num = (
            leak_detection[num_cols]
            .resample(freq_str)
            .interpolate(method="time")
        )
        df_str = (
            leak_detection[str_cols]
            .resample(freq_str)
            .ffill()
        )
        leak_detection = pd.concat([df_num, df_str], axis=1)
        leak_detection = leak_detection.rename(columns={"leak": "leak_ml_mean"})


        root_dirname = Path(data_path) / 'results' 
        os.makedirs(root_dirname, exist_ok=True)

        features_path = root_dirname.joinpath("features.h5")

        if features_path.exists():
            features = read_hdf_dataset(features_path)
            logger.info(f"features loaded from : {features_path}")

            df_features = pd.DataFrame(features['features'])

            # Récupérer le niveau 'time' du MultiIndex
            time_index = df_features.index.get_level_values('time')

            # Si naïf, le localiser en UTC
            if time_index.tz is None:
                time_index = time_index.tz_localize('UTC')

            # Reindexer leak_detection sur le niveau 'time'
            df_features["leak_ml_mean"] = leak_detection.reindex(
                time_index,
                method='ffill'
            )["leak_ml_mean"].values
            to_hdf_dataset(
                dataframes = {'features' : df_features},
                root_dirname = root_dirname,
                output_name = "features.h5",
            )
        else:
            leak_detection = leak_detection.set_index("Label", append=True)
            to_hdf_dataset(
                dataframes = {'features' : leak_detection},
                root_dirname = root_dirname,
                output_name = "features.h5",
            )
        logger.info(f"'Leak' result added to 'features' dataset, saved at : {root_dirname}")
        logger.info("Computation completed.")

        return root_dirname
        


if DEBUG:
    
    root_dirname = computation(settings,sensor,sensor_id,date,selected_configuration)
else:
    # ---- Bouton de validation ----
    if st.button("Selection to validate"):
        st.write("Selection summary")
        st.write(f"**Sensor:** {sensor}")
        st.write(f"**Sensor ID:** {sensor_id if sensor_id else 'aucun'}")
        st.write(f"**Date:** {date if date else 'aucune'}")

        parser = argparse.ArgumentParser(
            description="""Compute features for training the classifier.
        This script is used to compute the features and targets
        used by the classifier for training.""",
            formatter_class=argparse.RawTextHelpFormatter,
        )
        root_dirname = computation(settings,sensor,sensor_id,date,selected_configuration)
        st.success(
            f"Compilation completed successfully. You may now close this page.  \n Feature & BandLeq results are saved at : {root_dirname}"
        )



# %%
