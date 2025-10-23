# ==========================
# Imports standard Python
# ==========================
import os
import re
import shutil
import numpy as np
import logging
import datetime
import uuid
import glob
import wave
import contextlib
import gzip
import zipfile
import sqlite3
from collections import defaultdict
from pathlib import Path
from zoneinfo import ZoneInfo
from typing import Optional, Dict, List, Tuple, Any, Set
import warnings
import matplotlib.cm as cm

# ==========================
# Imports tiers
# ==========================
import pandas as pd
import pytz
from tqdm import tqdm
import librosa
import soundfile as sf
from pydub import AudioSegment
from sqlalchemy import create_engine, and_, Column, Integer, String
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
import rarfile
import plotly.graph_objects as go
from joblib import Parallel, delayed
from numba import njit

# ==========================
# Imports du projet / locaux (wavely)
# ==========================
from wavely.edge_metadata.models import Recording, Config
from wavely.edge_metadata import db
from wavely.datasets_analysis.preprocessing.metadata import parse_labels
from wavely.datasets_analysis.preprocessing import load_wav
from wavely.datasets_analysis import initialized_metadata_db
from wavely.eda.conf import settings


# ==========================
# Initialisations
# ==========================
warnings.filterwarnings("ignore")

# Logger principal
logger = logging.getLogger(__name__)
logging.getLogger("pydub.converter").setLevel(logging.WARNING)
logging.basicConfig(level=logging.INFO)
logging.basicConfig(level=logging.DEBUG)

# Base SQLAlchemy
Base = declarative_base()

# Initialiser la DB metadata
initialized_metadata_db()



LOG_TERMS = ["spectralirregularity", "spectralflux"]
Y_LABEL_SPECTRA = "Frequency (Hz)"  # ou récupérer dynamiquement si nécessaire

################################
### PRE-PROCESSING FUNCTIONS ###
################################
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

def get_wav_info(file_path: str) -> Dict[str, Any]:
    """Extract basic information from a WAV file (PCM or float).

    Args:
        file_path: Path to the WAV or audio file.

    Returns:
        A dictionary with the following keys:
            - n_channels: Number of audio channels.
            - sample_rate: Sample rate of the audio.
            - dtype: Data type ('int' or 'float').
            - subtype: Bit depth as a string (e.g., '16-bit').
            - od: Duration in seconds.
            - error: Error message if reading fails, otherwise None.
    """
    info = {
        "n_channels": None,
        "sample_rate": None,
        "dtype": None,
        "subtype": None,
        "od": None,  # Duration
        "error": None
    }
    try:
        with contextlib.closing(wave.open(file_path, 'rb')) as wf:
            n_channels = wf.getnchannels()
            sample_rate = wf.getframerate()
            sampwidth = wf.getsampwidth()
            n_frames = wf.getnframes()
            duration = n_frames / float(sample_rate)
            dtype = 'int' if sampwidth in [1, 2, 4] else 'float'
            subtype = f"{sampwidth*8}-bit"

        info.update({
            "n_channels": n_channels,
            "sample_rate": sample_rate,
            "dtype": dtype,
            "subtype": subtype,
            "od": duration,  
            "error": None
        })

    except wave.Error:
        try:
            data, samplerate = sf.read(file_path)
            n_channels = data.shape[1] if len(data.shape) > 1 else 1
            duration = data.shape[0] / samplerate
            if data.dtype.kind == 'i':
                dtype = 'int'
            elif data.dtype.kind == 'f':
                dtype = 'float'
            else:
                dtype = str(data.dtype)

            info.update({
                "n_channels": n_channels,
                "sample_rate": samplerate,
                "dtype": dtype,
                "subtype": str(data.dtype),
                "od": duration,  
                "error": None
            })
        except Exception as e:
            info["error"] = str(e)

    except Exception as e:
        info["error"] = str(e)

    return info

def copy_db_structure(source_db: str, new_db: str) -> Dict[str, List[Tuple]]:
    """Copy the structure of a SQLite database to a new database without data.

    Internal SQLite tables are excluded. Ensures the 'Label' table has
    'start_time' and 'end_time' columns.

    Args:
        source_db: Path to the source SQLite database.
        new_db: Path to the destination SQLite database.

    Returns:
        A dictionary mapping table names to a list of their column info
        (as returned by `PRAGMA table_info`).
    """
    src = sqlite3.connect(source_db)
    dst = sqlite3.connect(new_db)
    c_src = src.cursor()
    c_dst = dst.cursor()

    c_src.execute("SELECT sql, name FROM sqlite_master WHERE type='table';")
    tables = c_src.fetchall()
    table_columns = {}

    for create_sql, table_name in tables:
        if not create_sql or table_name.startswith("sqlite_"):
            continue

        # Créer la table dans la DB destination
        c_dst.execute(create_sql)

        # Vérifier les colonnes dans la DB destination
        c_dst.execute(f"PRAGMA table_info({table_name})")
        cols = c_dst.fetchall()
        table_columns[table_name] = cols

        # Cas spécial pour Label : ajouter start_time et end_time si manquantes
        if table_name == "Label":
            existing_cols = {col[1] for col in cols}

            for col_name in ["start_time", "end_time"]:
                if col_name not in existing_cols:
                    try:
                        c_dst.execute(
                            f"ALTER TABLE Label ADD COLUMN {col_name} VARCHAR NOT NULL DEFAULT ''"
                        )
                    except sqlite3.OperationalError as e:
                        logger.warning(f"Skipping {col_name}: {e}")

    src.close()
    dst.commit()
    dst.close()
    return table_columns

def populate_db(
    selected_configuration: Dict[str, Any],
    sensor: str,
    db_path: str,
    table_columns: Dict[str, list],
    data_path: str
) -> None:
    """Populate a database with WAV file info and configuration data.

    Args:
        selected_configuration: Dictionary containing configuration parameters.
        sensor: Name of the sensor.
        db_path: Path to the SQLite database to populate.
        table_columns: Dictionary mapping table names to their column info
            (as returned by `PRAGMA table_info`).
        data_path: Path where additional data (e.g., time_shift.csv) is stored.

    Raises:
        ValueError: If no WAV files are found in the expected folder.
    """
    conn = sqlite3.connect(db_path)
    c = conn.cursor()
    
    # --- Locate WAV folder based on DB location ---
    wav_folder = os.path.dirname(os.path.abspath(db_path))
    
    # --- Find WAV files ---
    wav_files = sorted(glob.glob(os.path.join(wav_folder, "**", "*.[Ww][Aa][Vv]"), recursive=True))
    if not wav_files:
        raise ValueError(f"No WAV files found in folder '{wav_folder}'.")
    
    # --- Use first WAV to get audio parameters for Config ---
    first_wav_info = get_wav_info(wav_files[0])
    sample_rate = first_wav_info['sample_rate']
    n_channels = first_wav_info['n_channels']
    dtype = first_wav_info['dtype']
    subtype = first_wav_info['subtype']
    
    # --- Populate Config table ---
    if "Config" in table_columns:
        cols_info = table_columns["Config"]
        for i, mic in enumerate(selected_configuration['microphone'], start=1):
            insert_cols = []
            insert_values = []
            
            for cid, col_name, col_type, notnull, _ , pk in cols_info:
                if pk == 1 and "INT" in col_type:
                    continue  # AUTOINCREMENT
                elif col_name == "sample_rate":
                    insert_cols.append(col_name)
                    insert_values.append(sample_rate)
                elif col_name == "n_channels":
                    insert_cols.append(col_name)
                    insert_values.append(n_channels)
                elif col_name == "dtype":
                    insert_cols.append(col_name)
                    insert_values.append(dtype)
                elif col_name == "subtype":
                    insert_cols.append(col_name)
                    insert_values.append(subtype)
                elif col_name == "device":
                    insert_cols.append(col_name)
                    insert_values.append(selected_configuration['device'])
                elif col_name == "microphone":
                    insert_cols.append(col_name)
                    insert_values.append(mic)
                elif col_name == "sensor":
                    insert_cols.append(col_name)
                    insert_values.append(f"{sensor}_ch{i}")
                elif col_name == "rss_calibration":
                    insert_cols.append(col_name)
                    insert_values.append(selected_configuration['rss_calibration'])
                elif col_name == "preamp_gain":
                    insert_cols.append(col_name)
                    insert_values.append(selected_configuration['preamp_gain'][i-1])
                elif col_name == "normalise_gain":
                    insert_cols.append(col_name)
                    insert_values.append(selected_configuration['normalise_gain'])
                elif col_name == "campaign":
                    insert_cols.append(col_name)
                    insert_values.append(settings.campaign)
                elif col_name == "device_name":
                    insert_cols.append(col_name)
                    insert_values.append(selected_configuration['device_name'])
                elif col_name == "mic_calibration":
                    insert_cols.append(col_name)
                    insert_values.append(selected_configuration['mic_calibration'][i-1])
                elif notnull:
                    # default values
                    if "CHAR" in col_type or "TEXT" in col_type:
                        insert_cols.append(col_name)
                        insert_values.append("")
                    elif "INT" in col_type:
                        insert_cols.append(col_name)
                        insert_values.append(0)
                    elif "REAL" in col_type or "FLOAT" in col_type or "DOUBLE" in col_type:
                        insert_cols.append(col_name)
                        insert_values.append(0.0)
                    else:
                        insert_cols.append(col_name)
                        insert_values.append(None)
            
            if insert_cols:
                cols_str = ", ".join(insert_cols)
                placeholders = ", ".join(["?"] * len(insert_cols))
                sql = f"INSERT INTO Config ({cols_str}) VALUES ({placeholders})"
                c.execute(sql, insert_values)
    if selected_configuration['time_shift_ms']:
        time_shift_sec = selected_configuration['time_shift_ms'] / 1000.0
        df_shift = pd.DataFrame({"time_shift_ms": [selected_configuration['time_shift_ms']]})
        df_shift.to_csv(os.path.join(data_path,"time_shift.csv"), index=False)
    else:
        time_shift_sec = 0.0
    # --- Populate other tables from WAV files ---
    for file in wav_files:
        full_path = file
        wav_info = get_wav_info(full_path)
        mod_timestamp = os.path.getmtime(full_path)
        # Interpréter le timestamp comme relevé par le capteur
        file_mod_time_local = datetime.datetime.fromtimestamp(mod_timestamp, tz=ZoneInfo(settings.data_acquisition.timezone))
        offset = file_mod_time_local.utcoffset().total_seconds()
        duration_sec = wav_info.get("od", 0)
        # Correction (timestamp UTC final)
        start_time = mod_timestamp - offset - duration_sec + time_shift_sec
        start_time = int(start_time)

        basename = os.path.basename(full_path)
        config_id = None
        prefix = selected_configuration['channel_prefix']  

        match = re.search(rf"{prefix}(\d+)", basename)
        if match:
            config_id = int(match.group(1))
        
        for table, cols_info in table_columns.items():
            if table == "Config":
                continue
            if table not in ["Config", "Recording"]:
                continue
            
            insert_cols = []
            insert_vals = []
            
            for cid, col_name, col_type, notnull, dflt_value, pk in cols_info:
                if table == "Recording":
                    if col_name == "start_time":
                        insert_cols.append(col_name)
                        insert_vals.append(start_time)  # store as timestamp
                        continue
                    elif col_name == "recording_type":
                        insert_cols.append(col_name)
                        insert_vals.append("continuous")
                        continue
                    elif col_name == "filename":
                        insert_cols.append(col_name)
                        relative_path = os.path.relpath(full_path, os.path.dirname(db_path))
                        insert_vals.append(relative_path)
                        # insert_vals.append(os.path.basename(full_path))
                        continue
                    elif col_name == "duration":
                        insert_cols.append(col_name)
                        insert_vals.append(duration_sec)
                        continue
                    elif col_name == "config_id":
                        insert_cols.append(col_name)
                        insert_vals.append(config_id)
                        continue

                if col_name in wav_info:
                    insert_cols.append(col_name)
                    insert_vals.append(wav_info[col_name])
                elif notnull:
                    if "uuid" in col_name.lower() or (pk == 1 and "INT" not in col_type):
                        insert_cols.append(col_name)
                        insert_vals.append(str(uuid.uuid4()))
                    elif pk == 1 and "INT" in col_type:
                        continue
                    elif "CHAR" in col_type or "TEXT" in col_type:
                        insert_cols.append(col_name)
                        insert_vals.append("")
                    elif "INT" in col_type:
                        insert_cols.append(col_name)
                        insert_vals.append(0)
                    elif "REAL" in col_type or "FLOAT" in col_type or "DOUBLE" in col_type:
                        insert_cols.append(col_name)
                        insert_vals.append(0.0)
                    else:
                        insert_cols.append(col_name)
                        insert_vals.append(None)

            if insert_cols:
                cols_str = ", ".join(insert_cols)
                placeholders = ", ".join(["?"] * len(insert_cols))
                sql = f"INSERT INTO {table} ({cols_str}) VALUES ({placeholders})"
                c.execute(sql, insert_vals)
    
    conn.commit()
    conn.close()
    logger.info(f"New database '{os.path.basename(db_path)}' created and populated successfully.")

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

def get_recordings_from_time_range(metadata_path: str, 
                                   start_time: datetime, 
                                   end_time: datetime,
                                   sensor: str,
                                   selected_configuration: dict) -> tuple:
    """
    Retrieves .wav file paths from the SQLite metadata database within a specified time range.

    Args:
        metadata_path (str): Path to the SQLite metadata database file.
            Example: "/path/to/metadata.db"
        start_time (datetime): Start datetime for filtering recordings.
        end_time (datetime): End datetime for filtering recordings.

    Returns:
        tuple:
            recordings (list[str]): List of full paths to valid .wav files existing on disk.
            db_url (str): SQLAlchemy database URL used to connect.
            start_times (pd.DataFrame): DataFrame indexed by filename containing start times as UNIX timestamps.

    Notes:
        - The function filters recordings based on SENSOR, RECORDING_TYPE if set.
        - Only files physically existing on disk are returned.
        - Logging messages provide info about the retrieval process and missing files.
    """
    results, db_url, start_times = query_recordings(
        metadata_path, start_time, end_time, sensor, selected_configuration, time_buffer=0
    )

    if not results:
        return [], db_url, start_times

    base_dir = os.path.dirname(metadata_path)
    valid_paths, valid_filenames = [], []

    for rec in results:
        full_path = os.path.join(base_dir, rec.filename)
        if os.path.exists(full_path):
            valid_paths.append(full_path)
            valid_filenames.append(rec.filename)
        else:
            logger.debug(f"File not found: {full_path}")

    # Restrict start_times to valid files
    start_times = start_times.loc[valid_filenames]

    # --- Resample logic (spécifique à cette fonction) ---
    resample_rate = settings.data_acquisition.resampling_rate
    if resample_rate:
        valid_file_paths = [Path(metadata_path).parent / fname for fname in valid_filenames]
        for wav_path in tqdm(valid_file_paths, desc="Resampling WAV files"):
            data, original_sr = librosa.load(wav_path, sr=None)
            if original_sr != resample_rate:
                data = librosa.resample(data, orig_sr=original_sr, target_sr=resample_rate)
                sf.write(wav_path, data, resample_rate, format="WAV")

        # Mise à jour DB
        engine = create_engine(db_url)
        Session = sessionmaker(bind=engine)
        with Session() as session:
            config_ids = session.query(Recording.config_id).filter(
                Recording.filename.in_(valid_filenames)
            ).distinct().all()
            config_ids = [cid[0] for cid in config_ids]
            if config_ids:
                configs_to_update = session.query(Config).filter(Config.id.in_(config_ids)).all()
                for cfg in configs_to_update:
                    cfg.sample_rate = resample_rate
                session.commit()
                logger.info(f"Updated sample_rate to {resample_rate} for configurations: {config_ids}")

    return valid_paths, db_url, start_times

def decompress_and_remove_files(folder_path: str) -> List[str]:
    """Decompress all ZIP, RAR, and GZ files in a folder and remove them afterward.

    This function processes all compressed files in the given folder. If a file
    fails to decompress, it is skipped, and the function continues with the next one.

    Args:
        folder_path: Path to the folder containing compressed files.
            Example: "/path/to/compressed_files"

    Returns:
        A list of decompressed file paths or filenames extracted from the archives.

    Prints:
        - Info messages for missing files
        - Progress during decompression
        - Errors encountered during decompression
        - Summary of total files processed
    """
    decompressed_files = []
    
    # Filtrer uniquement les fichiers compressés
    compressed_files = [f for f in os.listdir(folder_path) if f.endswith(('.zip', '.rar', '.gz'))]

    if not compressed_files:
        logger.info("No file to decompress")
        return decompressed_files

    for filename in tqdm(compressed_files, desc='Decompressing files'):
        file_path = os.path.join(folder_path, filename)
        
        try:
            if filename.endswith('.zip'):
                with zipfile.ZipFile(file_path, 'r') as zip_ref:
                    zip_ref.extractall(folder_path)
                    decompressed_files.extend(zip_ref.namelist())
                os.remove(file_path)

            elif filename.endswith('.gz'):
                decompressed_file_path = os.path.splitext(file_path)[0]
                with gzip.open(file_path, 'rb') as f_in:
                    with open(decompressed_file_path, 'wb') as f_out:
                        shutil.copyfileobj(f_in, f_out)
                        decompressed_files.append(decompressed_file_path)
                os.remove(file_path)

            elif filename.endswith('.rar'):
                with rarfile.RarFile(file_path, 'r') as rar_ref:
                    rar_ref.extractall(folder_path)
                    decompressed_files.extend(rar_ref.namelist())
                os.remove(file_path)

        except Exception as e:
            logger.info(f"Error decompressing file {filename}: {e}")
            continue

    logger.info(f"Decompression completed. Total files processed: {len(decompressed_files)}")
    return decompressed_files

def concatenate_wav_files(
    output_dir: str, 
    recordings: List[str], 
    start_times: pd.DataFrame,
    selected_configuration
) -> None:
    """Concatenate WAV files into time-aligned batches under a maximum duration.

    This function groups and exports WAV files into batches such that the cumulative
    duration stays below `settings.max_long_format_duration`, positioning sounds on
    a silent base according to their start times.

    Files are sorted by their start timestamps, gain is applied, 32-bit files are
    converted to 16-bit, and each sound is overlaid at the correct relative position
    in the batch. When the batch duration limit is reached, the batch is exported.

    This facilitates data annotation by creating time-aligned audio batches.

    Args:
        output_dir: Directory where output WAV batches and label data are saved.
        recordings: List of WAV file paths to process.
        start_times: DataFrame indexed by recording identifiers (str) containing a
            'start_times' column with UNIX timestamps (float or int). Index entries
            should match or contain the basename of the files in `recordings`.
        selected_configuration: Configuration dictionary for processing metadata.

    Returns:
        None. WAV files are exported in batches into `output_dir`.

    Prints:
        Processing progress and error messages for files that cannot be processed.
    """
    os.makedirs(output_dir, exist_ok=True)

    # Organisation des fichiers par start_time
    recordings = sorted(
        recordings,
        key=lambda p: next(start_times.loc[idx, 'start_times']
                           for idx in start_times.index
                           if os.path.basename(p) in idx)
    )
    batch = []
    batch_start_time = None
    combined = None
    start_ref = start_times['start_times'].min()

    for path in tqdm(recordings, desc="Processing batches"):
        try:
            matches = start_times.index[start_times.index.to_series().str.contains(os.path.basename(path))]
            timestamp = start_times.loc[matches[0], 'start_times']
            delta = timestamp - start_ref
            if isinstance(delta, pd.Timedelta):
                file_start_ms = int(delta.total_seconds() * 1000)
            else:
                file_start_ms = int(delta * 1000)


            sound = AudioSegment.from_wav(path)
            duration_ms = len(sound)
            sound = sound.set_sample_width(2)

            if not batch:
                batch_start_time = file_start_ms
                batch = [path]
                combined = AudioSegment.silent(duration=duration_ms)
                combined = combined.overlay(sound, position=0)
                continue

            file_end_time = file_start_ms + duration_ms
            if file_end_time - batch_start_time <= settings.max_long_format_duration*1000:
                position_delta = file_start_ms - batch_start_time
                position_ms = position_delta

                combined_duration = len(combined)
                if position_ms + duration_ms > combined_duration:
                    pad = position_ms + duration_ms - combined_duration
                    combined += AudioSegment.silent(duration=pad)

                combined = combined.overlay(sound, position=position_ms)
                batch.append(path)
            else:
                export_batch(output_dir, batch, combined, start_times,selected_configuration)
                batch_start_time = file_start_ms
                batch = [path]
                combined = AudioSegment.silent(duration=duration_ms)
                combined = combined.overlay(sound, position=0)

        except Exception as e:
            logger.info(f"Erreur avec le fichier {path}: {e}")

    if batch:
        export_batch(output_dir, batch, combined, start_times,selected_configuration)

def export_batch(
    output_dir: str, 
    batch: List[str], 
    combined: AudioSegment, 
    start_times, 
    selected_configuration
) -> None:
    """Export a combined audio batch as a WAV file with timestamped filename.

    The function determines the UTC start time of the first audio file in the batch,
    converts it to the target timezone, generates a timestamped filename, creates a
    corresponding folder for label data, and exports the combined audio.

    This process facilitates data annotation by keeping audio batches time-aligned.

    Args:
        output_dir: Directory where the WAV file and label folder will be saved.
        batch: List of file paths in the batch. The first file is used to determine
            the export timestamp.
        combined: Combined AudioSegment of all batch files.
        start_times: DataFrame indexed by file identifiers (or basenames) with a
            'start_times' column containing UNIX timestamps (seconds).
        selected_configuration: Configuration dictionary that may contain channel prefix.

    Returns:
        None. The WAV file and a label subdirectory are created on disk.

    Prints:
        Confirmation message with the path of the exported WAV file.
    """
    first_file_path = batch[0]

    first_start_timestamp = start_times.loc[start_times.index[start_times.index.to_series().apply(lambda idx: os.path.basename(idx) in os.path.basename(first_file_path))][0], 'start_times']

    first_datetime_utc = datetime.datetime.fromtimestamp(first_start_timestamp.timestamp(), tz=pytz.UTC)
    first_datetime_tz = first_datetime_utc.astimezone(pytz.timezone(settings.data_acquisition.timezone))
    
    output_filename = f"{first_datetime_tz.strftime('%Y-%m-%d_%H:%M:%S')}.wav"
    output_path = os.path.join(output_dir, output_filename)

    batch_name = os.path.splitext(output_filename)[0]
    if selected_configuration['channel_prefix']:
        data_folder = os.path.join(os.path.dirname(output_dir), 'labels', batch_name + '_data')
    else:
        data_folder = os.path.join(output_dir, 'labels', batch_name + '_data')
    os.makedirs(data_folder, exist_ok=True)

    combined.export(output_path, format='wav')

    logger.info(f"Exported file : {output_path}")
######################################
### PROCESSING CLASSES & FUNCTIONS ###
######################################


class Label(Base):
    __tablename__ = "Label"

    id = Column(Integer, primary_key=True, autoincrement=True)
    name = Column(String, nullable=False)
    start_time = Column(String, nullable=False)  
    end_time = Column(String, nullable=False)    


class ExpertLeakDetector:
    def __init__(
        self,
        min_limit_mean: float = 0.2,
        max_limit_std: float = 0.05,
        max_limit_spectralflatness: float | None = None,
        n_blocks_integration: int = 10,
        n_blocks_hop: int = 5,
        ultrasound_threshold_db: float = -50.0,
        sample_rate: float = 48000.0,
    ):
        # Initialisation des attributs utilisés par expert()
        self._min_limit_mean = min_limit_mean
        self._max_limit_std = max_limit_std
        self._max_limit_spectralflatness = max_limit_spectralflatness
        self._n_blocks_integration = n_blocks_integration
        self._n_blocks_hop = n_blocks_hop
        self._ultrasoundlevel = ultrasound_threshold_db
        self._data_acquisition_settings = type("MockDAQ", (), {"sample_rate": sample_rate})()

    def expert_model_predict(self, features_df: pd.DataFrame) -> pd.DataFrame:
        """Predicts the event type based on the provided acoustic features.

        Args:
            features_df: DataFrame containing the acoustic features.

        Outputs:
            predictions: DataFrame containing the predictions as a pandas series.

        """
        ultrasoundlevel_bgn_lin = 10 ** (self._ultrasoundlevel / 20)
        # Convert ultrasound_level to linear scale and normalize by the background level
        
        features_df["ultrasoundlevel_lin"] = 10 ** (features_df["ultrasoundlevel"] / 20)
        # MODIF ROMAIN 'ultrasoundlevel => ultrasoundlevel_lin'
        features_df["ultrasoundlevel_norm"] = (
            (features_df["ultrasoundlevel_lin"] / ultrasoundlevel_bgn_lin) - 1
        )

        # Compute the spectral_centroid and spectral flatness
        features_df["spectralcentroid_norm"] = features_df["spectralcentroid"] / (
            self._data_acquisition_settings.sample_rate / 2
        )

        export_leak_estimator = []
        # Iterate over the features_df to compute the leak detection
        for i in range(
            0, len(features_df) - self._n_blocks_integration + 1, self._n_blocks_hop
        ):
            window_data = features_df.iloc[i : i + self._n_blocks_integration]
            expert_features = {
                "ultrasoundlevel_mean": window_data["ultrasoundlevel_norm"].mean(),
                "spectralcentroid_mean": window_data["spectralcentroid_norm"].mean(),
                "ultrasoundlevel_std": window_data["ultrasoundlevel_norm"].std(),
                "spectralcentroid_std": window_data["spectralcentroid_norm"].std(),
                "time": features_df.index[i],
            }

            expert_features["product_mean"] = (
                expert_features["ultrasoundlevel_mean"]
                * expert_features["spectralcentroid_mean"]
            )
            expert_features["product_std"] = (
                expert_features["ultrasoundlevel_std"]
                * expert_features["spectralcentroid_std"]
            )

            # expert_features["leak_expert_mean"] = (
            #     (expert_features["product_mean"] > self._min_limit_mean)
            #     & (expert_features["product_std"] < self._max_limit_std)
            # ).astype(int)
            # MODIF Romain : privilégier ue droite (plus performant)
            # Définition des deux points de la droite
            x1, y1 = 0.05, 1e-4
            x2, y2 = 10, 0.5

            # Calcul des coefficients de la droite y = m*x + c
            m = (y2 - y1) / (x2 - x1)
            c = y1 - m * x1

            # Calcul de la valeur de la droite pour chaque product_mean
            y_line = m * expert_features["product_mean"] + c

            # Condition : être sous la droite
            expert_features["leak_expert_mean"] = (
                expert_features["product_std"] < y_line
            ).astype(int)

            # Add spectral flatness condition if defined
            if self._max_limit_spectralflatness is not None:
                expert_features["spectralflatness_mean"] = window_data[
                    "spectralflatness"
                ].mean()
                expert_features["leak_expert_mean"] &= (
                    expert_features["spectralflatness_mean"]
                    < self._max_limit_spectralflatness
                ).astype(int)
            export_leak_estimator.append(expert_features)

        if len(export_leak_estimator) == 0:
            return None
        else:
            return pd.DataFrame(export_leak_estimator).set_index("time")

def apply_expert_model(dataset_1d,filenames_root_dir):
    dt = (dataset_1d.index.get_level_values('time')[1] - dataset_1d.index.get_level_values('time')[0]).total_seconds()
    detector = ExpertLeakDetector(
        min_limit_mean=settings.features.leak_models.expert_model.min_limit_mean,
        max_limit_std=settings.features.leak_models.expert_model.max_limit_std,
        max_limit_spectralflatness=settings.features.leak_models.expert_model.max_limit_spectralflatness,
        n_blocks_integration=int(np.floor(settings.features.aggregation_sliding_window/dt)),
        n_blocks_hop=int(np.floor(settings.features.aggregation_sliding_window/dt)),
        ultrasound_threshold_db=settings.features.leak_models.expert_model.ultrasound_threshold_db,
        sample_rate=pd.read_csv(os.path.join(filenames_root_dir,'audio_files_info.csv'))['sample_rate'].iloc[0],
    )
    features_df = dataset_1d.copy()
    result = detector.expert_model_predict(features_df.droplevel(level=[0,1,3]))
    return result

def group_audacity_labels(labels_dir: str, label_name: str, tz="UTC") -> pd.DataFrame:
    """
    Traverse all subfolders in labels_dir and group Audacity label files into a single DataFrame.
    
    Args:
        labels_dir (str): Folder containing subfolders with label files.
        label_name (str): Name of the label file to look for in each subfolder.
        tz (str): Timezone for the timestamps.
    
    Returns:
        pd.DataFrame: Concatenated labels, or empty DataFrame if no files found.
    """
    labels_list = []

    for subdir in sorted(os.listdir(labels_dir)):
        subdir_path = os.path.join(labels_dir, subdir)
        label_file = os.path.join(subdir_path, label_name)
        if os.path.exists(label_file):
            df_label = format_audacity_label(label_file, tz)
            labels_list.append(df_label)

    if not labels_list:
        logger.warning(f"No label file '{label_name}' found in {labels_dir}.")
        return pd.DataFrame()
    
    df_labels = pd.concat(labels_list, ignore_index=True)
    logger.info(f"Concatenation complete. Total number of labels: {len(df_labels)}")
    return df_labels

def format_audacity_label(label_file: str, tz: str = "UTC") -> pd.DataFrame:
    """Convert an Audacity label file into a DataFrame with absolute timestamps.

    The function parses a standard Audacity label file (text format),
    aligns labels with the absolute start time extracted from the
    parent folder name (`YYYY-MM-DD_hh:mm:ss`), and returns
    a DataFrame containing the formatted label intervals.

    Args:
        label_file (str): Path to the Audacity label file.
        tz (str, optional): Timezone to localize timestamps. Defaults to "UTC".

    Returns:
        pd.DataFrame: DataFrame with columns:
            - `Label` (str): The label text.
            - `Start` (datetime64[ns, tz]): Absolute start timestamp.
            - `End` (datetime64[ns, tz]): Absolute end timestamp.

    Raises:
        ValueError: If the folder name does not contain a valid datetime string.
        FileNotFoundError: If the label file does not exist.

    Example:
        >>> df = format_audacity_label("2025-08-04_01:42:29_data/Label 1.txt")
        >>> print(df.head())
               Label                      Start                        End
        0   Drone A 2025-08-04 01:42:35+00:00 2025-08-04 01:42:40+00:00
    """
    # parse_labels should return a list of tuples (Label, Start, End)
    labels = parse_labels(label_file)  
    df = pd.DataFrame(labels, columns=["Label", "Start", "End"])

    folder_path = os.path.dirname(label_file)
    datetime_match = re.search(r'(\d{4}-\d{2}-\d{2}_\d{2}:\d{2}:\d{2})', folder_path)
    if not datetime_match:
        raise ValueError(f"No valid datetime found in folder path: {folder_path}")

    t0 = (
        pd.Timestamp(
            datetime.datetime.strptime(datetime_match.group(0), "%Y-%m-%d_%H:%M:%S")
        ).tz_localize(tz)
    )

    df["Start"] = df["Start"].apply(lambda x: t0 + datetime.timedelta(seconds=x))
    df["End"] = df["End"].apply(lambda x: t0 + datetime.timedelta(seconds=x))

    return df

def write_labels_to_csv(df_labels: pd.DataFrame, base_path: str) -> None:
    """Write labels DataFrame to a CSV file inside the 'labels' subfolder.

    This function ensures that the `labels` directory exists inside the
    provided `base_path`, then writes the given DataFrame as `labels.csv`.

    Args:
        df_labels (pd.DataFrame): DataFrame containing labels with columns such as
            `Label`, `Start`, and `End`.
        base_path (str): Path to the parent directory where the `labels`
            folder will be created (if it does not already exist).

    Returns:
        None

    Raises:
        OSError: If the CSV file cannot be written (e.g., due to permissions).

    Example:
        >>> labels = pd.DataFrame([
        ...     {"Label": "Drone", "Start": "2025-08-04 01:42:35", "End": "2025-08-04 01:42:40"}
        ... ])
        >>> write_labels2csv(labels, "/path/to/session")
        labels.csv written to: /path/to/session/labels/labels.csv
    """
    labels_dir = os.path.join(base_path, "labels")
    os.makedirs(labels_dir, exist_ok=True)

    csv_path = os.path.join(labels_dir, "labels.csv")
    df_labels.to_csv(csv_path, index=False)
    logger.info(f"labels.csv written to: {csv_path}")

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


def build_dataset(
    filenames_root_dir: Path,
    fullpaths: Path,
    block_overlap: float,
    params: Dict[str, Any],
    df_labels: pd.DataFrame,
    start: datetime = None,
    end:datetime = None,
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Build a dataset of features and targets from WAV recordings.

    This function performs the following steps:

    1. Loads audio recordings from the specified file paths.
    2. Extracts features using the parameters provided in `params`.
    3. Optionally filters the data based on a time range:
       - If both `start` and `end` are provided, only recordings within that interval are kept.
       - If only `start` is provided, all recordings after `start` are kept.
       - If only `end` is provided, all recordings before `end` are kept.
    4. Optionally assigns labels from a global `df_labels` DataFrame, if it is not empty.

    **Args:**
        filenames_root_dir (Path): Root directory containing the audio files.
        fullpaths (Path): List or sequence of file paths to process.
        block_overlap (float): Overlap between signal blocks in the range [0, 1).
        params (Dict[str, Any]): Additional keyword arguments to pass to
            `load_wav.load_wav_dataset`, e.g., filter parameters, feature settings.
        df_labels: dataframe containing annotation informations (start, end, labels)
        start (datetime, optional): Start time for filtering recordings. Defaults to None.
        end (datetime, optional): End time for filtering recordings. Defaults to None.

    **Returns:**
        Tuple[pd.DataFrame, pd.DataFrame]:
            - `dataset_1d`: Features extracted in tabular (1D) format.
            - `dataset_2d`: Features extracted in spectrogram-like (2D) format.

    **Raises:**
        ValueError: If required parameters are missing or invalid.
    """


    dataset_1d, dataset_2d, _ = load_wav.load_wav_dataset(
        root_dir=filenames_root_dir,
        filenames=fullpaths,
        block_overlap=block_overlap,
        **params,  # e.g., filter params, feature settings
    )

    # --- Time filtering ---
    if start and end:
        dataset_1d = dataset_1d[
            (dataset_1d.index.get_level_values("time") >= start)
            & (dataset_1d.index.get_level_values("time") <= end)
        ]
        dataset_2d = dataset_2d[
            (dataset_2d.index.get_level_values("time") >= start)
            & (dataset_2d.index.get_level_values("time") <= end)
        ]
    elif start:
        dataset_1d = dataset_1d[
            dataset_1d.index.get_level_values("time") >= start
        ]
        dataset_2d = dataset_2d[
            dataset_2d.index.get_level_values("time") >= start
        ]
    elif end:
        dataset_1d = dataset_1d[
            dataset_1d.index.get_level_values("time") <= end
        ]
        dataset_2d = dataset_2d[
            dataset_2d.index.get_level_values("time") <= end
        ]
    
    # --- Label assignment ---
    if not df_labels.empty:
        dataset_1d = assign_labels_to_dataset(dataset_1d, df_labels)
        dataset_2d = assign_labels_to_dataset(dataset_2d, df_labels)

    return dataset_1d, dataset_2d

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



# Numba-optimized skew
@njit
def skew_numba(x):
    n = len(x)
    if n == 0:
        return 0.0
    mean = np.mean(x)
    m2 = np.mean((x - mean) ** 2)
    if m2 == 0:
        return 0.0
    m3 = np.mean((x - mean) ** 3)
    return m3 / (m2 ** 1.5)


def compute_stat_features(df: pd.DataFrame, window_seconds: float, n_jobs: int = -1) -> pd.DataFrame:
    """
    Optimized rolling statistics per (filename, channel, Label),
    with window size expressed in seconds and parallelization.
    The result is resampled at intervals of window_seconds.
    """
    # Ensure time is datetime
    if not pd.api.types.is_datetime64_any_dtype(df.index.get_level_values("time")):
        df = df.reset_index()
        df["time"] = pd.to_datetime(df["time"], utc=True, errors="coerce")
        df = df.set_index(["filename", "channel", "Label", "time"])
    
    # Convert channel to string
    df = df.reset_index()
    df["channel"] = df["channel"].astype(str)

    # Round time to milliseconds
    df["time"] = df["time"].dt.round("ms")

    # Window in ms
    window_ms = int(round(window_seconds * 1000))
    window = f"{window_ms}ms"

    # Numeric columns only
    num_cols = df.select_dtypes(include="number").columns

    # Split groups for parallel processing
    groups = [g for _, g in df.groupby(["filename", "channel", "Label"])]

    def compute_group(g):
        g = g.sort_values("time").copy()
        roll = g.rolling(window=window, on="time")
        out = pd.DataFrame(index=g.index)

        for c in num_cols:
            series_roll = roll[c]
            out[f"{c}_mean"] = series_roll.mean().values
            out[f"{c}_std"] = series_roll.std().values
            out[f"{c}_min"] = series_roll.min().values
            out[f"{c}_max"] = series_roll.max().values
            out[f"{c}_skew"] = series_roll.apply(skew_numba, raw=True).values

        df_group = pd.concat([g, out], axis=1)

        # -------------------------------
        # Resample at intervals of window_seconds
        # -------------------------------
        t_start = df_group["time"].min()
        t_end = df_group["time"].max()
        new_index = pd.date_range(start=t_start, end=t_end, freq=pd.Timedelta(seconds=window_seconds))
        df_group = df_group.set_index("time").reindex(new_index, method="nearest").reset_index()
        df_group.rename(columns={"index": "time"}, inplace=True)
        df_group["channel"] = g["channel"].iloc[0]
        df_group["filename"] = g["filename"].iloc[0]
        df_group["Label"] = g["Label"].iloc[0]

        return df_group

    # Parallel computation
    results = Parallel(n_jobs=n_jobs)(
        delayed(compute_group)(g) for g in tqdm(groups, desc="Rolling (parallélisé)")
    )

    df_out = pd.concat(results, axis=0)
    df_out = df_out.set_index(["filename", "channel", "Label", "time"]).sort_index()
    return df_out

def downsample_bandleq(df: pd.DataFrame, 
        window_seconds: float
        ) -> pd.DataFrame:
    """Downsample 'bandleq' values over fixed time windows.

    This function averages 'bandleq' measurements within consecutive time
    windows of a specified length for each frequency, channel, and file.
    The resulting DataFrame maintains all frequencies for each downsampled
    time step and preserves the MultiIndex structure.

    Args:
        df (pd.DataFrame): MultiIndex DataFrame with levels
            ('filename', 'channel', 'freq', 'time', 'Label') containing a
            column 'bandleq'.
        window_seconds (float): Size of the downsampling window in seconds.

    Returns:
        pd.DataFrame: Downsampled DataFrame with the same MultiIndex levels
            ('filename', 'channel', 'freq', 'time', 'Label'), sorted by
            filename, channel, time, frequency, and Label.
    """
    df_copy = df.copy()

    # Ensure 'time' is a datetime type
    if not pd.api.types.is_datetime64_any_dtype(df_copy.index.get_level_values('time')):
        df_copy = df_copy.reset_index()
        df_copy['time'] = pd.to_datetime(df_copy['time'], utc=True)
        df_copy = df_copy.set_index(['filename', 'channel', 'freq', 'time', 'Label'])
    # Reset index for calculations
    df_reset = df_copy.reset_index()
    df_reset['channel'] = df_reset['channel'].astype(str)

    # Create time blocks
    time_ns = df_reset['time'].view('int64')
    window_ns = int(window_seconds * 1e9)
    df_reset['time_block'] = pd.to_datetime((time_ns // window_ns) * window_ns, utc=True)

    # Group by file, channel, frequency, label, and time block
    group_cols = ['filename', 'channel', 'freq', 'Label', 'time_block']
    df_down = df_reset.groupby(group_cols, as_index=False)['bandleq'].mean()

    # Restore MultiIndex
    df_down = df_down.set_index(
        ['filename', 'channel', 'freq', 'time_block', 'Label']
    )
    df_down.index.rename(
        ['filename', 'channel', 'freq', 'time', 'Label'], inplace=True
    )

    # Sort by filename, channel, time, frequency, and Label
    df_down = df_down.sort_index(
        level=['filename', 'channel', 'time', 'freq', 'Label']
    )

    return df_down

def assign_labels_to_dataset(
    df: pd.DataFrame,
    df_labels: pd.DataFrame,
    time_index_name: str = "time",
    label_index_name: str = "Label",
    default_label: str = "other"
) -> pd.DataFrame:
    """Assign labels to a DataFrame based on time intervals and add as a MultiIndex level.

    This function assigns labels to each row in `df` according to the intervals
    defined in `df_labels`. The assigned label is added as a new level in the
    MultiIndex. Rows not falling into any interval receive the `default_label`.

    Args:
        df (pd.DataFrame): DataFrame with a MultiIndex containing a time level.
        df_labels (pd.DataFrame): DataFrame containing 'Label', 'Start', and 'End' columns.
        time_index_name (str): Name of the time level in df (default 'time').
        label_index_name (str): Name of the new MultiIndex level for the label (default 'Label').
        default_label (str): Label for times not in any interval (default 'other').

    Returns:
        pd.DataFrame: DataFrame with the same data as `df` but with an added
        MultiIndex level containing labels.
    """
    # Copy to avoid modifying original
    df_labeled = df.copy()

    # Drop rows with missing values
    df_labeled = df_labeled.dropna(how="any")

    # Ensure label intervals are datetime
    df_labels = df_labels.copy()
    df_labels["Start"] = pd.to_datetime(df_labels["Start"], format="mixed")
    df_labels["End"] = pd.to_datetime(df_labels["End"], format="mixed")

    # Temporary columns for assignment
    df_labeled["_label_temp"] = default_label
    df_labeled["_time_temp"] = pd.to_datetime(df_labeled.index.get_level_values(time_index_name))

    # Assign labels based on intervals
    for _, interval in df_labels.iterrows():
        mask = (df_labeled["_time_temp"] >= interval["Start"]) & (df_labeled["_time_temp"] <= interval["End"])
        df_labeled.loc[mask, "_label_temp"] = interval["Label"]

    # Remove temporary time column
    df_labeled = df_labeled.drop(columns=["_time_temp"])

    # Add label as a new MultiIndex level
    df_labeled = df_labeled.set_index("_label_temp", append=True)
    df_labeled.index = df_labeled.index.rename(label_index_name, level=-1)

    return df_labeled

#################################
### POST-PROCESSING FUNCTIONS ###
#################################

def preprocess_results(df: pd.DataFrame, 
    channel: int, 
    start_time: Optional[str] = None, 
    end_time: Optional[str] = None
    ) -> pd.DataFrame:
    """Preprocess results by remapping/removing labels and filtering a MultiIndex DataFrame.

    This function filters a DataFrame for a given channel, remaps labels based on
    settings, optionally removes labels, and restricts the data to a given time interval.

    Args:
        df (pd.DataFrame): 
            DataFrame with a MultiIndex containing 'Label', 'channel', and 'time'.
        channel (int): 
            Channel number to keep (e.g., CHANNEL-1).
        start_time (str, optional): 
            Start time in format '%Y-%m-%d %H:%M:%S'. Defaults to None.
        end_time (str, optional): 
            End time in format '%Y-%m-%d %H:%M:%S'. Defaults to None.

    Returns:
        pd.DataFrame: 
            A filtered DataFrame with labels remapped/removed, restricted to the
            specified channel and time interval, and sorted by time.

    Raises:
        ValueError: If start_time or end_time is not in the format '%Y-%m-%d %H:%M:%S'.
    """
    df = df.copy()
    df = df[~df.index.get_level_values("Label").isna()]

    # -- Parsing dates si fournis
    start = None
    end = None
    if start_time:
        start = datetime.datetime.strptime(start_time, "%Y-%m-%d %H:%M:%S")
        start = start.replace(tzinfo=pytz.timezone(settings.data_acquisition.timezone))
    if end_time:
        end = datetime.datetime.strptime(end_time, "%Y-%m-%d %H:%M:%S")
        end = end.replace(tzinfo=pytz.timezone(settings.data_acquisition.timezone))

    # -- Filtrage par temps
    if start or end:
        times = df.index.get_level_values("time")
        mask = pd.Series(True, index=df.index)

        if start:
            mask &= times >= start
        if end:
            mask &= times <= end
        df = df[mask]

    # -- Remap des labels
    if settings.labels.label_removals:
        df = df[~df.index.get_level_values("Label").isin(settings.labels.label_removals)]  
    if settings.labels.label_replacements:      
        df.index = df.index.map(
            lambda idx: tuple(
                next((val for key, val in settings.labels.label_replacements.items() if key in str(x)), x)
                if name == "Label" else x
                for x, name in zip(idx, df.index.names)
            )
        )
        df.index = pd.MultiIndex.from_tuples(df.index, names=df.index.names)

    # -- Filtrage par channel
    df = df[df.index.get_level_values("channel") == str(channel)]

    # -- Tri final
    df = df.sort_index(level="time")

    return df

def display_spectrogram(
    fig: go.Figure,
    segments: List[pd.DataFrame],
    colorname: str = "magma",
    show_colorbar: bool = True
) -> go.Figure:
    """Plots consecutive segments of a spectrogram with blank areas for gaps."""
    
    for i, seg in enumerate(segments):
        df_agg = seg.groupby(['freq','time'])['bandleq'].mean().reset_index()
        pivot = df_agg.pivot(index='freq', columns='time', values='bandleq')

        color = cm.get_cmap(colorname, 256)
        colorscale = [
            [i/255, f"rgba{color(i)}"] for i in range(256)
        ]

        fig.add_trace(go.Heatmap(
            x=pivot.columns,
            y=pivot.index,
            z=pivot.values,
            colorscale=colorscale,
            zsmooth=False,
            showscale=(show_colorbar and i==0),
            colorbar=dict(
                title="Level (dB)",
                tickfont=dict(color="black"),
                title_font=dict(color="black"),
                orientation='h',
                x=0,
                y=1.015,
                len=0.6,
                thickness=20,
                xanchor='left',
                yanchor='bottom'
            ) if (show_colorbar and i==0) else None
        ))

    fig.update_yaxes(
        type = settings.visualization.frequency_scale,
        title_text=Y_LABEL_SPECTRA,  # unique label affiché pour le spectrogramme
        title_font_color="black",
        tickfont_color="black",
        # type= settings.visualization.frequency_scale
    )   
    
    return fig


def split_spectrograms(df: pd.DataFrame, 
    threshold_ms: float = 40.0
    ) -> List[pd.DataFrame]:
    """Splits a spectrogram DataFrame into consecutive segments based on time gaps.

    Consecutive rows are grouped into the same segment until the time difference
    between measurements for the same frequency exceeds the threshold.

    Args:
        df (pd.DataFrame): 
            DataFrame with at least 'freq' and 'time' columns.
        threshold_ms (float, optional): 
            Maximum allowed time gap in milliseconds between consecutive measurements
            within the same segment. Defaults to 40.0 ms.

    Returns:
        List[pd.DataFrame]: 
            A list of DataFrames, each representing a consecutive segment of the spectrogram.
    """
    df = df.reset_index()
    df['dt'] = df.groupby('freq')['time'].diff().dt.total_seconds() * 1000
    df['segment'] = (df['dt'] > threshold_ms).cumsum()
    segments = [g.drop(columns='dt') for _, g in df.groupby('segment')]
    return segments

def display_spectra(
    fig: go.Figure,
    df_bandleq: pd.DataFrame,
    unique_labels: List[str],
    colors: Dict[str, str],
    row: int = 1,
    col: int = 2,
    added_labels_legend: Optional[Set[str]] = None
    ) -> Tuple[go.Figure, Set[str]]:
    """Displays mean spectra for each label on a given subplot of a Plotly figure.

    Each unique label is plotted as a line of mean 'bandleq' values across frequencies.
    The legend is shown only once per label.

    Args:
        fig (go.Figure): 
            Plotly figure to which traces will be added.
        df_bandleq (pd.DataFrame): 
            DataFrame with MultiIndex including 'Label' and 'freq', and a column 'bandleq'.
        unique_labels (List[str]): 
            List of unique labels to plot.
        colors (Dict[str, str]): 
            Mapping from label to line color (hex or named color).
        row (int, optional): 
            Row of the subplot where the traces will be added. Defaults to 1.
        col (int, optional): 
            Column of the subplot where the traces will be added. Defaults to 2.
        added_labels_legend (Set[str], optional): 
            Set of labels already added to the legend to avoid duplicates. Defaults to None.

    Returns:
        Tuple[go.Figure, Set[str]]: 
            The updated Plotly figure and the updated set of labels added to the legend.
    """
    if added_labels_legend is None:
        added_labels_legend = set()
    for label in unique_labels:
        df_label = df_bandleq[df_bandleq.index.get_level_values('Label') == label]
        if df_label.empty: 
            continue
        mean_spectrum = df_label.groupby('freq')['bandleq'].mean().sort_index()
        show_leg = label not in added_labels_legend
        fig.add_trace(go.Scatter(
            x=mean_spectrum.values, y=mean_spectrum.index,
            mode='lines', name=label,
            legendgroup=label, showlegend=show_leg,
            line=dict(color=colors.get(label,'#666'))
        ), row=row, col=col)
        added_labels_legend.add(label)

    # retirer yticks (sharey avec col=1) mais garder xticks pour le mean spectra
    fig.update_yaxes(
        matches = f'y{(row-1)*2+1}',
        title_text="",
        showticklabels=True if (row == 0 and col == 0) else False,
        tickfont=dict(color="black"),
        row=row,
        col=col
    )
    
    fig.update_xaxes(
        title_text="Level (dB)",
        title_font_color="black",   # couleur du titre
        title_standoff=0,          # distance entre le titre et les ticks
        side='top',
        tickfont=dict(color="black"),
        showticklabels=True, 
        row=row, 
        col=col
    )

    return fig, added_labels_legend


def display_features(
    fig: go.Figure,
    df_features: pd.DataFrame,
    feature_list: List[str],
    unique_labels: List[str],
    threshold: float,
    colors: Dict[str, str],
    start_row: int = 2,
    added_labels_legend: Optional[Set[str]] = None
) -> Tuple[go.Figure, Set[str]]:
    """Displays multiple feature time series in a Plotly figure for each label.

    Each feature is plotted in its own row, with lines for each label.
    Values are set to NaN where the time difference between consecutive measurements
    exceeds the threshold. The background label is drawn in black.

    Args:
        fig (go.Figure):
            Plotly figure to which traces will be added.
        df_features (pd.DataFrame):
            DataFrame with a MultiIndex including 'Label' and 'time', and columns for features.
        feature_list (List[str]):
            List of feature column names to plot.
        unique_labels (List[str]):
            List of labels to display.
        threshold (float):
            Maximum allowed time gap in milliseconds. Values exceeding this are replaced by NaN.
        colors (Dict[str, str]):
            Mapping from label to line color.
        start_row (int, optional):
            Row number to start plotting features. Defaults to 2.
        added_labels_legend (Set[str], optional):
            Set of labels already added to the legend to avoid duplicates. Defaults to None.

    Returns:
        Tuple[go.Figure, Set[str]]:
            Updated Plotly figure and updated set of labels added to the legend.
    """
    if added_labels_legend is None:
        added_labels_legend = set()

    # Groupes par label
    groups = dict(tuple(df_features.groupby(level="Label")))

    for i, feature in enumerate(feature_list, start=start_row):
        # --- Courbes par label ---
        for label in unique_labels:
            df_label = groups.get(label)
            if df_label is None or df_label.empty:
                continue

            time = df_label.index.get_level_values('time')
            dt = time.to_series().diff().dt.total_seconds().to_numpy() * 1000
            y_values = np.where(dt > threshold, np.nan, df_label[feature].to_numpy())

            show_legend = label not in added_labels_legend
            color = 'black' if label == settings.labels.label_background else colors.get(label, '#666')

            fig.add_trace(
                go.Scattergl(
                    mode='lines+markers',  # <-- ajoute les markers en plus des lignes
                    marker=dict(symbol='circle', size=10),  # personnalisation des points
                    line=dict(color=color),
                    name=label,
                    legendgroup=label,
                    showlegend=show_legend
                ),
                hf_x=time,
                hf_y=y_values,
                row=i, col=1
            )
            added_labels_legend.add(label)

        fig.update_yaxes(
            type = 'linear',
            title=dict(text=feature, font=dict(color="black")),
            tickfont=dict(color="black"),
            row=i,
            col=1
        )

        fig.update_xaxes(
            tickfont=dict(color="black"),
            row=i,
            col=1
        )

        fig.update_layout(
            legend=dict(
                font=dict(color="black")
            )
)

    return fig, added_labels_legend

def display_boxplots(
    fig: go.Figure,
    df_features: pd.DataFrame,
    feature_list: List[str],
    colors: Dict[str, str],
    start_row: int = 2
    ) -> go.Figure:
    """Displays boxplots of feature values per label on a Plotly figure.

    Each feature is plotted in its own row. Values are aggregated by time and
    separated by label. Boxplots show outliers and are colored according to the given mapping.

    Args:
        fig (go.Figure):
            Plotly figure to which boxplots will be added.
        df_features (pd.DataFrame):
            DataFrame with MultiIndex including 'Label' and 'time', and feature columns.
        feature_list (List[str]):
            List of feature column names to plot.
        colors (Dict[str, str]):
            Mapping from label to boxplot color.
        start_row (int, optional):
            Row number to start plotting features. Defaults to 2.

    Returns:
        go.Figure:
            Updated Plotly figure with boxplots added.
    """
    for i, feature in enumerate(feature_list, start=start_row):
        selected_data = df_features.reset_index().pivot_table(
            index='time', columns='Label', values=feature, aggfunc='mean'
        )
        for col in selected_data.columns:
            fig.add_trace(go.Box(
                y=selected_data[col].values,
                name=col,
                legendgroup=col,
                marker_color=colors.get(col,'#666'),
                boxpoints='outliers',
                opacity=0.6,
                showlegend=False
            ), row=i, col=2)


        fig.update_yaxes(
            matches=f'y{(i-1)*2+1}',
            title="",
            showticklabels=False,
            tickfont=dict(color="black"),
            row=i,
            col=2
        )
        fig.update_xaxes(tickangle=45,
                         tickfont=dict(color="black"),
                         row=i, 
                         col=2)
    return fig



