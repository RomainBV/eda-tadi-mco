#%%
# ==========================
# Imports standard Python
# ==========================
import argparse
import os
import glob
from pathlib import Path
import logging
import datetime
import warnings
import pytz
import pandas as pd

# ==========================
# Imports tiers
# ==========================
import streamlit as st
from streamlit_jupyter import StreamlitPatcher
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker


# ==========================
# Imports du projet / locaux
# ==========================
from wavely.datasets_analysis import initialized_metadata_db
from wavely.datasets_analysis.units.helpers import to_hdf_dataset
from wavely.eda.conf import settings
from wavely.eda import utils
from wavely.metadata.edge_metadata import load 
from wavely.edge_metadata.models import Label, Recording



# ==========================
# Initialisations
# ==========================

# Ignorer les warnings
warnings.filterwarnings('ignore')

# Initialiser la DB metadata
initialized_metadata_db()

# Logger principal
logger = logging.getLogger(__name__)
logging.getLogger("pydub.converter").setLevel(logging.WARNING)
logging.basicConfig(level=logging.INFO)
logging.basicConfig(level=logging.DEBUG)

# TODO : Penser à refactorer signal.preprocessing.preprocessing make_butter_filter
# Ajouter les tqdm dans streamlit
# VOIR COMMENT TRAITER LES RECOUVREMENTS DE LABELS
# AJOUTER LEAK DETECTION
# Faire Notebook 
# Faire fonction conf.py avec pydanctic V2
# Gérer les metadata via le dernier repo metadata et non edge_metadata (pour gérer les timestamps en milliseconde)


DEBUG = False


# =====================
# ==== USER INPUTS ====
# =====================


if DEBUG:
    start_time_utc_str = None
    end_time_utc_str = None
    sensor = "MAGNETO"
    sensor_id = "RCA-08"
    date = "2025-10-10"   
    channel = 1 
    config = {
        "data_selection": {
            "MAGNETO": {
                "sensor_ids": [
                    {
                        "sensor_id": "RCA-08",
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
    st.title("Processing : Features compilation")

    st.markdown(
        f"""
        ## About this page

        This page allows you to **compute features and spectrograms (BandLeq)**.  
        These computations rely on the settings provided in the `settings.yaml` file, including:

        ### Necessary information
        - **Feature settings**: `block_duration`, `block_overlap`, `aggregation_sliding_window`, `features_list`  

        ### Optional information
        - **Filter configuration**: `apply_filter_config`  
        - **Signal calibration**: `calibrate`  
        - **Data acquisition**: `resampling_rate`, `timezone`  
        - **Labelization**: `label_replacements`, `label_removals`, `label_background`  

        The process includes the following steps:

        - **Feature & BandLeq computation**  
        - **Rolling window application** on aggregated features  
        - **Time-domain downsampling of the BandLeq** for reasonable export and visualization  
        - **Export of the Features & BandLeq** in `.h5` format in a `results` subdirectory  

        ## User inputs
        """
    )


    # --- Texte explicatif ---
    data = utils.config_to_dict(settings.data_selection)

    
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
selected_configuration = utils.select_configuration(settings.data_selection, sensor, sensor_id, date)


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
    # labels = settings.labels
    filter_params = settings.apply_filter_config
    features_computer_params = settings.features.features_computer_kwargs.dict()
    if features_computer_params.get("band_freq") == "third":
        features_computer_params["block_duration"] = 0.125

    block_overlap = settings.features.block_overlap
    signal_calibration = {"calibrate": selected_configuration['calibrate']}
    params = {
        **filter_params,
        **features_computer_params,
        **signal_calibration,
    }

    data_path = os.path.join(raw_signals_dir,sensor,sensor_id,date)
    metadata_path = os.path.join(data_path,settings.source_db)

    labels_path = os.path.join(data_path, "formatted_data", "labels")
    df_labels = utils.group_audacity_labels(labels_path, settings.audacity_label_name, timezone)
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
        labels_path = os.path.join(data_path, "formatted_data", "labels")
        df_labels = utils.group_audacity_labels(labels_path, settings.audacity_label_name, timezone)

        if not df_labels.empty or not os.path.exists(labels_path):
            utils.write_labels_to_csv(df_labels, data_path)

    if not os.path.exists(metadata_path):
        raise FileNotFoundError(f"Non existing path : {metadata_path}")
    db_url = f"sqlite:///{metadata_path}"
    engine = create_engine(db_url, echo=False)

    if not DEBUG:
        SessionLocal = sessionmaker(bind=engine)

        # Ajouter les colonnes si elles n'existent pas
        with engine.connect() as conn:
            result = conn.execute('PRAGMA table_info(Label);')
            existing_columns = [row[1] for row in result]
            if 'start_time' not in existing_columns:
                conn.execute('ALTER TABLE Label ADD COLUMN start_time TEXT NOT NULL DEFAULT "";')
            if 'end_time' not in existing_columns:
                conn.execute('ALTER TABLE Label ADD COLUMN end_time TEXT NOT NULL DEFAULT "";')

        # Réinitialiser et remplir la table
        with SessionLocal() as session:
            # Supprimer toutes les entrées
            session.query(Label).delete()
            session.commit()

            # Réinitialiser l'auto-increment (séquence) à 1
            with engine.connect() as conn:
                conn.execute('DELETE FROM sqlite_sequence WHERE name="Label";')

            # Ajouter les nouvelles entrées
            label_entries = [
                utils.Label(
                    name=str(row["Label"]),
                    start_time=str(row["Start"]),
                    end_time=str(row["End"])
                )
                for _, row in df_labels.iterrows()
            ]
            session.add_all(label_entries)
            session.commit()


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

        metadata,db_url,_,_ = utils.group_recordings(metadata_path,start,end,sensor,selected_configuration)

        recordings = [os.path.join(data_path,meta.filename) for meta in metadata]
        all_files = glob.glob(os.path.join(data_path, "**", "*"), recursive=True)

        # Filtrer uniquement les .wav ou .WAV (hors formatted data)
        wav_files = [
            f for f in all_files
            if f.lower().endswith(".wav") and not "formatted_data" in str(f).lower()
        ]
        recordings = list(set(wav_files).intersection(set(recordings)))
        recordings = sorted(recordings, key=utils.sort_key)
        
        fullpaths = [os.path.join(root_dir,path) for path in recordings]

        # --- 5. Insertion des Labels dans metadata.db ---
        metadata_fullpath = Path(os.path.join(root_dir,metadata_path))

        filenames_root_dir = metadata_fullpath.parent
        
        load(
            client_name=settings.campaign,
            project_name=settings.project,
            edge_metadata_url=f"sqlite:///{metadata_fullpath.resolve()}",
            filenames_root_dir=filenames_root_dir,
        )
        
        dataset_1d,dataset_2d = utils.build_dataset(
            filenames_root_dir,
            fullpaths,
            block_overlap,
            params,
            df_labels,
            start,
            end,
        )

    if settings.features.leak_models.expert_model.enabled:
        result = utils.apply_expert_model(dataset_1d,filenames_root_dir)

          
    dataset_1d = utils.compute_stat_features(dataset_1d, window_seconds=aggregation_sliding_window)  
    logger.info(f"Rolling mean/std ({aggregation_sliding_window} s) applied on the 'feature' dataset")


    if settings.features.leak_models.expert_model.enabled:
        dataset_1d = dataset_1d.copy()
        dataset_1d["leak_expert_mean"] = result.reindex(dataset_1d.index.get_level_values('time'), method='ffill')["leak_expert_mean"].values

    dataset_2d = utils.downsample_bandleq(dataset_2d, window_seconds=aggregation_sliding_window)  
    logger.info(f"Time downsampling to {aggregation_sliding_window} s applied on the 'bandleq' dataset")

    root_dirname = filenames_root_dir / 'results' / f"ch{channel}"

    os.makedirs(root_dirname, exist_ok=True)

    to_hdf_dataset(
        dataframes = {'features' : dataset_1d},
        root_dirname = root_dirname,
        output_name = "features.h5",
    )
    logger.info(f"'features' dataset saved at : {root_dirname}")
    to_hdf_dataset(
        dataframes = {'bandleq' : dataset_2d},
        root_dirname = root_dirname,
        output_name = "bandleq.h5",
    )
    logger.info(f"'bandleq' dataset saved at : {root_dirname}")
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
