#%%
# ==========================
# Imports standard Python
# ==========================
import os
import re
import logging
import datetime
from zoneinfo import ZoneInfo
from pathlib import Path

# ==========================
# Imports tiers
# ==========================
import streamlit as st
from streamlit_jupyter import StreamlitPatcher
import pandas as pd

# ==========================
# Imports du projet / locaux
# ==========================
from wavely.datasets_analysis.preprocessing.metadata import audio_file_info
from wavely.eda.conf import settings
from wavely.eda import utils


import logging
# Logger principal
logger = logging.getLogger(__name__)
logging.getLogger("pydub.converter").setLevel(logging.WARNING)
logging.basicConfig(level=logging.INFO)
logging.basicConfig(level=logging.DEBUG)
# TODO : Ajouter les tqdm dans streamlit
# Ajouter INFINEON dans wavely/signal/settings.yaml



DEBUG = False


# =====================
# ==== USER INPUTS ====
# =====================


if DEBUG:
    root_dir = "/media/rbeauvais/Elements/romainb/test-eda-template"
    start_time_utc_str = None
    end_time_utc_str = None
    sensor = "OTHER"
    sensor_id = ""
    date = ""   
    channel = 1 
    config = {
        "data_selection": {
            "OTHER": {
                "sensor_ids": [
                    {
                        "sensor_id": "",
                        "dates": [""],
                        "calibrate": False,
                        "time_shift_ms": 0.0,
                        "device": None,
                        "rss_calibration": 0.0,
                        "normalise_gain": 0.0,
                        "device_name": "",
                        "channel_prefix": "",
                        "microphone": [],
                        "preamp_gain": [],
                        "mic_calibration": []
                    }
                ]
            }
        }
    }
    start_time_utc = None
    end_time_utc = None
    import sys
    import yaml
    # Dossier parent contenant 'wavely'
    project_root = Path(root_dir).resolve()
    sys.path.insert(0, str(project_root))

    # Maintenant Python peut trouver wavely.eda
    from wavely.eda.conf import Settings

    # Chemin vers ton YAML
    yaml_path = project_root / "wavely" / "eda" / "settings.yaml"

    # Charger le YAML
    with open(yaml_path, "r") as f:
        yaml_data = yaml.safe_load(f)

    # Instancier Settings avec le YAML
    settings = Settings(**yaml_data)
    from wavely.eda.conf import Settings
    from wavely.eda import utils

else:

    # =====================
    # ==== USER INPUTS ====
    # =====================
    StreamlitPatcher().jupyter()


    # --- Titre principal ---
    st.title("Pre-processing of raw recordings")

    st.markdown(
        f"""
        ## About this page

        This page allows you to **format and prepare your raw recordings** for feature extraction.  
        The process includes the following steps:

        - **Decompression** of the original audio files  
        - **Creation of a `metadata.db` database** for recordings coming from devices other than **MAGNETO**  
        - **Creation of a subfolder**  `formatted_data`
            This folder will include:  
        1. **Concatenation of recordings into 30-minute sessions** (files named `YYYY-MM-DD hh:mm:ss.wav`) from the present compilation,  
        in order to simplify the **annotation process** (You can use the [Recording Time Label plugin for Audacity (by Steve Daulton)](https://forum.audacityteam.org) for assisted labeling)  
        2. **Creation of a `formatted_data/labels` subfolder**, which contains subdirectories named:  
        ```
        YYYY-MM-DD hh:mm:ss_data
        ```
        Each subdirectory is linked to its corresponding `.wav` file as in the Audacity formalism.  
        Labels exported from **Audacity** go there under the filename:  
        ```
        Label 1.txt
        ```

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

if DEBUG == False:
    # ---- Bouton de validation ----
    if st.button("Valider la sélection"):
        st.write("### Résumé de la sélection")
        st.write(f"**Sensor:** {sensor}")
        st.write(f"**Sensor ID:** {sensor_id if sensor_id else 'aucun'}")
        st.write(f"**Date:** {date if date else 'aucune'}")

        logger.info("Compilation started")

        root_dir = settings.root_dir
        timezone = settings.data_acquisition.timezone


        # FAIRE LISTE DE CHOIX SENSOR/SENSOR_ID puid DATE
        selected_configuration = utils.select_configuration(settings.data_selection, sensor, sensor_id, date)
        if not selected_configuration:
            msg = f"Non-existing data path. Check path values: sensor_name={sensor}, sensor_id={sensor_id}, date={date}, "
            logger.error(msg)
            raise ValueError(msg)


        # =====================
        # ==== COMPUTATION ====
        # =====================
        os.chdir(root_dir)
        logger.info(f"Root directory: {root_dir}")
        data_path = os.path.join('data', sensor, sensor_id, date)
        logger.info(f"Data directory: {data_path}")
        # Generate metadata.db
        if 'MAGNETO' not in sensor:
            source_path = os.path.join(data_path,settings.source_db)
            db_path = os.path.join(data_path,settings.source_db+'.tmp')
            logger.info(f"Temporary metadata file path: {db_path}")
            if os.path.exists(db_path):
                os.remove(db_path)
            logger.info(f"Final metadata file path: {source_path}")
            table_columns = utils.copy_db_structure(source_path, db_path)
            utils.populate_db(selected_configuration,sensor,db_path, table_columns,data_path)
            os.replace(db_path, source_path)

    
        metadata_path = os.path.join(data_path, settings.source_db)
        if start_time_utc is not None and date:
            start_time = datetime.datetime.strptime(date + ' ' + start_time_utc, '%Y-%m-%d %H:%M:%S')
            start_time = start_time.replace(tzinfo=ZoneInfo(timezone))
        else:
            start_time = None
        if end_time_utc is not None and date:
            end_time = datetime.datetime.strptime(date + ' ' + end_time_utc, '%Y-%m-%d %H:%M:%S')
            end_time = end_time.replace(tzinfo=ZoneInfo(timezone))
        else:
            end_time = None

        if sensor == 'MAGNETO':
            selected_configuration['channel_prefix'] = None

        utils.decompress_and_remove_files(data_path)

        
        wav_files = [
            f for f in Path(metadata_path).parent.rglob('*.wav')
            if 'formatted_data' not in f.parts
        ]
        wav_info = []
        for wav_file in wav_files:
            info = audio_file_info(wav_file)

            wav_info.append({
                "filename": wav_file.name,
                "path": wav_file.as_posix(),
                "sample_rate": info.sample_rate,
                "n_channels": info.n_channels,
                "nframes": info.nframes,
                "duration": info.duration,
                "dtype": info.dtype,
                "subtype": info.subtype,
                "start_time": info.start_time,
                "end_time": info.end_time,
                "file_size": info.file_size
            })
        df_wav_info = pd.DataFrame(wav_info)


        output_file = Path(metadata_path).parent / "audio_files_info.csv"
        if output_file.exists():
            logger.info(f"The file '{output_file.name}' already exists — no export performed.")
        else:
            df_wav_info.to_csv(output_file, index=False)
            logger.info(f"WAV informations exported in CSV to: {output_file}")


        recordings, db_url, start_times = utils.get_recordings_from_time_range(metadata_path, start_time, end_time,sensor,selected_configuration)

        channels = set()
        if selected_configuration['channel_prefix']: 
            for path in recordings:
                match = re.search(rf"{selected_configuration['channel_prefix']}(\d+)", path)
                if match:
                    channels.add(f"{selected_configuration['channel_prefix']}{int(match.group(1))}")  # convertir en int si tu veux
        else:
            channels = ['']

        for channel in channels:
            output_dir = os.path.join(data_path,'formatted_data',channel)
            os.makedirs(output_dir, exist_ok=True)
            if selected_configuration['channel_prefix']: 
                filtered_recordings = [r for r in recordings if channel in r]
            else:
                filtered_recordings = recordings
                
            utils.concatenate_wav_files(output_dir,filtered_recordings, start_times,selected_configuration)
        logger.info("Compilation completed.")
        st.success(
            f"Compilation completed successfully. You may now close this page.  \n If needed, continue with the annotation process in the following data folder: {os.path.join(root_dir,data_path, 'formatted_data')}"
        )


#%%