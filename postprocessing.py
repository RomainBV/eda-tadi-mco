#%%
# -----------------------
# Standard library imports
# -----------------------
import os
import math
import logging
from pathlib import Path

# -----------------------
# Third-party imports
# -----------------------
import pandas as pd

from plotly.subplots import make_subplots
from plotly_resampler import FigureResampler
import plotly.express as px


import streamlit as st
from streamlit_jupyter import StreamlitPatcher
# -----------------------
# Local / project imports
# -----------------------
from wavely.eda.conf import settings
from wavely.eda import utils
from wavely.datasets_analysis.units.helpers import read_hdf_dataset

# -----------------------
# Logging configuration
# -----------------------
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)
logging.getLogger("pydub.converter").setLevel(logging.WARNING)




# =====================
# ==== USER INPUTS ====
# =====================


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
    from wavely.datasets_analysis.units.helpers import read_hdf_dataset

else:
    StreamlitPatcher().jupyter()


    # --- Titre principal ---
    st.title("Post-Processing : Vizualisation")

    st.markdown(
        f"""
        ## About this page

        This page allows you to **visualize results** in two complementary forms:

        ### Temporal representation
        - Displays **spectrograms and selected features over time**.  
        **Nota Bene:** This view is ideal for **micro-analysis** of a limited dataset (e.g., a single sensor with one ID on a specific date) to inspect the evolution of features in detail.

        ### Static representation
        - Shows **histograms and PCA projections** with **SVM-based classification**, optimized via the **ROC curve**.  
        **Nota Bene:** This view is intended for **macro-analysis** on a complete dataset, providing statistical insights across multiple sensors or sessions.

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
    selected_configuration = utils.select_configuration(settings.data_selection, sensor, sensor_id, date)

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

def computation(settings,sensor,sensor_id,date,channel):
    raw_signals_dir = Path(settings.raw_signals_dir)
    datasets_name = Path(settings.datasets_dir)
    root_dir = settings.root_dir
    aggregation_sliding_window = settings.features.aggregation_sliding_window
    os.chdir(root_dir)
    print(root_dir)

    #########################
    ### FEATURES FUNCTION ###
    #########################

    data_path = os.path.join(raw_signals_dir,sensor,sensor_id,date)

    device_config = getattr(settings.data_selection, sensor, None)
    if device_config is None or not device_config.sensor_ids:
        channel_prefix = None
    else:
        sensor_obj = device_config.sensor_ids[0]
        channel_prefix = getattr(sensor_obj, "channel_prefix", None) or None

    if channel_prefix is not None:
        datasets_dir = data_path / datasets_name / f"ch{channel}"
    else:
        datasets_dir = data_path / datasets_name

    features = read_hdf_dataset(datasets_dir.joinpath("features.h5"))

    bandleq = read_hdf_dataset(datasets_dir.joinpath("bandleq.h5"))


    df_features = pd.DataFrame(features['features'])

    st.title("Features selection")


    # ------------------------------------------
    # Extract unique features après validation
    # ------------------------------------------
    features = []
    for col in df_features.columns:
        for ext in ["_mean", "_std", "_min", "_max"]:
            if col.endswith(ext):
                col = col.replace(ext, "")
        if col not in features:
            features.append(col)

    features = sorted(features)
    # ------------------------------------------
    # Multi-selection of features
    # ------------------------------------------
    st.write("### Select features:")
    selected_features = st.multiselect("Available features:", features)

    # If nothing selected, use all features
    if not selected_features:
        selected_features = features

    # ------------------------------------------
    # Multi-selection of extensions
    # ------------------------------------------
    st.write("### Select extensions:")
    extension_options = ["mean", "std", "min", "max"]
    selected_exts = st.multiselect("Extensions:", extension_options)

    # If nothing selected, use all extensions
    if not selected_exts:
        selected_exts = extension_options

    # ------------------------------------------
    # Generate selected columns
    # ------------------------------------------
    dataset_selection = []

    for feat in selected_features:
        for ext in selected_exts:
            colname = f"{feat}_{ext}"
            if colname in df_features.columns:   # Check if it really exists
                dataset_selection.append(colname)

    # ------------------------------------------
    # Display result
    # ------------------------------------------
    st.write("### Selected columns:")
    if dataset_selection:
        st.success(dataset_selection)
    else:
        st.info("No columns exist for the selected combination.")

    df_bandleq = pd.DataFrame(bandleq['bandleq'])

    if sensor == 'MAGNETO':
        channel = channel-1
    else:
        channel = 0


    df_features = utils.preprocess_results(df_features, channel,start_time_utc,end_time_utc,dataset_selection)
    df_bandleq = utils.preprocess_results(df_bandleq, channel,start_time_utc,end_time_utc,None)

    return df_features,df_bandleq,aggregation_sliding_window,datasets_dir

df_features,df_bandleq,aggregation_sliding_window,datasets_dir = computation(settings,sensor,sensor_id,date,channel)
    
if not DEBUG:
    # ---- Bouton de validation ----
    if st.button("Valider la sélection"):

        logger.info(f"features and bandleq loaded from : {datasets_dir}" )
        st.write("### Résumé de la sélection")
        st.write(f"**Sensor:** {sensor}")
        st.write(f"**Sensor ID:** {sensor_id if sensor_id else 'aucun'}")
        st.write(f"**Date:** {date if date else 'aucune'}")

        # === 2. Couleurs ===
        unique_labels = df_features.index.get_level_values(level='Label').unique()

        palette = px.colors.qualitative.Plotly
        colors = {}

        for i, label in enumerate(unique_labels):
            if label == 'other':
                colors[label] = 'rgba(0,0,0,0.1)'  # presque invisible
            else:
                colors[label] = palette[i % len(palette)]

        time = df_features.index.get_level_values('time')
        dt = time.to_series().diff().dt.total_seconds() * 1000
        dt_usual = dt.round().median()
        threshold = 2*dt_usual
        segments = utils.split_spectrograms(df_bandleq, threshold_ms=settings.features.aggregation_sliding_window*1000)

        feature_list = list(df_features.columns)

        # Découper feature_list en sous-listes de taille "features_count_per_plot"
        num_groups = math.ceil(len(feature_list) / settings.visualization.features_count_per_plot)
        feature_groups = [feature_list[i* settings.visualization.features_count_per_plot:(i+1)* settings.visualization.features_count_per_plot] for i in range(num_groups)]

        # Boucle sur chaque groupe pour créer une figure séparée
        for group_features in feature_groups:

            # Création de la grille
            base_fig = make_subplots(
                rows=len(group_features)+1,
                cols=2,
                shared_xaxes=True,
                shared_yaxes=False,
                vertical_spacing=0.02,
                column_widths=[0.67, 0.33],   # 2/3 gauche, 1/3 droite
            )
            total_height = (len(group_features) + 1)  * settings.visualization.base_plot_height 
            base_fig.update_layout(height=total_height)
            fig = FigureResampler(base_fig)
            # --- Spectrogrammes ---
            fig = utils.display_spectrogram(fig, segments)

            # --- Spectres moyens ---
            fig, added_labels_legend = utils.display_spectra(fig, df_bandleq, unique_labels, colors, row=1, col=2)
            
            # --- Courbes features pour ce groupe ---
            fig, added_labels_legend = utils.display_features(
                fig, df_features, group_features, unique_labels, threshold, colors, start_row=2, added_labels_legend=added_labels_legend
            )

            # --- Boxplots features pour ce groupe ---
            fig = utils.display_boxplots(fig, df_features, group_features, colors, start_row=2)

            # --- Layout final avec super-titre et légende ---
            extension_title = f" - aggreg. window ({settings.features.aggregation_sliding_window} sec.)"

            fig.update_layout(
                title=dict(
                    text=f"<b>{settings.campaign} / {settings.project} / {sensor} ({sensor_id}) </b>{extension_title}",
                    x=0.5, 
                    xanchor='center',
                    font=dict(color="black")  # <-- titre en noir
                ),
                width=1200,
                height=300*(len(group_features)+1),
                hovermode="x unified",
                margin=dict(l=60,r=60,t=120,b=50),
            )


            fig.update_layout(template="plotly_white")  # même rendu partout

            fig.update_layout(
                plot_bgcolor="white",  
                paper_bgcolor="white",  # <-- virgule ajoutée
            )
            
            st.plotly_chart(
                fig,
                use_container_width=True,
                key=f"{sensor_id}_{group_features}"
            )
            

            logger.info(f"{settings.campaign}_{settings.project}_{sensor}_{sensor_id}{extension_title}_ {group_features}")


    # %%
