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

# TODO :voir s'il n'existe pas un méthode permettant de ne conserver pour la pca que les indicateurs pertiernts

### TODO : Permettre l'analyse sur plusieurs jeux  e de données pour l'analyse PCA
### TODO : PAppliquer le K-fold pour faire des train set et des tests sets


# =====================
# ==== USER INPUTS ====
# =====================


DEBUG = False
SELECTED_FEATURES = ['spectralcentroid','ultrasoundlevel','leak_expert','leak_ml']

# SELECTED_FEATURES = ['spectralcentroid','ultrasoundlevel','leak_expert']
# SELECTED_FEATURES = None

DISPLAY_FEATURES = True
# =====================
# ==== USER INPUTS ====
# =====================
EXTENSIONS = ['_mean']

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

    datasets_dir = data_path / datasets_name / f"ch{channel}"

    features = read_hdf_dataset(datasets_dir.joinpath("features.h5"))
    logger.info(f"features loaded from : {datasets_dir}")
    bandleq = read_hdf_dataset(datasets_dir.joinpath("bandleq.h5"))
    logger.info(f"bandleq loaded from : {datasets_dir}")



    df_features = pd.DataFrame(features['features'])
    if SELECTED_FEATURES:
        selected_features = [feat + EXTENSIONS[0] for feat in SELECTED_FEATURES]
        df_features = df_features[selected_features]
    df_bandleq = pd.DataFrame(bandleq['bandleq'])

    if sensor == 'MAGNETO':
        channel = channel-1
    else:
        channel = 0


    df_features = utils.preprocess_results(df_features, channel,start_time_utc,end_time_utc)
    logger.info(f"features processed")
    df_bandleq = utils.preprocess_results(df_bandleq, channel,start_time_utc,end_time_utc)
    logger.info(f"bandleq  processed")

    return df_features,df_bandleq,aggregation_sliding_window

df_features,df_bandleq,aggregation_sliding_window = computation(settings,sensor,sensor_id,date,channel)


    
if not DEBUG:
    # ---- Bouton de validation ----
    if st.button("Valider la sélection"):
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


        if DISPLAY_FEATURES:
            time = df_features.index.get_level_values('time')
            dt = time.to_series().diff().dt.total_seconds() * 1000
            dt_usual = dt.round().mode().iloc[0]
            threshold = 2*dt_usual
            segments = utils.split_spectrograms(df_bandleq, threshold_ms=settings.features.aggregation_sliding_window*1000)

            for extension in EXTENSIONS :

                feature_list = list(df_features.columns)
                if extension == "":
                    # garder les colonnes originales (pas "_mean" ni "_std")
                    feature_list = [feat for feat in feature_list if "_mean" not in feat and "_std" not in feat]
                    continue
                else:
                    # garder uniquement les colonnes contenant l'extension
                    feature_list = [feat for feat in feature_list if extension in feat]

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
                    if extension == '_mean':
                        extension_title = f" - rolling mean ({settings.features.aggregation_sliding_window} sec.)"
                    elif extension == '_std':
                        extension_title = f" - rolling std ({settings.features.aggregation_sliding_window} sec.)"
                    else:
                        extension_title = ""

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
                        key=f"{sensor_id}_{group_features}_{extension}"
                    )
                 

                    logger.info(f"{settings.campaign}_{settings.project}_{sensor}_{sensor_id}{extension_title}_ {group_features}")


    # %%
