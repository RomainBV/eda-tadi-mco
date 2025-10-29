#%%
import pandas as pd
import os
import plotly.graph_objects as go
from ipywidgets import widgets
from IPython.display import display
from plotly.subplots import make_subplots
import numpy as np
from typing import  Dict, List
from pathlib import Path
from sklearn.linear_model import LogisticRegression
import re

root_dir = "/media/rbeauvais/Elements/romainb/2025-n09-TADI-MCO"
os.chdir(root_dir)
print(root_dir)


start_str = '2025-10-10'
end_str = '2025-10-11'
leak_configurations = "TADI - All datasets analyses.xlsx"
sensors = ['RDA-04','RCA-07','RCA-08','RCA-04','RCA-09','RCA-06','ROGUE4']
results = 'TADI_MCO-2025_Sound-levels_Leak-detection.xlsx'
data = {
    "Public ID": ["RDA-04", "RCA-07", "RCA-08", "RCA-04", "RCA-09", "RCA-06", "ROGUE4"],
    "sensor ID": ["e4-5f-01-8f-6f-95", "e4-5f-01-8f-6f-04", "e4-5f-01-8f-6d-ab",
                  "e4-5f-01-8f-6f-fb", "e4-5f-01-8f-6f-9c", "e4-5f-01-8f-72-89", "e4-5f-01-8f-6f-09"],
    "x (m)": [-0.3, -0.3, 20.0, 40.2, 40.2, 21.5, 28.4],
    "y (m)": [28.9, -0.3, -0.3, -0.3, 29.0, 55.5, 11.4],
    "z (m)": [3, 3, 3, 3, 3, 3, 5.45]
}
channel = 1
min_emergence = 10

# A retirer pour plus
Tests_correspondances = {'other' : None, 
                         'leak_Q251_D3' : 1, 
                         'leak_Q84_D3' : 2 , 
                         'leak_Q42_D3' : 3, 
                         'leak_Q502_D5' : 4 ,
                         'leak_Q335_D5' : 5, 
                         'leak_Q168_D5' : 6, 
                         'alarme' : None, 
                         'leak_Q461_D6' : 7,
                         'leak_Q184_D10' : 8, 
                         'purge' : None}


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

def remove_jumps(group, threshold):
    times = group.index.get_level_values('time')
    time_diff = times.to_series().diff().fillna(pd.Timedelta(seconds=0))
    
    # Index des sauts
    jump_positions = time_diff[time_diff > threshold].index
    
    to_drop = set()
    locs = group.index.to_series().index  # positions dans le DataFrame
    
    for idx in jump_positions:
        loc = locs.get_loc(idx)
        start = max(loc - 2, 0)        # au moins la première ligne
        end = min(loc + 3, len(group)) # +3 car end exclu
        to_drop.update(group.iloc[start:end].index)
    
    # Ajouter les 2 premières et 2 dernières lignes du groupe
    if len(group) > 0:
        to_drop.update(group.iloc[:2].index)
        to_drop.update(group.iloc[-2:].index)
    
    return group.drop(to_drop)

def plot_detection_vs_distance(df_det_grouped, df_config, Tests_correspondances):
    """
    Trace les nuages ML/Expert vs distance et vitesse d’éjection,
    calcule la limite de détection à 50 % (logistique),
    et renvoie deux tableaux :
      - table_theo : distance limite (m) par vitesse (0→700 m/s, pas de 100)
      - table_real : distance limite (m) par test réel (Qm, D, vitesse)
    """

    # === Fonction utilitaire pour un modèle (ML ou Expert) ===
    def fit_and_plot(df, col_value, col_idx):
        scatter = go.Scatter(
            x=df['distance_m'],
            y=df['ejection_speed'],
            mode='markers',
            marker=dict(
                size=12,
                color=df[col_value],
                colorscale=[[0, '#FF0000'], [1, '#00FF00']],
                cmin=0, cmax=100,
                colorbar=dict(title=f"{col_value} (%)") if col_idx == 1 else None
            ),
            text=df['sensor_clean'],
            hovertemplate=f"Sensor: {{text}}<br>{col_value}: %{{marker.color:.1f}}%<br>"
                          "Distance: %{x:.1f} m<br>Ejection: %{y:.1f} m/s",
            showlegend=False
        )

        # Classification binaire (>=50 %)
        y = (df[col_value] >= 50).astype(int)
        X = df[['distance_m', 'ejection_speed']].values

        clf = LogisticRegression()
        clf.fit(X, y)

        # Frontière
        x_vals = np.linspace(0, 60, 100)
        a, b = clf.coef_[0]
        c = clf.intercept_[0]
        y_vals = (-a * x_vals - c) / b

        boundary = go.Scatter(
            x=x_vals, y=y_vals,
            mode='lines',
            line=dict(color='black', dash='dash'),
            showlegend=False
        )

        # Table théorique (vitesses de 0 à 700 m/s par pas de 100)
        speeds = np.arange(0, 701, 100)
        distances = (-b * speeds - c) / a
        distances = np.clip(distances, 0, None).round().astype(int)

        table = pd.DataFrame({
            "Vitesse (m/s)": speeds,
            f"{col_value}_limite (m)": distances
        })

        return scatter, boundary, clf, table

    # === Création du graphique ===
    fig = make_subplots(rows=1, cols=2, shared_xaxes=True,
                        subplot_titles=("ML leak mean", "Expert leak mean"))

    scatter_ml, boundary_ml, clf_ml, table_ml = fit_and_plot(df_det_grouped, 'ml_leak_mean', 1)
    scatter_expert, boundary_expert, clf_expert, table_expert = fit_and_plot(df_det_grouped, 'expert_leak_mean', 2)

    fig.add_trace(scatter_ml, row=1, col=1)
    fig.add_trace(boundary_ml, row=1, col=1)
    fig.add_trace(scatter_expert, row=1, col=2)
    fig.add_trace(boundary_expert, row=1, col=2)

    # Axes
    for c in [1, 2]:
        fig.update_xaxes(title_text="Distance à la fuite réelle (m)", range=[0, 60], row=1, col=c)
        fig.update_yaxes(title_text="Vitesse d’éjection (m/s)", range=[0, 700], row=1, col=c)

    fig.update_layout(
        width=1000, height=500,
        title="Limite de détection à 50% (ML vs Expert)",
        showlegend=False
    )

    fig.show()

    # === TABLEAU 1 : Limites théoriques ===
    table_theo = pd.merge(table_ml, table_expert, on="Vitesse (m/s)", how="outer")
    table_theo = table_theo.set_index("Vitesse (m/s)").T
    table_theo.index = ["ML limite (m)", "Expert limite (m)"]

    # === TABLEAU 2 : Limites réelles par test ===
    tests = df_det_grouped[['label', 'ejection_speed']].drop_duplicates().copy()
    tests['test_num'] = tests['label'].map(Tests_correspondances)

    # Calcul des distances limites (arrondies à l’entier)
    tests['ML_limite (m)'] = (
        (-clf_ml.coef_[0][1] * tests['ejection_speed'] - clf_ml.intercept_[0]) / clf_ml.coef_[0][0]
    ).clip(0).round().astype(int)

    tests['Expert_limite (m)'] = (
        (-clf_expert.coef_[0][1] * tests['ejection_speed'] - clf_expert.intercept_[0]) / clf_expert.coef_[0][0]
    ).clip(0).round().astype(int)

    # --- Ajout du débit massique et diamètre ---
    # Correction : on utilise "Internal diameter - (mm)"
    col_diam = 'Internal diameter - (mm)' if 'Internal diameter - (mm)' in df_config.columns else 'Leak diameter - (mm)'

    df_info = df_config[['Test N°', 'flowrate - (g/s)', col_diam]].rename(columns={
        'Test N°': 'test_num',
        'flowrate - (g/s)': 'Qm (g/s)',
        col_diam: 'D (mm)'
    })

    table_real = tests.merge(df_info, on='test_num', how='left')
    table_real = table_real.sort_values('ejection_speed').reset_index(drop=True)

    table_real = table_real.set_index('label')[[
        'test_num', 'Qm (g/s)', 'D (mm)', 'ejection_speed', 
        'ML_limite (m)', 'Expert_limite (m)'
    ]]
    table_real['test_num'] = table_real['test_num'].astype(int)
    table_real = table_real.sort_values(by='test_num')
    # === AFFICHAGE ===
    print("\n=== Table 1 : Limite de détection par vitesse ===")
    display(table_theo)

    print("\n=== Table 2 : Limite de détection par test ===")
    display(table_real)

    return table_theo, table_real

def compute_detection_dataframe(df_stat_local, df_positions, df_config, Tests_correspondances):
    """
    Construit un DataFrame regroupant pour chaque capteur et chaque test (Q...):
    - la distance entre le capteur et la fuite réelle
    - la vitesse d’éjection correspondante
    - les moyennes ml_leak_mean et expert_leak_mean
    
    Args:
        df_stat_local : DataFrame contenant les stats locales par capteur et label
        df_positions : DataFrame avec l'index = 'Public ID' et colonnes ['x (m)', 'y (m)', ...]
        df_config : DataFrame de configuration des tests (positions réelles et vitesse d’éjection)
        Tests_correspondances : dict ou série mappant label → Test N°
        
    Returns:
        df_det_grouped : DataFrame avec colonnes :
            ['label', 'sensor_clean', 'ml_leak_mean', 'expert_leak_mean',
             'distance_m', 'ejection_speed']
    """

    # --- Nettoyage ---
    df_stat_local = df_stat_local.copy()
    df_stat_local['sensor_clean'] = df_stat_local['sensor'].str.strip().str.upper()

    # --- Préparer df_positions ---
    df_positions = df_positions.copy()
    df_positions.index = df_positions.index.astype(str).str.strip().str.upper()

    # --- Filtrer uniquement les essais contenant "Q" ---
    df_det = df_stat_local[df_stat_local['label'].str.contains("Q", case=False)].copy()

    # --- Ajouter le numéro de test via la correspondance ---
    df_det['test_num'] = df_det['label'].map(Tests_correspondances)

    # --- Fusion avec les coordonnées (index = capteur) ---
    df_det = df_det.merge(
        df_positions[['x (m)', 'y (m)']],
        left_on='sensor_clean',
        right_index=True,
        how='left'
    )

    # --- Fonction interne pour récupérer position réelle et vitesse ---
    def get_real_position_and_speed(test_num):
        if pd.isna(test_num):
            return None, None, np.nan
        row = df_config[df_config['Test N°'] == test_num]
        if row.empty:
            return None, None, np.nan
        x_real = row['Leak localisation - X (m)'].values[0]
        y_real = -row['Leak localisation - Y (m)'].values[0]  # inversion Y
        ejection_speed = row['Outlet ejection speed - (m/s)'].values[0]
        return x_real, y_real, ejection_speed

    # --- Appliquer à chaque test ---
    df_det[['x_real', 'y_real', 'ejection_speed']] = df_det['test_num'].apply(
        lambda tn: pd.Series(get_real_position_and_speed(tn))
    )

    # --- Calculer la distance capteur ↔ fuite réelle ---
    df_det['distance_m'] = np.sqrt(
        (df_det['x (m)'] - df_det['x_real'])**2 +
        (df_det['y (m)'] - df_det['y_real'])**2
    )

    # --- Moyennes par capteur et label ---
    df_det_grouped = df_det.groupby(['label', 'sensor_clean']).agg(
        ml_leak_mean=('ml_leak_mean', 'mean'),
        expert_leak_mean=('expert_leak_mean', 'mean'),
        distance_m=('distance_m', 'mean'),
        ejection_speed=('ejection_speed', 'first')  # identique pour le même test
    ).reset_index()

    # --- Convertir en pourcentage ---
    df_det_grouped['ml_leak_mean'] *= 100
    df_det_grouped['expert_leak_mean'] *= 100

    return df_det_grouped

def create_test_summary(df_stat_local, df_config, Tests_correspondances):
    """
    Retourne un DataFrame résumant les tests 'Q' avec :
    - Qv (Nl/min)
    - Qv (Nl/s)
    - débit massique en g/s
    - diamètre en mm
    - vitesse d'éjection en m/s
    """
    # --- Filtrer les labels contenant "Q" ---
    df_labels = df_stat_local[df_stat_local['label'].str.contains("Q", case=False)].copy()
    
    # --- Ajouter test_num via Tests_correspondances ---
    df_labels['test_num'] = df_labels['label'].map(Tests_correspondances)
    
    # --- Extraire Q et D depuis le label ---
    def parse_label(label):
        q_match = re.search(r'Q(\d+)', label)
        d_match = re.search(r'D(\d+)', label)
        q_val = int(q_match.group(1)) if q_match else np.nan
        d_val = int(d_match.group(1)) if d_match else np.nan
        return pd.Series([q_val, d_val])
    
    df_labels[['Qv_Nl_per_min', 'diameter_mm']] = df_labels['label'].apply(parse_label)
    df_labels['Qv_Nl_per_s'] = round(df_labels['Qv_Nl_per_min'] / 60.0,1)
    
    # --- Récupérer débit massique en g/s et vitesse d'éjection ---
    def get_config(test_num):
        if pd.isna(test_num):
            return pd.Series([np.nan, np.nan])
        row = df_config[df_config['Test N°'] == test_num]
        if row.empty:
            return pd.Series([np.nan, np.nan])
        massic_flowrate = row['flowrate - (g/s)'].values[0]       # g/s
        ejection_speed = row['Outlet ejection speed - (m/s)'].values[0]
        return pd.Series([massic_flowrate, ejection_speed])
    
    df_labels[['massic_flowrate_gs', 'ejection_speed_ms']] = df_labels['test_num'].apply(get_config)
    
    # --- Sélectionner et renommer les colonnes pour clarté ---
    df_summary = df_labels[['label', 'Qv_Nl_per_min', 'Qv_Nl_per_s', 
                            'massic_flowrate_gs', 'diameter_mm', 'ejection_speed_ms']].drop_duplicates()
    
    # --- Ajouter unités dans le nom de colonne ---
    df_summary = df_summary.rename(columns={
        'Qv_Nl_per_min': 'Qv (Nl/min)',
        'Qv_Nl_per_s': 'Qv (Nl/s)',
        'massic_flowrate_gs': 'Qm (g/s)',
        'diameter_mm': 'diameter (mm)',
        'ejection_speed_ms': 'ejection speed (m/s)'
    })
    
    # --- Trier par vitesse d'éjection ---
    df_summary = df_summary.sort_values('ejection speed (m/s)').reset_index(drop=True)
    
    return df_summary


def get_test_config(label=None, test_num=None):
    """
    Récupère la configuration d'un test et la position réelle de la fuite.
    """
    if test_num is None and label is not None:
        test_num = Tests_correspondances.get(label)
    
    if test_num is None:
        return np.nan, np.nan, np.nan, np.nan, None, None
    
    df_test_config = df_config[df_config['Test N°'] == test_num]
    if df_test_config.empty:
        return np.nan, np.nan, np.nan, np.nan, None, None
    
    row_cfg = df_test_config.iloc[0]
    massic_flowrate = row_cfg.get('flowrate - (g/s)', np.nan)
    volumic_flowrate = round(60 * row_cfg.get('Volumic Flowrate - (Nl/s)', np.nan))
    diameter = row_cfg.get('Internal diameter - (mm)', np.nan)
    ejection_speed = round(row_cfg.get('Outlet ejection speed - (m/s)', np.nan))
    x_real = row_cfg['Leak localisation - X (m)']
    y_real = -row_cfg['Leak localisation - Y (m)']
    
    return massic_flowrate, volumic_flowrate, diameter, ejection_speed, x_real, y_real

def update_plot_deployment(test_num):
    # --- Filtrage des données ---
    df_filtered = df_stat_deployment[df_stat_deployment['Test N°'] == test_num]
    df_loc = df_localisation[df_localisation['Test N°'] == test_num]

    if df_filtered.empty:
        print(f"Aucune donnée trouvée pour le Test {test_num}")
        return

    # --- Récupération configuration via get_test_config ---
    massic_flowrate, volumic_flowrate, diameter, ejection_speed, x_real, y_real = get_test_config(test_num=test_num)

    subplot_titles = [""]*len(freq_categories)

    fig = make_subplots(
        rows=len(freq_categories),
        cols=2,
        shared_xaxes=False,
        vertical_spacing=0.02,
        column_widths=[0.6, 0.4],
        specs=[[{}, {"rowspan": len(freq_categories)}]] + [[{} , None]]*(len(freq_categories)-1),
        subplot_titles=subplot_titles
    )

    # ---------- PLOTS DE FRÉQUENCES ----------
    for i, freq in enumerate(freq_categories, start=1):
        for sensor in df_filtered['sensor'].unique():
            values = df_filtered[df_filtered['sensor']==sensor][[c for c in df_filtered.columns if c.startswith(freq+'_')]]
            if values.empty:
                continue

            mean_val = values.get(f'{freq}_mean', [np.nan]).values[0]
            min_val  = values.get(f'{freq}_min', [np.nan]).values[0]
            max_val  = values.get(f'{freq}_max', [np.nan]).values[0]

            # Trait min/max
            fig.add_trace(
                go.Scatter(
                    x=[sensor, sensor],
                    y=[min_val, max_val],
                    mode='lines+markers',
                    line=dict(color='black', width=4),
                    marker=dict(color='black', size=8, symbol='circle'),
                    showlegend=False
                ),
                row=i, col=1
            )

            # Moyenne
            fig.add_trace(
                go.Scatter(
                    x=[sensor],
                    y=[mean_val],
                    mode='markers',
                    marker=dict(color='red', size=12, symbol='square'),
                    showlegend=False
                ),
                row=i, col=1
            )

        # Ylabels avec titre
        fig.update_yaxes(title_text=f"{freq}" if "Hz" in freq else "ML leak det. (%)",
                         title_standoff=40, row=i, col=1)

        # Limites Y
        fig.update_yaxes(range=[0, 50] if "Hz" in freq else [0, 100], row=i, col=1)
        # Xlabels seulement sur la dernière ligne
        if i != len(freq_categories):
            fig.update_xaxes(showticklabels=False, row=i, col=1)
        else:
            fig.update_xaxes(tickangle=45, row=i, col=1)

    # ---------- LIGNES DE SEUIL ----------
    fig.add_hline(y=20, line=dict(color="orange", width=2, dash="dot"), row=len(freq_categories), col=1)
    fig.add_hline(y=50, line=dict(color="green", width=2, dash="dot"), row=len(freq_categories), col=1)

    # ---------- PLOT DE LOCALISATION ----------
    freq_loc = '5-15 kHz'
    used_sensors = []
    if not df_loc.empty and 'sensors_used' in df_loc.columns:
        used_sensors = df_loc['sensors_used'].iloc[0].replace(" ", "").split(',')
        x_est = df_loc['x_est'].values[0]
        y_est = df_loc['y_est'].values[0]
        dist = df_loc['diff_total'].values[0]
    else:
        x_est, y_est, dist = None, None, None

    # Fusion avec positions
    df_levels = df_filtered[['sensor', f'{freq_loc}_mean']].set_index('sensor')
    df_merge = df_levels.join(df_positions, how='inner')
    df_merge = df_merge.dropna(subset=[f'{freq_loc}_mean', 'x (m)', 'y (m)'])

    # Bornes
    margin = 5.0
    x_min, x_max = df_merge['x (m)'].min()-margin, df_merge['x (m)'].max()+margin
    y_min, y_max = df_merge['y (m)'].min()-margin, df_merge['y (m)'].max()+margin

    # Taille normalisée entre 5 et max_marker_size pour colormap
    min_val = df_merge[f'{freq_loc}_mean'].min()
    max_val = df_merge[f'{freq_loc}_mean'].max()

    # Capteurs utilisés
    df_used = df_merge[df_merge.index.isin(used_sensors)]
    df_unused = df_merge[~df_merge.index.isin(used_sensors)]

    # Colormap des capteurs utilisés
    if not df_used.empty:
        fig.add_trace(
            go.Scatter(
                x=df_used['x (m)'],
                y=df_used['y (m)'],
                mode='markers+text',
                marker=dict(
                    size=14,
                    color=df_used[f'{freq_loc}_mean'],
                    colorscale=[
                        [0, "lightgray"],  # bas de l'échelle
                        [1, 'rgb(0,166,185)']       # haut de l'échelle
                    ],
                    cmin=10,  # borne minimale
                    cmax=40,  # borne maximale
                    colorbar=dict(
                        title=f"Niveau {freq_loc} (dB)",
                        x=1.02,
                        y=0.25,
                        len=0.5
                    ),
                    sizemode='diameter'
                ),
                text=df_used.index,
                textposition="top center",
                hovertemplate="Capteur: %{text}<br>Niveau: %{marker.color:.1f} dB",
                name='Capteurs utilisés pour la loc.'
            ),
            row=1, col=2
        )

    for _, sensor in df_used.iterrows():
        fig.add_trace(
            go.Scatter(
                x=[sensor['x (m)'], x_est],
                y=[sensor['y (m)'], y_est],
                mode="lines",
                line=dict(
                    color="rgb(0,166,185)",
                    dash="dot",
                    width=1.5
                ),
                showlegend=False,
                hoverinfo="skip"
            ),
            row=1, col=2
        )

    # Capteurs non utilisés avec contour gris (facultatif)
    if not df_unused.empty:
        fig.add_trace(
            go.Scatter(
                x=df_unused['x (m)'],
                y=df_unused['y (m)'],
                mode='markers+text',
                marker=dict(size=10, color='black', symbol='x'),
                text=df_unused.index,
                textposition="top center",
                hovertemplate="Capteur: %{text} (non utilisé)",
                name='Capteurs non utilisés pour la loc.'
            ),
            row=1, col=2
        )

    # ---------- CERCLES SELON LEAK_MEAN POUR TOUS LES CAPTEURS ----------
    colors_all = []
    for sensor in df_merge.index:
        leak_val = df_filtered[df_filtered['sensor'] == sensor]['leak_mean'].values[0]
        if leak_val > 50:
            colors_all.append('green')
        elif 20 <= leak_val <= 50:
            colors_all.append('orange')
        else:
            colors_all.append('red')  # couleur neutre

    fig.add_trace(
        go.Scatter(
            x=df_merge['x (m)'],
            y=df_merge['y (m)'],
            mode='markers',
            marker=dict(
                size=20,  # légèrement plus grand que la colormap
                color=colors_all,
                symbol='circle-open',  # cercle vide pour ne pas cacher la colormap
                line=dict(width=5)    # épaisseur du contour
            ),
            text=df_merge.index,
            textposition="top center",
            hovertemplate="Capteur: %{text}<br>Leak mean: %{marker.color}",
            name='Leak det. (<span style="color:red;">< 20 %</span>, <span style="color:orange;">[20-50] %</span>, <span style="color:green;"> > 50 %</span>)',
            showlegend=True
        ),
        row=1, col=2
    )

    # Fuite réelle
    fig.add_trace(go.Scatter(
        x=[x_real], y=[y_real],
        mode='markers',
        marker=dict(color='black', size=16, symbol='diamond'),
        name='Position réelle'
    ), row=1, col=2)

    # Fuite estimée si dispo
    if x_est is not None and y_est is not None:
        fig.add_trace(go.Scatter(
            x=[x_est], y=[y_est],
            mode='markers',
            marker=dict(color='rgb(0,166,185)', size=16, symbol='diamond'),
            name='Position estimée'
        ), row=1, col=2)

        # Flèche
        fig.add_annotation(
            x=x_est, y=y_est,
            ax=x_real, ay=y_real,
            xref='x2', yref='y2',
            axref='x2', ayref='y2',
            showarrow=True,
            arrowhead=3, arrowsize=1, arrowwidth=2, arrowcolor='black'
        )
        # Distance texte
        fig.add_annotation(
            x=x_real, y=y_real,
            xref='x2', yref='y2',
            text=f"<b>{dist:.1f} m</b>",
            showarrow=False,
            font=dict(size=18, color='black'),
            yshift=20
        )

    # Axes bornés
    fig.update_xaxes(range=[x_min, x_max], row=1, col=2)
    fig.update_yaxes(range=[y_max, y_min], row=1, col=2)

    # Rectangle pointillé noir
    fig.add_shape(type="rect",
                  x0=0, y0=0, x1=40, y1=55,
                  line=dict(color="black", width=1, dash="dot"),
                  row=1, col=2)

    # Layout final
    fig.update_layout(
        height=150*len(freq_categories),
        width=1200,
        title=dict(
            text=f"<b>Test N° {test_num} - Qm: {massic_flowrate} g/s - Qv: {volumic_flowrate} Nl/min - \u2300: {diameter} mm - V: {ejection_speed} m/s</b>",
            x=0.5
        ),       
        showlegend=True,
        margin=dict(l=40, r=40, t=40, b=40)
    )

    fig.show()



def update_plot_local(label):
    # --- Filtrage des données locales pour le label ---
    df_filtered = df_stat_local[df_stat_local['label'] == label]
    if df_filtered.empty:
        print(f"Aucune donnée trouvée pour le label '{label}'")
        return

    # --- Récupération configuration via get_test_config ---
    massic_flowrate, volumic_flowrate, diameter, ejection_speed, x_real, y_real = get_test_config(label=label)
    
    # --- Figure ---
    fig = make_subplots(
        rows=2, cols=2,
        shared_xaxes=False,
        vertical_spacing=0.08,
        column_widths=[0.55, 0.45],
        specs=[[{}, {"rowspan": 2}], [{}, None]],
        subplot_titles=(
            "<b><span style='color:deepskyblue;'>ML model</span></b>",
            "<b><span style='color:deepskyblue;'>ML model</span> "
            "<span style='color:black;'>vs</span> "
            "<span style='color:purple;'>Expert model</span></b>",
            "<b><span style='color:purple;'>Expert model</span></b>"
        )
    )

    # --- Graphes gauche : ml_leak / expert_leak ---
    colors_left = ['deepskyblue', 'purple']  # ligne 1 et ligne 2
    for i, colname in enumerate(['ml_leak', 'expert_leak'], start=1):
        for sensor in df_filtered['sensor'].unique():
            df_sensor = df_filtered[df_filtered['sensor'] == sensor]
            min_val = df_sensor[f'{colname}_min'].values[0] * 100
            max_val = df_sensor[f'{colname}_max'].values[0] * 100
            mean_val = df_sensor[f'{colname}_mean'].values[0] * 100

            # # Trait min/max
            # fig.add_trace(
            #     go.Scatter(
            #         x=[sensor, sensor], y=[min_val, max_val],
            #         mode='lines+markers',
            #         line=dict(color='black', width=4),
            #         marker=dict(color='black', size=8, symbol='circle'),
            #         showlegend=False
            #     ),
            #     row=i, col=1
            # )

            # Moyenne (carré)
            fig.add_trace(
                go.Scatter(
                    x=[sensor],
                    y=[mean_val],
                    mode='markers',
                    marker=dict(color=colors_left[i-1], size=16, symbol='square'),
                    showlegend=False
                ),
                row=i, col=1
            )

        fig.update_yaxes(title_text='%', row=i, col=1,range=[0, 100])
        fig.update_xaxes(showticklabels=(i==2), tickangle=45, row=i, col=1)

    # --- Fusion avec positions pour la carte à droite ---
    df_levels = df_filtered[['sensor', 'ml_leak_mean']].set_index('sensor')
    df_merge = df_levels.join(df_positions, how='inner')
    df_merge = df_merge.dropna(subset=['ml_leak_mean', 'x (m)', 'y (m)'])

    # Bornes
    margin = 5.0
    x_min, x_max = df_merge['x (m)'].min()-margin, df_merge['x (m)'].max()+margin
    y_min, y_max = df_merge['y (m)'].min()-margin, df_merge['y (m)'].max()+margin

    # --- Capteurs : carrés sous les points ---
    square_offset = -3.0  # décalage vertical sous le point
    square_size = 2.0    # taille des carrés
    offset_x = 0.5  # décalage horizontal
    line_width = 4  # épaisseur du contour
    for _, row in df_merge.iterrows():
        x, y = row['x (m)'], row['y (m)']
        leak_val = row['ml_leak_mean']
        expert_val = df_filtered[df_filtered['sensor']==row.name]['expert_leak_mean'].values[0]

        # Remplissage basé sur la valeur (rouge → vert)
        fill_leak = f"rgb({int(255*(1-leak_val*100/100))},{int(255*(leak_val*100/100))},0)"  # ou formule précédente
        fill_expert = f"rgb({int(255*(1-expert_val*100/100))},{int(255*(expert_val*100/100))},0)"

        # Carré gauche (décalé vers la gauche)
        fig.add_shape(
            type="rect",
            x0=x-square_size-offset_x, x1=x-offset_x,
            y0=y-square_offset-square_size, y1=y-square_offset,
            fillcolor=fill_leak,
            line=dict(color='deepskyblue', width=line_width),
            row=1, col=2
        )

        # Carré droit (décalé vers la droite)
        fig.add_shape(
            type="rect",
            x0=x+offset_x, x1=x+square_size+offset_x,
            y0=y-square_offset-square_size, y1=y-square_offset,
            fillcolor=fill_expert,
            line=dict(color='purple', width=line_width),
            row=1, col=2
        )

        # Point du capteur + nom
        fig.add_trace(
            go.Scatter(
                x=[x], y=[y],
                mode='markers+text',
                marker=dict(color='black', size=8),
                text=[row.name],
                textposition='top center',
                showlegend=False
            ),
            row=1, col=2
        )

    # --- Fuite réelle ---
    if x_real is not None and y_real is not None:
        fig.add_trace(
            go.Scatter(
                x=[x_real], y=[y_real],
                mode='markers',
                marker=dict(color='green', size=16, symbol='x'),
                name='Fuite réelle'
            ),
            row=1, col=2
        )

    # Axes bornés
    fig.update_xaxes(range=[x_min, x_max], row=1, col=2)
    fig.update_yaxes(range=[y_max, y_min], row=1, col=2)

    # Rectangle pointillé noir pour zone totale
    fig.add_shape(
        type="rect",
        x0=0, y0=0, x1=40, y1=55,
        line=dict(color="black", width=1, dash="dot"),
        row=1, col=2
    )

    # --- Colorbar rouge vif → vert vif ---
    fig.add_trace(
        go.Scatter(
            x=[None], y=[None], mode='markers',
            marker=dict(
                colorscale=[[0,'#FF0000'], [1,'#00FF00']],  # rouge pétant → vert pétant
                cmin=0, cmax=100,
                colorbar=dict(title="(%)", x=1.1, y=0.5, len=1, thickness=30)
            ),
            hoverinfo='none', showlegend=False
        ),
        row=1, col=2
    )

    # Layout final
    fig.update_layout(
        height=900, width=1200,
        title=dict(
            text=f"<b>Qm: {massic_flowrate} g/s - Qv: {volumic_flowrate} Nl/min - \u2300: {diameter} mm - V: {ejection_speed} m/s</b>",
            x=0.5),
        showlegend=False, margin=dict(l=40, r=80, t=60, b=40)
    )

    fig.show()

# ================================
# DataFrame - Leak configurations
# ================================
# Création du DataFrame
df_positions = pd.DataFrame(data)
df_config = pd.read_excel(leak_configurations, sheet_name="Tests", header=[0, 1, 2])
df_config.columns = df_config.columns.droplevel(0)
df_config.columns = [
    f"{lvl0} - {lvl1}" if lvl1 and 'Unnamed' not in str(lvl1) else f"{lvl0}"
    for lvl0, lvl1 in df_config.columns
]

start_date = pd.to_datetime(start_str)  # exemple
end_date = pd.to_datetime(end_str)    # exemple
df_config['Day'] = pd.to_datetime(df_config['Day'], errors='coerce')
df_config = df_config[df_config['Day'].between(start_date, end_date)].reset_index(drop=True)

columns_to_keep = ['campaign', 
       'Day',
       'Test N°',
       'Release Start - (local time)',
       'Release Start stabilized - (local time)',
       'Release End - (local time)',
       'Duration',
       'gas',
       'Localisation :\nNorth / South',
       'Leak equipment',
       'Leak Type / Location',  
       'Leak localisation - X (m)',
       'Leak localisation - Y (m)',
       'Leak localisation - Z (m)', 
       'flowrate - (g/s)', 
       'flowrate - (kg/h)',
       'OP Valve  - (%)',
       'Nominal size of orifice\n(For the highest flowrate)',
       'Internal diameter - (mm)', 'Pressure - (bar)',
       'Ratio Qm/ID', 
       'Volumic Flowrate - (Nl/s)',
       'Volumic Flowrate - (Nml/min)',
       'Outlet ejection speed - (m/s)',
       'Qty per test - (kg)',
       'Total quantity per test  (kg) - CH4',
       'Total quantity per test  (kg) - CO2',
       'Total quantity per test  (kg) - N2',
       'Total quantity per test  (kg) - C2H6',
       'Total quantity per test  (kg) - C3H8',
       'Total quantity per test  (kg) - C4H8',
       'Total quantity per test  (kg) - C4H10',
       'Total quantity per test  (kg) - H2',
       'flowrate_class:\n0-1 = Q<1 g/s;\n1-10 = 1≤Q<10 g/s;\n10+ = Q≥10 g/s',
       ]

# Conversion correcte de la colonne 'Day'


# Combinaison propre
df_config['start_datetime'] = [
    pd.Timestamp.combine(d, t)
    for d, t in zip(df_config['Day'], df_config['Release Start - (local time)'])
]
df_config['end_datetime'] = [
    pd.Timestamp.combine(d, t)
    for d, t in zip(df_config['Day'], df_config['Release End - (local time)'])
]
df_config['start_datetime'] = pd.to_datetime(df_config['start_datetime']).dt.tz_localize('Europe/Paris', nonexistent='NaT', ambiguous='NaT')
df_config['end_datetime'] = pd.to_datetime(df_config['end_datetime']).dt.tz_localize('Europe/Paris', nonexistent='NaT', ambiguous='NaT')

freq_categories = ['0-1 kHz', '1-5 kHz', '5-15 kHz', '15-45 kHz']
# Étape 1 : calculer le minimum global par fréquence
global_min = {}
for freq in freq_categories:
    all_values = []
    for sensor in sensors:
        df_soundlevels = pd.read_excel(results, sheet_name=sensor, header=[0])
        numeric_cols = [c for c in df_soundlevels.columns if freq in c]
        for col in numeric_cols:
            all_values.extend(pd.to_numeric(df_soundlevels[col], errors='coerce').dropna().values)
    global_min[freq] = min(all_values) if all_values else None

# ----------------------------
# Analyse statistique
# ----------------------------
df_stat_deployment = pd.DataFrame()
df_stat_local = pd.DataFrame()


for sensor in sensors:
    # ----------------------------
    # Lecture des données calculées en local
    # ----------------------------
    datasets_dir = Path("data") / "MAGNETO" / sensor / start_str / "results" / f"ch{channel}"
    features = read_hdf_dataset(datasets_dir.joinpath("features.h5"))
    df_features = pd.DataFrame(features['features'])

    labels = df_features.index.get_level_values('Label').unique()

    stats_list = []  # contiendra les stats pour chaque label de ce sensor

    for label in labels:
        df_subset = df_features.loc[df_features.index.get_level_values('Label') == label]
        # Suppression de 15 s avant et après chaque saut temporel pour éviter les effets de bords (i.e, lente réactivité du modèle ou imprécision d'annotation)
        df_subset = df_subset.groupby(level=['filename','channel','Label'], group_keys=False).apply(remove_jumps, threshold=pd.Timedelta(seconds=15))

        # dictionnaire des statistiques pour un (sensor, label)
        stats = {
            'sensor': sensor,
            'label': label,
            'ml_leak_min': df_subset['leak_ml_mean'].min(),
            'expert_leak_min': df_subset['leak_expert_mean'].min(),
            'ml_leak_max': df_subset['leak_ml_mean'].max(),
            'expert_leak_max': df_subset['leak_expert_mean'].max(),
            'ml_leak_mean': df_subset['leak_ml_mean'].mean(),
            'expert_leak_mean': df_subset['leak_expert_mean'].mean(),
            'ml_leak_std': df_subset['leak_ml_mean'].std(),
            'expert_leak_std': df_subset['leak_expert_mean'].std(),
        }

        stats_list.append(stats)

    # DataFrame des stats pour ce sensor
    df_stat_sensor = pd.DataFrame(stats_list)
    df_stat_local = pd.concat([df_stat_local, df_stat_sensor], ignore_index=True)

    # ----------------------------
    # Lecture des données de déploiement
    # ----------------------------
    df_soundlevels = pd.read_excel(results, sheet_name=sensor, header=[0])
    df_soundlevels['Time'] = pd.to_datetime(df_soundlevels['Time'], errors='coerce')
    
    numeric_cols = df_soundlevels.columns.difference(['Time'])
    for col in numeric_cols:
        df_soundlevels[col] = pd.to_numeric(df_soundlevels[col], errors='coerce')
    
    df_leak = pd.read_excel(results, sheet_name='Leak-Detection', header=[0])
    df_leak['Time'] = pd.to_datetime(df_leak['Time'], errors='coerce')
    numeric_cols_leak = df_leak.columns.difference(['Time'])
    for col in numeric_cols_leak:
        df_leak[col] = pd.to_numeric(df_leak[col], errors='coerce')

    # ----------------------------
    # Calcul des statistiques
    # ----------------------------
    list_results = []
    for _, row in df_config.iterrows():
        test_num = row['Test N°']
        if pd.isna(test_num):
            continue  # Ignore les tests sans numéro

        start = row['start_datetime']
        end = row['end_datetime']
        
        # Soundlevels
        mask = (df_soundlevels['Time'] >= start) & (df_soundlevels['Time'] <= end)
        subset = df_soundlevels.loc[mask]
        numeric_cols = subset.select_dtypes(include='number').columns
        
        stats = {}
        # Emergence
        for col in numeric_cols:
            freq = None
            for f in freq_categories:
                if f in col:
                    freq = f
                    break
            if freq is None:
                continue
            min_val_global = global_min[freq]
            stats[f'{col}_min'] = subset[col].min() - min_val_global 
            stats[f'{col}_max'] = subset[col].max() - min_val_global 
            stats[f'{col}_mean'] = subset[col].mean() - min_val_global 
            stats[f'{col}_std'] = subset[col].std() - min_val_global 
        # ML Leak
        mask = (df_leak['Time'] >= start) & (df_leak['Time'] <= end)
        subset = df_leak.loc[mask]
        stats['leak_min'] = subset[sensor].min()
        stats['leak_max'] = subset[sensor].max()
        stats['leak_mean'] = subset[sensor].mean()
        stats['leak_std'] = subset[sensor].std()

        # Ajouter le sensor et le test
        stats['Test N°'] = test_num
        stats['sensor'] = sensor
        list_results.append(stats)
    
    df_stat = pd.DataFrame(list_results)
    df_stat_deployment = pd.concat([df_stat_deployment, df_stat], ignore_index=True)

df_stat_deployment = df_stat_deployment.dropna(subset=['Test N°'])



df_positions = pd.DataFrame(data)
df_positions = df_positions.set_index("Public ID")
# Initialisation du dataframe
df_localisation = pd.DataFrame(columns=[
    'Test N°', 
    'x_real', 'y_real', 
    'x_est', 'y_est', 
    'diff_x', 'diff_y', 'diff_total',
    'n_sensors_used', 'sensors_used'
])


freq = '5-15 kHz'
ignore_sensor = 'ROGUE4'  
max_sensors_used = 5  # nombre max de capteurs à conserver

for _, row in df_config.iterrows():
    
    test_num = row['Test N°']
    
    # Vérifier que x_real est un float
    x_real = row['Leak localisation - X (m)']
    if not isinstance(x_real, (float, int)):
        continue
    
    # Positions réelles
    y_real = abs(row['Leak localisation - Y (m)'])
    
    # Niveaux moyens des capteurs pour ce test
    df_test = df_stat_deployment[df_stat_deployment['Test N°'] == test_num]
    df_levels = df_test[['sensor', f'{freq}_mean']].set_index('sensor')
    
    # Ignorer le capteur spécifique si demandé
    if ignore_sensor is not None:
        df_levels = df_levels[df_levels.index != ignore_sensor]
    
    # Filtrer les capteurs dépassant le seuil
    df_levels_filtered = df_levels[df_levels[f'{freq}_mean'] > min_emergence]
    
    # Si plus de 4 capteurs dépassent le seuil → on garde les 4 plus élevés
    if len(df_levels_filtered) > max_sensors_used:
        df_levels_filtered = df_levels_filtered.sort_values(by=f'{freq}_mean', ascending=False).head(max_sensors_used)
    
    # Calculer uniquement si au moins 3 capteurs dépassent le seuil
    if len(df_levels_filtered) < 3:
        continue
    
    # Merge avec positions des capteurs
    df_merge = df_levels_filtered.join(df_positions, how='inner')


    # Calcul de la position centrale pondérée avec la règle -6dB par doublement de distance
    # Autrement dit : décroissance de p ~ 1/r (i.e : I ~ 1/r²)
    # Convertir les niveaux dB en "pression relative"
    weights = 10 ** (df_merge[f'{freq}_mean'] / 20)

    # Barycentre pondéré
    x_est = (df_merge['x (m)'] * weights).sum() / weights.sum()
    y_est = (df_merge['y (m)'] * weights).sum() / weights.sum()

    
    # Différences
    diff_x = x_est - x_real
    diff_y = y_est - y_real
    diff_total = (diff_x**2 + diff_y**2)**0.5
    
    # Nombre et liste des capteurs utilisés
    n_sensors_used = len(df_merge)
    sensors_used = ', '.join(df_merge.index)
    
    # Ajouter au dataframe
    df_localisation = pd.concat([
        df_localisation,
        pd.DataFrame([{
            'Test N°': test_num,
            'x_real': x_real, 'y_real': y_real,
            'x_est': x_est, 'y_est': y_est,
            'diff_x': diff_x, 'diff_y': diff_y,
            'diff_total': diff_total,
            'n_sensors_used': n_sensors_used,
            'sensors_used': sensors_used
        }])
    ], ignore_index=True)




# ==========================
# === Statistic analysis ===
# ==========================

# ==== Localisation Dataframe
df_localisation
display(df_localisation)
print(f"mean localization error: {round(df_localisation['diff_total'].mean(),2)} m" )


# ==== Taux de détection
df_det = df_stat_local[df_stat_local['label'].str.contains('Q', case=False, na=False)]
df_other = df_stat_local[~df_stat_local['label'].str.contains('Q', case=False, na=False)]
max_per_label = (
    df_stat_local.groupby("label")[["expert_leak_mean", "ml_leak_mean"]]
    .max()
    .mul(100)
    .reset_index()
)
# --- Taux maximal par label (tous capteurs confondus) ---
max_per_label = (
    df_stat_local.groupby("label")[["expert_leak_mean", "ml_leak_mean"]]
    .max()
    .mul(100)
    .reset_index()
)
max_per_label["test_num"] = max_per_label["label"].map(Tests_correspondances).astype(pd.Int64Dtype())
max_per_label = max_per_label.sort_values("test_num", na_position="last").reset_index(drop=True)

# --- Moyenne par capteur pour les détections (labels contenant "Q") ---
mean_det = (
    df_det.groupby("sensor")[["expert_leak_mean", "ml_leak_mean"]]
    .mean()
    .mul(100)
    .reset_index()
)
if "label" in df_det.columns:
    label_map_det = df_det.groupby("sensor")["label"].first()
    mean_det["label"] = mean_det["sensor"].map(label_map_det)

# --- Moyenne par capteur pour les autres labels ---
mean_other = (
    df_other.groupby("sensor")[["expert_leak_mean", "ml_leak_mean"]]
    .mean()
    .mul(100)
    .reset_index()
)
if "label" in df_other.columns:
    label_map_other = df_other.groupby("sensor")["label"].first()
    mean_other["label"] = mean_other["sensor"].map(label_map_other)

mean_det = mean_det.round(0)
mean_other = mean_other.round(0)
max_per_label = max_per_label.round(0)

display(mean_det)
display(mean_other)
display(max_per_label)

# ==== Taux de détection en fonction de la vitesse d'éjection et de la distance
df_det_grouped = compute_detection_dataframe(
    df_stat_local, df_positions, df_config, Tests_correspondances
)


df_test_summary = create_test_summary(df_stat_local, df_config, Tests_correspondances)
display(df_test_summary)


# ======================================================================================================================

# =====================
# ====== FIGURES ======
# =====================

# ====================
# === Static visu. ===
# ====================

table_theo, table_real = plot_detection_vs_distance(df_det_grouped, df_config, Tests_correspondances)

# =====================
# === Dynamic visu. ===
# =====================

freq_categories = ['0-1 kHz', '1-5 kHz', '5-15 kHz', '15-45 kHz', 'leak']

test_selector = widgets.Dropdown(
    options=df_stat_deployment['Test N°'].unique(),
    description='Test N°:',
    value=df_stat_deployment['Test N°'].unique()[0]
)

# --- Dropdown pour choisir le type de visualisation ---
display_mode_selector = widgets.Dropdown(
    options=[
        ('ML leak + localisation', 'ml_loc'),
        ('ML model vs Expert model', 'ml_vs_expert')
    ],
    description='Mode affichage :',
    style={'description_width': 'initial'},
    layout=widgets.Layout(width='300px')
)

# --- Widget pour sélectionner le label / test ---
label_selector = widgets.Dropdown(
    options=sorted(df_stat_local['label'].unique()),
    description='Label :',
    style={'description_width': 'initial'},
    layout=widgets.Layout(width='300px')
)

test_selector = widgets.Dropdown(
    options=df_stat_deployment['Test N°'].unique(),
    description='Test N° :',
    style={'description_width': 'initial'},
    layout=widgets.Layout(width='300px')
)

# --- Fonction qui choisit quelle mise à jour appliquer ---
def update_display(mode, label, test_num):
    if mode == 'ml_loc':
        # ML leak + localisation → n'affiche que si test_num valide
        if pd.notna(test_num):
            update_plot_deployment(test_num)
        else:
            print(f"Aucun test valide pour ce label, affichage impossible")
    elif mode == 'ml_vs_expert':
        # ML vs Expert → utilise label
        update_plot_local(label)


# --- Widget interactif combiné ---
out = widgets.interactive_output(
    update_display,
    {
        'mode': display_mode_selector,
        'label': label_selector,
        'test_num': test_selector
    }
)

# --- Affichage des widgets ---
display(widgets.HBox([display_mode_selector, label_selector, test_selector]), out)

# --- Optionnel : masquage dynamique de label/test selon le mode ---
def toggle_inputs(change):
    if change['new'] == 'ml_loc':
        label_selector.layout.display = 'none'
        test_selector.layout.display = 'block'
    else:
        label_selector.layout.display = 'block'
        test_selector.layout.display = 'none'

display_mode_selector.observe(toggle_inputs, names='value')
toggle_inputs({'new': display_mode_selector.value})



# %%
