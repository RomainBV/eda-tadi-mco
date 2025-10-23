#%%
import pandas as pd
import os
import plotly.graph_objects as go
from ipywidgets import widgets
from IPython.display import display
from plotly.subplots import make_subplots
import numpy as np

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
    "Sensor ID": ["e4-5f-01-8f-6f-95", "e4-5f-01-8f-6f-04", "e4-5f-01-8f-6d-ab",
                  "e4-5f-01-8f-6f-fb", "e4-5f-01-8f-6f-9c", "e4-5f-01-8f-72-89", "e4-5f-01-8f-6f-09"],
    "x (m)": [-0.3, -0.3, 20.0, 40.2, 40.2, 21.5, 28.4],
    "y (m)": [28.9, -0.3, -0.3, -0.3, 29.0, 55.5, 11.4],
    "z (m)": [3, 3, 3, 3, 3, 3, 5.45]
}
min_emergence = 10

# AJOUTER TAUX DE VRAIS POSITIFS ET FAUX POSITIFS
# Taux de détection global / par capteur / vitesse d'éjection suivant les 2 modèles



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
df_stat_all = pd.DataFrame()
for sensor in sensors:

    # ----------------------------
    # Lecture des données en local
    # ----------------------------
    file_path = f'data/MAGNETO/{sensor}/{start_str}'
    df_expert = pd.read_excel(results, sheet_name=sensor, header=[0])
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
            continue  # 🔸 Ignore les tests sans numéro

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
        # Leak
        mask = (df_leak['Time'] >= start) & (df_leak['Time'] <= end)
        subset = df_leak.loc[mask]
        stats['leak_min'] = subset[sensor].min()
        stats['leak_max'] = subset[sensor].max()
        stats['leak_mean'] = subset[sensor].mean()
        stats['leak_std'] = subset[sensor].std()
        
        # Ajouter le sensor et le test
        stats['Test N°'] = test_num
        stats['Sensor'] = sensor
        list_results.append(stats)
    
    df_stat = pd.DataFrame(list_results)
    df_stat_all = pd.concat([df_stat_all, df_stat], ignore_index=True)

df_stat_all = df_stat_all.dropna(subset=['Test N°'])



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
    df_test = df_stat_all[df_stat_all['Test N°'] == test_num]
    df_levels = df_test[['Sensor', f'{freq}_mean']].set_index('Sensor')
    
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
    
    # # Calcul de la position centrale pondérée (X et Y seulement)
    # x_est = (df_merge['x (m)'] * df_merge[f'{freq}_mean']).sum() / df_merge[f'{freq}_mean'].sum()
    # y_est = (df_merge['y (m)'] * df_merge[f'{freq}_mean']).sum() / df_merge[f'{freq}_mean'].sum()

    # Calcul de la position centrale pondérée avec la règle -6dB par doublement de distance
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


# =================
# expert leak model
# =================


features_dir = "data/MAGNETO"




# ==========
# Figures
# ==========

freq_categories = ['0-1 kHz', '1-5 kHz', '5-15 kHz', '15-45 kHz', 'leak']

test_selector = widgets.Dropdown(
    options=df_stat_all['Test N°'].unique(),
    description='Test N°:',
    value=df_stat_all['Test N°'].unique()[0]
)
def update_plot(test_num):
    df_filtered = df_stat_all[df_stat_all['Test N°'] == test_num]
    df_loc = df_localisation[df_localisation['Test N°'] == test_num]
    df_test_config = df_config[df_config['Test N°'] == test_num]
    if df_test_config.empty:
        print(f"Aucune configuration trouvée pour Test {test_num}")
        return

    row_cfg = df_test_config.iloc[0]
    massic_flowrate = row_cfg.get('flowrate - (g/s)', np.nan)
    volumic_flowrate = round(60*row_cfg.get('Volumic Flowrate - (Nl/s)', np.nan))
    diameter = row_cfg.get('Internal diameter - (mm)', np.nan)
    ejection_speed = round(row_cfg.get('Outlet ejection speed - (m/s)', np.nan))
    x_real = row_cfg['Leak localisation - X (m)']
    y_real = -row_cfg['Leak localisation - Y (m)']

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
        for sensor in df_filtered['Sensor'].unique():
            values = df_filtered[df_filtered['Sensor']==sensor][[c for c in df_filtered.columns if c.startswith(freq+'_')]]
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
        fig.update_yaxes(title_text=f"{freq}" if "Hz" in freq else freq,
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
    fig.add_hline(y=40, line=dict(color="green", width=2, dash="dot"), row=len(freq_categories), col=1)

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
    df_levels = df_filtered[['Sensor', f'{freq_loc}_mean']].set_index('Sensor')
    df_merge = df_levels.join(df_positions, how='inner')
    df_merge = df_merge.dropna(subset=[f'{freq_loc}_mean', 'x (m)', 'y (m)'])

    # Bornes
    margin = 5.0
    x_min, x_max = df_merge['x (m)'].min()-margin, df_merge['x (m)'].max()+margin
    y_min, y_max = df_merge['y (m)'].min()-margin, df_merge['y (m)'].max()+margin

    # Taille normalisée entre 5 et max_marker_size pour colormap
    max_marker_size = 50
    min_marker_size = 5
    min_val = df_merge[f'{freq_loc}_mean'].min()
    max_val = df_merge[f'{freq_loc}_mean'].max()
    norm_size = min_marker_size + ((df_merge[f'{freq_loc}_mean'] - min_val) / (max_val - min_val + 1e-6)) * (max_marker_size - min_marker_size)

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
                    size=norm_size[df_used.index],
                    color=df_used[f'{freq_loc}_mean'],
                    colorscale='Viridis',
                    colorbar=dict(title="Niveau (dB)", x=1.02, y=0.25, len=0.5),
                    sizemode='diameter'
                ),
                text=df_used.index,
                textposition="top center",
                hovertemplate="Capteur: %{text}<br>Niveau: %{marker.color:.1f} dB",
                name='Capteurs utilisés'
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
                marker=dict(
                    size=8,
                    color='lightgray',
                    line=dict(width=1, color='gray')
                ),
                text=df_unused.index,
                textposition="top center",
                hovertemplate="Capteur: %{text} (non utilisé)",
                name='Capteurs non utilisés'
            ),
            row=1, col=2
        )

    # ---------- CERCLES SELON LEAK_MEAN POUR TOUS LES CAPTEURS ----------
    colors_all = []
    for sensor in df_merge.index:
        leak_val = df_filtered[df_filtered['Sensor'] == sensor]['leak_mean'].values[0]
        if leak_val > 50:
            colors_all.append('green')
        elif 20 <= leak_val <= 50:
            colors_all.append('orange')
        else:
            colors_all.append('gray')  # couleur neutre

    fig.add_trace(
        go.Scatter(
            x=df_merge['x (m)'],
            y=df_merge['y (m)'],
            mode='markers',
            marker=dict(
                size=14,  # légèrement plus grand que la colormap
                color=colors_all,
                symbol='circle-open',  # cercle vide pour ne pas cacher la colormap
                line=dict(width=3)    # épaisseur du contour
            ),
            text=df_merge.index,
            textposition="top center",
            hovertemplate="Capteur: %{text}<br>Leak mean: %{marker.color}",
            name='Leak moy. (<span style="color:gray;">gris < 20 %</span>, <span style="color:orange;">orange [20-50] %</span>, <span style="color:green;">vert > 50 %</span>)',
            showlegend=True
        ),
        row=1, col=2
    )

    # Fuite réelle
    fig.add_trace(go.Scatter(
        x=[x_real], y=[y_real],
        mode='markers',
        marker=dict(color='green', size=16, symbol='x'),
        name='Leak réel'
    ), row=1, col=2)

    # Fuite estimée si dispo
    if x_est is not None and y_est is not None:
        fig.add_trace(go.Scatter(
            x=[x_est], y=[y_est],
            mode='markers',
            marker=dict(color='red', size=16, symbol='x'),
            name='Leak estimé'
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
            text=f"<b>Test N° {test_num} - Massic flowrate: {massic_flowrate} g/s - Volumic Flowrate: {volumic_flowrate} Nl/min - Diameter: {diameter} mm - Ejection speed: {ejection_speed} m/s</b>",
            x=0.5
        ),       
        showlegend=True,
        margin=dict(l=40, r=40, t=40, b=40)
    )

    fig.show()


out = widgets.interactive_output(update_plot, {'test_num': test_selector})
display(test_selector, out)

df_localisation


# %%
