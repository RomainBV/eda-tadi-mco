#%%
import pandas as pd
import os
%matplotlib inline
import matplotlib.pyplot as plt
import numpy as np
from typing import  Dict, List
from pathlib import Path

root_dir = "/media/rbeauvais/Elements/romainb/2025-n09-TADI-MCO"
os.chdir(root_dir)
print(root_dir)

show_plot = True
dates = ['2025-10-10','2025-11-05']
end_str = '2025-11-06'
leak_configurations = "TADI - All datasets analyses.xlsx"
sensors = ['RDA-04','RCA-07','RCA-08','RCA-04','RCA-09','RCA-06','ROGUE4']
ultrasoundlevel_bgn = 15 



Refaire tous les calcules

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


def extract_leak(label):
    if 'leak' in label:
        return label.split(' / ')[-1]
    return label


list_features = []   # ← on stocke chaque df_features ici
reference_order = None
# for sensor in sensors:

#     for date in dates:

#         datasets_dir = Path("data") / "MAGNETO" / sensor / date / "results"
#         features = read_hdf_dataset(datasets_dir.joinpath("features.h5"))
#         df_features = pd.DataFrame(features['features'])

#         if sensor == 'ROGUE4':
#             df_features['ultrasoundlevel']-=10
        
#         df_lin = 10 ** (df_features["ultrasoundlevel"] / 20)

#         ultrasoundlevel_bgn_lin = 10 ** (ultrasoundlevel_bgn / 20)
#         df_norm = (
#             (df_lin / ultrasoundlevel_bgn_lin) - 1
#         )

#         # Compute the spectral_centroid and spectral flatness
#         df_sc = df_features["spectralcentroid"] / 96000


#         df_features['product_mean'] = 



#         list_features.append(df_features)

# ----------------------------
# CONCATÉNATION FINALE
# ----------------------------

df_all_features = pd.concat(list_features, axis=0)

    # y = df_features['product_std']
    # x = df_features['product_mean']



    # x1, y1 = 0.05, 1e-4
    # x2, y2 = 10, 0.5
    
    # # Calcul des coefficients de la droite y = m*x + c
    # m = (y2 - y1) / (x2 - x1)
    # c = y1 - m * x1

    # # Calcul de la valeur de la droite pour chaque product_mean
    # y_line = m * x + c

    # # Condition : être sous la droite
    # df_features["leak_expert_mean"] = (
    #     y < y_line
    # ).astype(int)

if show_plot:

    # plt.plot(df_all_features['product_std'],df_all_features['product_mean'],'.')
    # plt.xscale('log')
    # plt.yscale('log')
    # x1, y1 = 0.0001, 0.01
    # x2, y2 = 10, 10
    # m = (y2 - y1) / (x2 - x1)
    # c = y1 - m * x1
    # x_line = np.logspace(np.log10(x1), np.log10(x2), 1000)
    # y_line = m*x_line + c
    # plt.plot(x_line, y_line, 'r-', label='Reference line')

    df = df_all_features

    # masque : True si le Label contient "leak"
    mask_leak = (
        df.index.get_level_values('Label').str.contains('Q', case=False, na=False)
        & (df['ultrasoundlevel'] > 30)
    )
    plt.figure(figsize=(8,6))

    # Autres points → GRIS
    plt.plot(
        df.loc[~mask_leak, 'product_std'],
        df.loc[~mask_leak, 'product_mean'],
        '.',
        color='lightgray',
        label='other'
    )

    # Points leak + ultrasound > 30 → VERT
    plt.plot(
        df.loc[mask_leak, 'product_std'],
        df.loc[mask_leak, 'product_mean'],
        'g.',
        label='leak (ultrasound > 30)'
    )



    plt.xscale('log')
    plt.yscale('log')

    # Ligne de référence
    x1, y1 = 0.001, 0.001
    x2, y2 = 10, 10
    m = (y2 - y1) / (x2 - x1)
    c = y1 - m * x1
    x_line = np.logspace(np.log10(x1), np.log10(x2), 1000)
    y_line = m*x_line + c
    plt.plot(x_line, y_line, 'r-', label='Reference line')

    plt.legend()
    plt.show()



# %%
