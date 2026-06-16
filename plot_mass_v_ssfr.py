import sys
sys.path.append("..")

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import funcs
import astropy.io.fits as fits
from astropy.table import Table

path_simba = "../simba_rlf_data/df_simba.pkl"
df_simba = pd.read_pickle(path_simba)

path_manga = "../simba_rlf_data/df_manga.pkl"
df_manga = pd.read_pickle(path_manga)

# Set up the figure with a 1x2 grid
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

# Plot MaNGA Data
mesh_manga = funcs.plot_mass_v_ssfr(df_manga, ax=axes[0], title='MaNGA')

# Plot SIMBA Simulation
mesh_simba = funcs.plot_mass_v_ssfr(df_simba, ax=axes[1], title='SIMBA')


# Clean up spacing and display
plt.tight_layout()
plt.savefig("../simba_rlf_data/plots/ssfr_v_mass.png")
plt.show()