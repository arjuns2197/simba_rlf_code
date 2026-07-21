import sys
sys.path.append("..")

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import funcs
import astropy.io.fits as fits
from astropy.table import Table
from astropy.cosmology import WMAP9 as cosmo
import h5py

volume_mock = 100*(100/0.68)**3
path_mock = "../simba_rlf_data/df_mock.pkl"
df_mock = pd.read_pickle(path_mock)

fig, axs = plt.subplots(3,2)

# Mock GSMF
funcs.plot_gsmf_mock(df_mock, volume_mock,axs[0,1])

# Radio Luminosity Distribution SB24
funcs.plot_rld_of_M_mock_sb24(df_mock, axs[1,1])

# Radio Luminosity Function SB24
funcs.plot_rlf_mock_sb24(df_mock, volume_mock, axs[2,1])

path_simba = "../simba_rlf_data/df_simba.pkl"
df_simba = pd.read_pickle(path_simba)
volume = (100/0.68)**3

# SIMBA GSMF
funcs.plot_gsmf_simba(df_simba, volume,axs[0,0])

# Radio Luminosity Distribution NT21 
funcs.plot_rld_of_M_simba_nt21(df_simba, axs[1,0])

# Radio Luminosity Function NT21
funcs.plot_rlf_simba_nt21(df_simba, volume, axs[2,0])

plt.savefig("../simba_rlf_data/plots/gsmf_rlf_mock_vs_simba.png")
plt.show()