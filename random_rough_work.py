import sys
sys.path.append("..")

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import funcs
import h5py
from scipy.integrate import quad
from scipy.integrate import cumulative_trapezoid


path_simba = "../simba_rlf_data/df_simba.pkl"
df_simba = pd.read_pickle(path_simba)

print(df_simba.columns)

print(df_simba['log_lr_agn_nt_whz'])