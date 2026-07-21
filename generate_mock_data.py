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
from scipy.integrate import cumulative_trapezoid
from scipy.interpolate import interp1d
from scipy.integrate import quad


log_M_star = 10.66
phi1 = 3.96e-3
alpha1 = -0.35
phi2 = 0.79e-3
alpha2 = -1.47

volume = 10000*(100/0.68)**3
num_samples = volume*quad(lambda log_mass: funcs.baldry_gsmf_logspace(10**log_mass,log_M_star,
                                                                phi1, alpha1, phi2, alpha2), 8, 11.5)[0]

num_samples = int(num_samples)

df_mock = funcs.generate_mock_df(volume)
    



