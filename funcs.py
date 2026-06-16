import sys
sys.path.append("..")

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import astropy.io.fits as fits
from astropy.table import Table
import h5py
import caesar

# nt = Nicole Thomas


def get_sf_radio_thermal_nt(log_sfr):

    sfr = 10**log_sfr

    nu = 1.4
    coef = 5.5e20
    term1 = (nu)**(-0.1)
    
    L_sf_t = (nu*1e9)*coef*term1*sfr*(1e7)
    L_sf_t = L_sf_t
    
    return L_sf_t


def get_sf_radio_non_thermal_nt(log_sfr):

    sfr = 10**log_sfr
    
    nu = 1.4
    coef = 5.3e21
    term1 = (nu)**(-0.8)
    
    L_sf_nt = (nu*1e9)*coef*term1*sfr*(1e7)
    
    return L_sf_nt


def get_sf_radio_nt(log_sfr):
    
    L_t = get_sf_radio_thermal_nt(log_sfr)
    L_nt = get_sf_radio_non_thermal_nt(log_sfr)
    
    log_Lsfr = np.log10(L_t + L_nt)
    
    return log_Lsfr


def get_agn_radio_nt(bhm_dot, fedd, log_bhm, log_mstar):
    
    # 1. Create a boolean mask where ALL conditions are True
    # Enclose each condition in parentheses and use '&' instead of 'and'
    mask = (log_bhm > 8) & (fedd > 0) & (fedd < 0.02) & (log_mstar > 9.5)
    
    # 2. Initialize an array of zeros with the same length as your dataframe
    L_agn = np.zeros(len(bhm_dot))
    
    # 3. Calculate the formula ONLY for the rows where the mask is True
    # We slice bhm_dot using [mask] to only do the math on valid rows
    valid_bhm_dot = bhm_dot[mask]
    
    # Use errstate to ignore divide by zero warnings internally if any valid_bhm_dot == 0
    with np.errstate(divide='ignore'):
        L_agn[mask] = (1e30) * (valid_bhm_dot * 6.31e25 / 4e17)**(17/12)
        log_L_agn = np.log10(L_agn)
        
    return log_L_agn


def get_total_radio_nt(log_sfr, bhm_dot, fedd, log_bhm, log_mstar):
    

    L_sfr = 10**get_sf_radio_nt(log_sfr)
    L_agn = 10**get_agn_radio_nt(bhm_dot,fedd, log_bhm, log_mstar)
    
    L_radio = L_sfr + L_agn
    L_radio = np.log10(L_radio)

    return L_radio



def read_df_simba_till_z014():

    path = "../simba_rlf_data/df.pkl"
    df = pd.read_pickle(path)

    # Calculate luminosities
    lsf_thermal = get_sf_radio_thermal_nt(df['log_sfr'])
    lsf_non_thermal = get_sf_radio_non_thermal_nt(df['log_sfr'])
    log_lsf_nt = get_sf_radio_nt(df['log_sfr'])

    log_lagn_nt = get_agn_radio_nt(df['bhmdot'], df['fedd'], df['log_bhm'], df['log_mass'])

    log_lr_total_nt = get_total_radio_nt(df['log_sfr'], df['bhm_dot_50Myr'],
                                           df['fedd'], df['log_bhm'], df['log_mass'])
    
    # Add to dataframe in log space (fixed lsf_t typo)
    df['log_lsf_thermal'] = np.log10(lsf_thermal)
    df['log_lsf_non_thermal'] = np.log10(lsf_non_thermal)
    df['log_lsf_nt'] = log_lsf_nt
    df['log_lagn_nt'] = log_lagn_nt
    df['log_lr_total_nt'] = log_lr_total_nt

    # Overwrite the pickle file
    df.to_pickle("../simba_rlf_data/df_simba_z014.pkl")


    return None



def load_simba_df():

    log_sfr = []
    bhm_dot = []
    log_bhm = []
    fedd = []
    log_ssfr = []
    log_mass = []

    bhm_nt = []
    bhm_dot_50Myr = []

    for num in range(151,152):
        num = str(num)
        path = "../simba_rlf_data/m100n1024_" + num + ".hdf5"
        data = caesar.load(path)


        for gal in data.galaxies:

            sfr = gal.sfr.to_value(units = None, equivalence = None)
            logsfr = np.log10(sfr)
            log_sfr.append(logsfr)

            bhmdot = gal.bhmdot.to_value(units = None, equivalence = None)
            bhm_dot.append(bhmdot)

            bhm = gal.masses['bh'].to_value(units = None, equivalence = None)
            logbhm = np.log10(bhm)
            log_bhm.append(logbhm)

            mass = gal.masses['stellar'].to_value(units = None, equivalence = None)
            logmass = np.log10(mass)
            log_mass.append(logmass)

            fedd_value = gal.bh_fedd.to_value(units = None, equivalence = None)
            fedd.append(fedd_value)
           

            ssfr = gal.sfr.to_value(units = None, equivalence = None)/gal.masses['stellar'].to_value(units = None, equivalence = None)

            if ssfr == 0:
                log_ssfr.append(-1*np.inf)

            else:
                log_ssfr.append(np.log10(ssfr))


        path_nt = "../simba_rlf_data/BH_Bondi_m100n1024_" + num + ".hdf5"
        data_nt = h5py.File(path_nt)

        bhm_nt_this_snap = np.log10(data_nt['BH_Mass'])
        bhm_dot_50Myr_this_snap = 10**np.array(data_nt['BHAR_50MYR'])   #/0.68

        bhm_nt.extend(bhm_nt_this_snap.tolist())
        bhm_dot_50Myr.extend(bhm_dot_50Myr_this_snap.tolist())



    dict = {"log_sfr" : log_sfr, 'bhmdot' : bhm_dot, "log_bhm" : log_bhm, "fedd" : fedd, 
    "log_ssfr" : log_ssfr, "log_mass" : log_mass,
    "bhm_nt": bhm_nt, "bhm_dot_50Myr" : bhm_dot_50Myr}


    df = pd.DataFrame(dict)


    # Calculate luminosities
    lsf_thermal = get_sf_radio_thermal_nt(df['log_sfr'])
    lsf_non_thermal = get_sf_radio_non_thermal_nt(df['log_sfr'])
    log_lsf_nt = get_sf_radio_nt(df['log_sfr'])

    log_lagn_nt = get_agn_radio_nt(df['bhm_dot_50Myr'], df['fedd'], df['log_bhm'], df['log_mass'])

    log_lr_total_nt = get_total_radio_nt(df['log_sfr'], df['bhm_dot_50Myr'],
                                           df['fedd'], df['log_bhm'], df['log_mass'])
    
    # Add to dataframe in log space (fixed lsf_t typo)
    df['log_lsf_thermal'] = np.log10(lsf_thermal)
    df['log_lsf_non_thermal'] = np.log10(lsf_non_thermal)
    df['log_lsf_nt'] = log_lsf_nt
    df['log_lagn_nt'] = log_lagn_nt
    df['log_lr_total_nt'] = log_lr_total_nt

    # Overwrite the pickle file
    df.to_pickle("../simba_rlf_data/df_simba.pkl")

    return None


def load_nt_df():

    path = "../simba_rlf_data/BH_Bondi_m100n1024_151.hdf5"
    file = h5py.File(path, 'r')

    data_dict = {}

    # Loop through all the keys you just found
    for key in file.keys():
        dataset = file[key][:] # The [:] loads the actual data into memory
        
        # If it's a standard 1D array (e.g., single values like mass or density)
        if dataset.ndim == 1:
            data_dict[key] = dataset
            
        # If it's a 2D array with 3 columns (e.g., 3D vectors like velocity 'v' or 'Jgas')
        elif dataset.ndim == 2 and dataset.shape[1] == 3:
            data_dict[f'{key}_x'] = dataset[:, 0]
            data_dict[f'{key}_y'] = dataset[:, 1]
            data_dict[f'{key}_z'] = dataset[:, 2]
            
        # Catch anything else just in case, so pandas doesn't throw an error
        else:
            print(f"Skipped '{key}' because of unhandled shape: {dataset.shape}")

    # Convert the dictionary directly into a DataFrame
    df = pd.DataFrame(data_dict)
    df.to_pickle("../simba_rlf_data/df_nt.pkl")

    file.close()


    return None


def read_df_manga():

    path_manga = "../simba_rlf_data/manga_radio_vac_1.0.fits"
    tbl = Table.read(path_manga, hdu=1)

    # Find only the 1D columns
    good_columns = [name for name in tbl.colnames if len(tbl[name].shape) <= 1]

    # Convert only the 1D columns to pandas
    df = tbl[good_columns].to_pandas()

    # Find all columns that hold 'object' types (which includes strings/bytes)
    object_columns = df.select_dtypes(include=['object']).columns

    for col in object_columns:
        # Check if the first valid item is a byte-string
        if isinstance(df[col].dropna().iloc[0], bytes):
            df[col] = df[col].str.decode('utf-8')

    df = df[(df['log_mass'] > 10) & (df['log_mass'] < 12) & (df['log_ssfr'] > -16) &
            (df['log_ssfr'] < -9) & (~np.isnan(df['log_mass'])) & (~np.isnan(df['log_ssfr']))]
    
    df = df[df['in_erd_analysis']]
    df.to_pickle("../simba_rlf_data/df_manga.pkl")

    return None



def plot_mass_v_ssfr(df, ax, title=""):
    """
    Plots a 2D histogram of mass vs sSFR.
    Colors represent the log10 of the fraction of total galaxies in each bin.
    """
    x = df['log_mass']
    y = df['log_ssfr']
    
    nx = 15
    ny = 15

    # 1. Get the raw counts
    hist, xedges, yedges = np.histogram2d(x, y, bins=[nx, ny])
    hist = hist.T

    # 2. Convert counts to a fraction of the total plotted galaxies
    hist_frac = hist / hist.sum()

    # 3. Mask the empty bins (fraction == 0) to avoid log(0) errors
    hist_frac_masked = np.ma.masked_where(hist_frac == 0, hist_frac)

    # 4. Take the log10 of the fractions directly
    hist_log = np.log10(hist_frac_masked)

    xx, yy = np.meshgrid(xedges, yedges)

    # 5. Plot the log values standardly (no LogNorm needed)
    mesh = ax.pcolormesh(xx, yy, hist_log, cmap='viridis')
    
    ax.set_title(title, fontsize = 20)
    ax.set_xlabel(r'$ \log_{\rm 10} \left(M_{\star}/ M_\odot \right)$', fontsize = 15)
    ax.set_ylabel(r'$ \log_{\rm 10} \left(sSFR/~ yr^{-1} \right)$', fontsize = 15)
    ax.set_xticks([10, 10.5, 11, 11.5, 12])
    ax.set_yticks([-15, -13, -11, -9])
    ax.tick_params(labelsize =15)
    ax.set_xlim([10,12])
    ax.set_ylim(-16,-9)
    
    return mesh