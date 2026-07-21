import sys
sys.path.append("..")

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import astropy.io.fits as fits
from astropy.table import Table
import h5py
import caesar
from scipy.stats import rv_continuous
from scipy.integrate import cumulative_trapezoid
from scipy.integrate import quad
from scipy.interpolate import interp1d
from scipy.stats import binom

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


def suppression_factor(log_lr, log_lc):

  '''
  Returns --> Suppression factor "f".
  Inputs --> Radio Luminosity 'Lr' and 
  the Hopkins characteristic radio luminosity 'Lc'
  
  Refer to 2003ApJ...599..971H for further information.
  '''

  f = (0.1 + 0.9 * (10**log_lr/10**log_lc)**0.3)**-1

  return np.log10(f)


def hopkins_log_sfr(log_lr, log_lc, intercept, dex_difference_imf):
    
  '''
  Returns --> Star formation rate according to the Hopkins relation (Can include a dex difference in SFR to
  account for the usage of IMFs different than Salpeter. Set dex_difference to 0 for Salpeter).
  Inputs --> The radio luminosity at which SFR is required. Lc and the log space y-intercept of the
  Hopkins L-SFR relation'''

  logsfr = log_lr - intercept - dex_difference_imf

  if log_lr < 37.95:
    logsfr = suppression_factor(log_lr, log_lc) + logsfr

  return logsfr

def generate_lsfr_interp_func():

    v_hopkins_log_sfr = np.vectorize(hopkins_log_sfr)
    log_lc = 37.95 ## Characteristic luminosity from Hopkins.
    log_lrs = np.linspace(1,50,1000)
    log_lr_sf_interp = interp1d(v_hopkins_log_sfr(log_lrs,log_lc,37.4,0.25), log_lrs)

    np.save("../simba_rlf_data/log_lr_sf_interp.npy", log_lr_sf_interp ,allow_pickle= True)

    return log_lr_sf_interp

def get_log_lr_sf_sb(logsfr, log_lr_sf_interp_func):

    if np.isnan(logsfr):
        loglr = -np.inf
    
    elif np.isinf(logsfr):
        loglr = -np.inf
    
    else:
        loglr = log_lr_sf_interp_func(logsfr)
    
    return loglr



def generate_mock_df(volume):

    log_M_star = 10.66
    phi1 = 3.96e-3
    alpha1 = -0.35
    phi2 = 0.79e-3
    alpha2 = -1.47

    num_samples = volume*quad(lambda log_mass: baldry_gsmf_logspace(10**log_mass,log_M_star,
                                                                    phi1, alpha1, phi2, alpha2), 8, 11.5)[0]
    
    num_samples = int(num_samples)

    log_mass = generate_mock_galaxies_baldry(log_M_star, phi1, alpha1, phi2, alpha2, num_samples)
    log_mbh = log_mass-3


    log_lam = []
    for logmass in log_mass:

        alpha_0 = 2.01
        beta = -5.02
        lambda_star = 0.01
        log_lambda_min = -5.61

        loglam = generate_sb24_erd(logmass, alpha_0, beta, lambda_star, log_lambda_min,1)[0]
        log_lam.append(loglam)
    
    log_lam = np.array(log_lam)
    log_lr_agn = get_log_lr_agn_of_lambda(log_lam, log_mbh)

    log_lr_agn_whz = log_lr_agn - 16.15

    
    dict = {"log_mass": log_mass,"log_mbh" : log_mbh, "log_lam" : log_lam, "log_lr_agn" : log_lr_agn,
            "log_lr_agn_whz" : log_lr_agn_whz}

    df = pd.DataFrame(dict)
    df.to_pickle("../simba_rlf_data/df_mock.pkl")

    return df


def ms07_agn_rlf(log_lr_agn_bin):

    C = 10**(-5.5) 
    P_star = 10**(24.59)
    alpha = 1.27
    beta = 0.49
    
    lr_agn_bin = 10**log_lr_agn_bin
    phi = C/((lr_agn_bin/P_star)**alpha + (lr_agn_bin/P_star)**beta)

    phi = phi 
    return np.log10(phi)


def load_simba_df():

    log_sfr = []
    bhm_dot = []
    log_bhm = []
    fedd = []
    log_ssfr = []
    log_mass = []
    log_lam_sb = []
    log_lr_agn_sb = []
    log_lr_sf_sb = []
    log_lr_total_sb = []
    log_lr_sf_interp_func = np.load("../simba_rlf_data/log_lr_sf_interp.npy", allow_pickle= True)[()]

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
        

            alpha_0 = 2.01
            beta = -5.02
            lambda_star = 0.01
            log_lambda_min = -5.61

            loglam = generate_sb24_erd(logmass, alpha_0, beta, lambda_star, log_lambda_min,1)[0]
            log_lam_sb.append(loglam)

            loglr_agn = get_log_lr_agn_of_lambda(loglam, logbhm)
            log_lr_agn_sb.append(loglr_agn)

        
            loglr_sf_sb = get_log_lr_sf_sb(logsfr, log_lr_sf_interp_func)
            log_lr_sf_sb.append(loglr_sf_sb)

            log_lr_total_sb.append(np.log10(10**loglr_agn + 10**loglr_sf_sb))




        path_nt = "../simba_rlf_data/BH_Bondi_m100n1024_" + num + ".hdf5"
        data_nt = h5py.File(path_nt)

        bhm_nt_this_snap = np.log10(data_nt['BH_Mass'])
        bhm_dot_50Myr_this_snap = 10**np.array(data_nt['BHAR_50MYR'])   #/0.68

        bhm_nt.extend(bhm_nt_this_snap.tolist())
        bhm_dot_50Myr.extend(bhm_dot_50Myr_this_snap.tolist())



    dict = {"log_sfr" : log_sfr, 'bhmdot' : bhm_dot, "log_bhm" : log_bhm, "fedd" : fedd, 
    "log_ssfr" : log_ssfr, "log_mass" : log_mass,
    "bhm_nt": bhm_nt, "bhm_dot_50Myr" : bhm_dot_50Myr,
    "log_lam_sb": log_lam_sb, "log_lr_agn_sb" : log_lr_agn_sb, "log_lr_sf_sb" : log_lr_sf_sb,
    "log_lr_total_sb": log_lr_total_sb}


    df = pd.DataFrame(dict)


    # Calculate luminosities
    lsf_thermal = get_sf_radio_thermal_nt(df['log_sfr'])
    lsf_non_thermal = get_sf_radio_non_thermal_nt(df['log_sfr'])
    log_lsf_nt = get_sf_radio_nt(df['log_sfr'])
    log_lr_agn_nt = get_agn_radio_nt(df['bhm_dot_50Myr'], df['fedd'], df['log_bhm'], df['log_mass'])
    log_lr_agn_nt_whz = log_lr_agn_nt - 16.15
    log_lr_total_nt = get_total_radio_nt(df['log_sfr'], df['bhm_dot_50Myr'],
                                           df['fedd'], df['log_bhm'], df['log_mass'])
    
    log_lam_pcav_nt = np.log10(get_lambda_pcav(log_lr_agn_nt,log_bhm))
    
    # Add to dataframe in log space (fixed lsf_t typo)
    df['log_lsf_thermal'] = np.log10(lsf_thermal)
    df['log_lsf_non_thermal'] = np.log10(lsf_non_thermal)
    df['log_lsf_nt'] = log_lsf_nt
    df['log_lr_agn_nt'] = log_lr_agn_nt
    df['log_lr_agn_nt_whz'] = log_lr_agn_nt_whz
    df['log_lr_total_nt'] = log_lr_total_nt
    df['log_lam_pcav_nt'] = log_lam_pcav_nt

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



def bin_galaxies_3mbins(property1,df): # Input exact column name as in Dataframe (str)
 
  '''Returns list of arrrays containing data-AGN-fraction as a function of required property, 
  values of galaxy the property, model-AGN-fraction as a function of required property
  and the errorbars on the model.

  Inputs column name of property, manga dataframe, model dataframe,'''

  if property1 == 'log_ssfr':
    nbins = [-np.inf,-11.5,-8]

  else:
    nbins = [10,10.67,11.34,12]    

  sorted_byprop1 = df.sort_values(by = [property1])
  binned_byprop1 = pd.cut(sorted_byprop1[property1], bins= nbins)
  df[property1 + "_bin"] = binned_byprop1

  return df


def bin_galaxies_5mbins(property1,df): # Input exact column name as in Dataframe (str)
 
  '''Returns list of arrrays containing data-AGN-fraction as a function of required property, 
  values of galaxy the property, model-AGN-fraction as a function of required property
  and the errorbars on the model.

  Inputs column name of property, manga dataframe, model dataframe,'''

  if property1 == 'log_ssfr':
    nbins = [-np.inf,-11.5,-8]

  else:
    nbins = [10,10.4,10.8,11.2,11.6,12]    

  sorted_byprop1 = df.sort_values(by = [property1])
  binned_byprop1 = pd.cut(sorted_byprop1[property1], bins= nbins)
  df[property1 + "_bin"] = binned_byprop1

  return df

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


def baldry_gsmf_logspace(M,log_M_star, phi1, alpha1, phi2, alpha2):

    M_star = 10**log_M_star
    func = (np.log(10))*np.exp(-M/M_star)*(phi1*(M/M_star)**(alpha1+1) + phi2*(M/M_star)**(alpha2+1))

    return func


def baldry_gsmf(M,log_M_star, phi1, alpha1, phi2, alpha2):

    M_star = 10**log_M_star
    func = np.exp(-M/M_star)*(phi1*(M/M_star)**(alpha1) + phi2*(M/M_star)**(alpha2))/M_star

    return func


def generate_mock_galaxies_baldry(log_M_star, phi1, alpha1, phi2, alpha2, num_samples):

    lower_bound = 8
    upper_bound = 11.5

    log_M_grid = np.linspace(lower_bound, upper_bound, 10000)
    pdf = baldry_gsmf_logspace(10**log_M_grid,log_M_star, phi1, alpha1, phi2, alpha2)

    cdf = cumulative_trapezoid(pdf,log_M_grid, initial = 0)
    cdf = cdf/cdf[-1]


    inverse_cdf  = interp1d(cdf, log_M_grid)
    unif = np.random.uniform(0,1,num_samples)
    log_M_sample = inverse_cdf(unif)

    return log_M_sample


def schechter_function(lam,log_mass,alpha_0,beta,lambda_star):

    alpha = alpha_0*(log_mass/10.5)**(beta)
    f = np.exp(-lam/lambda_star)*(lam/lambda_star)**(-alpha+1)

    return f

def schechter_function_logspace(lam,log_mass,alpha_0,beta,lambda_star):

    alpha = alpha_0*(log_mass/10.5)**(beta)
    f = (np.log(10)*lambda_star)*np.exp(-lam/lambda_star)*(lam/lambda_star)**(-alpha+1)

    return f


def generate_sb24_erd(log_mass,alpha_0, beta, lambda_star, log_lambda_min, num_samples):

    lower_bound = log_lambda_min
    upper_bound = 2

    log_lambda_grid = np.linspace(lower_bound, upper_bound)
    pdf = schechter_function_logspace(10**log_lambda_grid,log_mass,alpha_0,beta,lambda_star)

    cdf = cumulative_trapezoid(pdf,log_lambda_grid, initial=0)
    cdf = cdf/cdf[-1]

    inverse_cdf = interp1d(cdf, log_lambda_grid)
    unif_sample = np.random.uniform(0,1,num_samples)

    log_lam_sample = inverse_cdf(unif_sample)

    return log_lam_sample


def get_log_lr_agn_of_lambda(log_lam, log_mbh):

    mbh = 10**log_mbh
    Ledd = 1.26e38*mbh
    log_pcav = np.log10(10**log_lam*Ledd)
    log_lr = (log_pcav - 13.91)/0.75

    return log_lr



def get_log_pcav(log_lr):

  log_pcav = 0.75*log_lr + 13.91

  return log_pcav


def get_lambda_pcav(log_lr, log_bhm):

  log_pcav = get_log_pcav(log_lr)
  log_L_Edd= np.float64(np.log10(1.26e38)+log_bhm)
  L_Edd= 10**log_L_Edd

  lam = 10**log_pcav/L_Edd

  return lam


def sfr_ms(log_mass):

    log_sfr = 0.76*log_mass - 7.64 #From Renzini et al. 2015

    return log_sfr


def get_quenched_fraction(log_mass):

    mass = 10**log_mass
    M_star = 10**10.56

    f_q = 1 - np.exp((-mass/M_star)**0.8) # Ignoring environmental quenching from Baldry

    return f_q



def get_binom_ebars(frac, size):


    ppf_low = binom.ppf(0.16,size, frac)
    ppf_median = binom.ppf(0.5,size, frac)
    ppf_high = binom.ppf(0.84,size, frac)

    frac_low = ppf_low/size
    frac_high = ppf_high/size

    ebar_low = frac - frac_low
    ebar_high = frac_high - frac

    return [ebar_low, ebar_high]


def get_Fagn_of_lcrit(df, lambda_crit, radio_model):

    if radio_model == 'NT':
        string = 'log_lam_pcav_nt'
    
    if radio_model == "SB":
        string = 'log_lam_sb'

    is_strong_ragn = []
    for i in range(len(df)):

        lam = 10**df[string][i]

        if lam > lambda_crit:
            is_strong_ragn.append(True)

        else:
            is_strong_ragn.append(False)


    df['is_strong_ragn'] = is_strong_ragn

    df_mass_bins = bin_galaxies_5mbins("log_mass", df)
    grped_df_mass = df_mass_bins.groupby('log_mass_bin')

    size = []
    frac = []
    ebars_low = []
    ebars_high = []
    mass_center = []

    frac_q = []
    frac_sf = []
    ebars_q_low = []
    ebars_q_high = []

    size_q = []
    size_sf = []
    ebars_sf_low = []
    ebars_sf_high = []

    for mass, group in grped_df_mass:

        string = str(mass)
        split = string.split(',')
        left = float(split[0][1:])
        right = float(split[1][:-1])
        center = (left + right)/2
        mass_center.append(center)

        n_agn_all = group['is_strong_ragn'].sum()
        size_all = len(group)
        f_all = n_agn_all/size_all
        size.append(size_all)
        frac.append(f_all)

        [ebar_low, ebar_high] = get_binom_ebars(f_all,size_all)
        ebars_low.append(ebar_low)
        ebars_high.append(ebar_high)

        s_q = 0
        n_agn_q = 0

        s_sf = 0
        n_agn_sf = 0

        for i in range(len(group)):
            
            if group['log_ssfr'].iloc[i] < -10.8:

                s_q = s_q + 1

                if group['is_strong_ragn'].iloc[i] == True:
                    n_agn_q = n_agn_q + 1


            if group['log_ssfr'].iloc[i] > -10.8:

                s_sf = s_sf + 1

                if group['is_strong_ragn'].iloc[i] == True:
                    n_agn_sf = n_agn_sf + 1
        

        
        # --- Quiescent Population (q) ---
        size_q.append(s_q)
        if s_q > 0:
            frac_q.append(n_agn_q / s_q)
            ebar_q_low, ebar_q_high = get_binom_ebars(n_agn_q / s_q, s_q)
            ebars_q_low.append(ebar_q_low)
            ebars_q_high.append(ebar_q_high)
        else:
            # If size is 0, fraction and error bars are mathematically undefined
            frac_q.append(float('nan'))
            ebars_q_low.append(float('nan'))
            ebars_q_high.append(float('nan'))

        # --- Star-Forming Population (sf) ---
        size_sf.append(s_sf)
        if s_sf > 0:
            frac_sf.append(n_agn_sf / s_sf)
            ebar_sf_low, ebar_sf_high = get_binom_ebars(n_agn_sf / s_sf, s_sf)
            ebars_sf_low.append(ebar_sf_low)
            ebars_sf_high.append(ebar_sf_high)
        else:
            # Safe fallback to keep list lengths synchronized
            frac_sf.append(float('nan'))
            ebars_sf_low.append(float('nan'))
            ebars_sf_high.append(float('nan'))
        

    mass_center = np.array(mass_center)
    frac = np.array(frac)
    ebars_low = np.array(ebars_low)
    ebars_high = np.array(ebars_high)

    size_q = np.array(size_q)
    frac_q = np.array(frac_q)
    ebars_q_low = np.array(ebars_q_low)
    ebars_q_high = np.array(ebars_q_high)

    size_sf = np.array(size_sf)
    frac_sf = np.array(frac_sf)
    ebars_sf_low = np.array(ebars_sf_low)
    ebars_sf_high = np.array(ebars_sf_high)

    return [mass_center,frac,ebars_low,ebars_high,frac_q,ebars_q_low,ebars_q_high,frac_sf,ebars_sf_low, ebars_sf_high]



def plot_Fagn_mbins_constraint():
    # Bin edges and midpoints
    bin_edges = np.array([10.0, 10.4, 10.8, 11.2, 11.6, 12.0])
    Ms = 0.5 * (bin_edges[:-1] + bin_edges[1:])

    # Log F_AGN values (from board)
    log_Fracs_q = np.array([-0.41, -1.41, -1.66, -1.17, -0.84])
    log_Fracs_sf = np.array([-0.11, -1.62, -1.58, -1.10, -0.65])

    # Errors only for last 3 Q bins and last 4 SF bins
    log_ebars_q = np.array([[0.10, 0.05, 0.05],    # lower
                            [0.09, 0.05, 0.05]])   # upper
    log_ebars_sf = np.array([[0.4, 0.22, 0.13, 0.22],
                             [0.37, 0.19, 0.12, 0.17]])

    # Colors
    red_hex = '#d62728'
    blue_hex = '#1f77b4'

    # Convert to linear
    Fracs_q = 10**log_Fracs_q
    Fracs_sf = 10**log_Fracs_sf

    # Compute asymmetric errors in linear space
    q_err_low = Fracs_q[2:] - 10**(log_Fracs_q[2:] - log_ebars_q[0])
    q_err_high = 10**(log_Fracs_q[2:] + log_ebars_q[1]) - Fracs_q[2:]

    sf_err_low = Fracs_sf[1:] - 10**(log_Fracs_sf[1:] - log_ebars_sf[0])
    sf_err_high = 10**(log_Fracs_sf[1:] + log_ebars_sf[1]) - Fracs_sf[1:]

    # === Plot Quiescent error bands (bins 3–5) ===
    q_edges = bin_edges[2:]  # From 10.8 to 12
    for i in range(3):  # 3 bins (i=0 to 2)
        x = [q_edges[i], q_edges[i+1]]
        y_low = Fracs_q[i+2] - q_err_low[i]
        y_high = Fracs_q[i+2] + q_err_high[i]
        plt.fill_between(x, [y_low]*2, [y_high]*2, color=red_hex, alpha=0.5, label='Intrinsic Quiescent' if i == 0 else "")
        
    # === Plot Star-Forming error bands (bins 2–5) ===
    sf_edges = bin_edges[1:]  # From 10.4 to 12
    for i in range(4):  # 4 bins (i=0 to 3)
        x = [sf_edges[i], sf_edges[i+1]]
        y_low = Fracs_sf[i+1] - sf_err_low[i]
        y_high = Fracs_sf[i+1] + sf_err_high[i]
        plt.fill_between(x, [y_low]*2, [y_high]*2, color=blue_hex, alpha=0.5, label='Intrinsic Star forming' if i == 0 else "")


    # === Upper limits for first 2 Q bins ===
    for i in [0, 1]:
        plt.fill_between([bin_edges[i], bin_edges[i+1]], [0, 0], [10**log_Fracs_q[i]]*2,
                         color=red_hex, alpha=0.5)

    # === Upper limit for first SF bin ===
    plt.fill_between([bin_edges[0], bin_edges[1]], [0, 0], [10**log_Fracs_sf[0]]*2,
                     color=blue_hex, alpha=0.5)



def plot_gsmf_mock(df_mock,volume,ax):

    # GSMF
    log_M_star = 10.66
    phi1 = 3.96e-3
    alpha1 = -0.35
    phi2 = 0.79e-3
    alpha2 = -1.47

    log_mass_mock = df_mock['log_mass']

    nbins_gsmf = 15
    bins_gsmf = np.linspace(8,11.5,nbins_gsmf)
    bins_centers_gsmf = (bins_gsmf[1:] + bins_gsmf[:-1])/2
    bin_width_gsmf = (11.5 - 8)/nbins_gsmf

    counts_mock, _ = np.histogram(log_mass_mock, bins = bins_gsmf)
    counts_per_dex_mock = counts_mock/bin_width_gsmf

    log_gsmf_mock = np.log10(counts_per_dex_mock/volume)
    log_gsmf_baldry = np.log10(baldry_gsmf_logspace(10**bins_centers_gsmf,
                                                    log_M_star, phi1, alpha1, phi2, alpha2))

    ax.plot(bins_centers_gsmf, log_gsmf_mock, label = 'Mock Data')
    ax.plot(bins_centers_gsmf, log_gsmf_baldry, label = 'Baldry GSMF Expression')
    ax.legend(loc = 'lower left')
    ax.set_xlabel(r"$\log \left( M_{\star}/ M_{\odot} \right)$", fontsize = 10)
    ax.set_ylabel(r"$\log \left( \Phi/ \rm Mpc^{-3}~dex^{-1} \right)$", fontsize = 10)



def plot_gsmf_simba(df_simba,volume,ax):

    # GSMF
    log_M_star = 10.66
    phi1 = 3.96e-3
    alpha1 = -0.35
    phi2 = 0.79e-3
    alpha2 = -1.47

    log_mass_simba = df_simba['log_mass']

    nbins_gsmf = 15
    bins_gsmf = np.linspace(8,11.5,nbins_gsmf)
    bins_centers_gsmf = (bins_gsmf[1:] + bins_gsmf[:-1])/2
    bin_width_gsmf = (11.5 - 8)/nbins_gsmf

    counts_simba, _ = np.histogram(log_mass_simba, bins = bins_gsmf)
    counts_per_dex_simba = counts_simba/bin_width_gsmf

    log_gsmf_simba = np.log10(counts_per_dex_simba/volume)
    log_gsmf_baldry = np.log10(baldry_gsmf_logspace(10**bins_centers_gsmf,
                                                    log_M_star, phi1, alpha1, phi2, alpha2))

    ax.plot(bins_centers_gsmf, log_gsmf_simba, label = 'SIMBA')
    ax.plot(bins_centers_gsmf, log_gsmf_baldry, label = 'Baldry GSMF Expression')
    ax.set_xlabel(r"$\log \left( M_{\star}/ M_{\odot} \right)$", fontsize = 10)
    ax.set_ylabel(r"$\log \left( \Phi/ \rm Mpc^{-3}~dex^{-1} \right)$", fontsize = 10)
    ax.legend(loc = 'lower left')




def plot_rlf_mock_sb24(df_mock, volume, ax):
    log_lr_agn_sb24 = df_mock['log_lr_agn_whz']

    nbins = 15
    log_lr_min = 19.5
    log_lr_max = 26.5
    bins = np.linspace(log_lr_min,log_lr_max, nbins)
    bin_centers = (bins[:-1] + bins[1:])/2
    bin_width = (log_lr_max - log_lr_min)/nbins

    counts_sb24, _ =  np.histogram(log_lr_agn_sb24, bins = bins)
    counts_per_dex_sb24 = counts_sb24/bin_width
    rlf_sb24 = counts_per_dex_sb24/volume
    log_rlf_sb24 = np.log10(rlf_sb24)

    # MS07 RLF
    log_rlf_ms07 = ms07_agn_rlf(bin_centers)

    ax.plot(bin_centers, log_rlf_sb24, label = 'SB24')
    ax.plot(bin_centers, log_rlf_ms07, label = 'MS07')
    ax.legend(loc = 'lower left')
    ax.set_xlabel(r"$\log \left( L_{\rm 1.4~GHz}/ \rm W~Hz^{-1} \right)$", fontsize = 10)
    ax.set_ylabel(r"$\log \left( \Phi/ \rm Mpc^{-3}~dex^{-1} \right)$", fontsize = 10)
    plt.tight_layout()


def plot_rlf_simba_nt21(df_simba, volume, ax):
    log_lr_agn_nt21 = df_simba['log_lr_agn_nt_whz']

    nbins = 15
    log_lr_min = 19.5
    log_lr_max = 26.5
    bins = np.linspace(log_lr_min,log_lr_max, nbins)
    bin_centers = (bins[:-1] + bins[1:])/2
    bin_width = (log_lr_max - log_lr_min)/nbins

    counts_nt21, _ =  np.histogram(log_lr_agn_nt21, bins = bins)
    counts_per_dex_nt21 = counts_nt21/bin_width
    rlf_nt21 = counts_per_dex_nt21/volume
    log_rlf_nt21 = np.log10(rlf_nt21)

    # MS07 RLF
    log_rlf_ms07 = ms07_agn_rlf(bin_centers)

    ax.plot(bin_centers, log_rlf_nt21, label = 'NT21')
    ax.plot(bin_centers, log_rlf_ms07, label = 'MS07')
    ax.legend(loc = 'lower left')
    ax.set_xlabel(r"$\log \left( L_{\rm 1.4~GHz}/ \rm W~Hz^{-1} \right)$", fontsize = 10)
    ax.set_ylabel(r"$\log \left( \Phi/ \rm Mpc^{-3}~dex^{-1} \right)$", fontsize = 10)

    plt.tight_layout()


def plot_rld_of_M_mock_sb24(df_mock, ax):

    df_mass_bins = bin_galaxies_3mbins("log_mass", df_mock)
    grped_df_mass = df_mass_bins.groupby('log_mass_bin')

    for mass, group in grped_df_mass:

        string = str(mass)
        split = string.split(',')
        left = float(split[0][1:])
        right = float(split[1][:-1])
        mass_center = str(float(f"{(left + right) / 2:.4g}"))
        
        mass_label = r"$\log_{\rm 10} \left( M_{\star}/ M_{\odot}\right) \sim ~$" + mass_center

        log_lr_agn_sb24 = group['log_lr_agn_whz']
        nbins = 10
        log_lr_min = 19.5
        log_lr_max = 26.5
        bins = np.linspace(log_lr_min,log_lr_max, nbins)
        bin_centers = (bins[:-1] + bins[1:])/2
        bin_width = (log_lr_max - log_lr_min)/nbins

        counts, _ = np.histogram(log_lr_agn_sb24, bins = bins)
        counts_per_dex = counts/bin_width 

        rld = counts_per_dex/counts.sum()
        log_rld = np.log10(rld)

        ax.plot(bin_centers, log_rld, label = mass_center)
        ax.legend(loc = 'lower left')
        ax.set_xlabel(r"$\log \left( L_{\rm 1.4~GHz}/ \rm W~Hz^{-1} \right)$", fontsize = 10)
        ax.set_ylabel(r"$\log \left( PDF/ \rm ~dex^{-1} \right)$", fontsize = 10)
        plt.tight_layout()



def plot_rld_of_M_simba_nt21(df_simba, ax):

    df_mass_bins = bin_galaxies_3mbins("log_mass", df_simba)
    grped_df_mass = df_mass_bins.groupby('log_mass_bin')

    for mass, group in grped_df_mass:

        string = str(mass)
        split = string.split(',')
        left = float(split[0][1:])
        right = float(split[1][:-1])
        mass_center = str(float(f"{(left + right) / 2:.4g}"))
        
        mass_label = r"$\log_{\rm 10} \left( M_{\star}/ M_{\odot}\right) \sim ~$" + mass_center

        log_lr_agn_nt21 = group['log_lr_agn_nt_whz']
        nbins = 10
        log_lr_min = 19.5
        log_lr_max = 26.5
        bins = np.linspace(log_lr_min,log_lr_max, nbins)
        bin_centers = (bins[:-1] + bins[1:])/2
        bin_width = (log_lr_max - log_lr_min)/nbins

        counts, _ = np.histogram(log_lr_agn_nt21, bins = bins)
        counts_per_dex = counts/bin_width 

        rld = counts_per_dex/counts.sum()
        log_rld = np.log10(rld)

        ax.plot(bin_centers, log_rld, label = mass_center)
        ax.legend(loc = 'lower left')
        ax.set_xlabel(r"$\log \left( L_{\rm 1.4~GHz}/ \rm W~Hz^{-1} \right)$", fontsize = 10)
        ax.set_ylabel(r"$\log \left( PDF/ \rm ~dex^{-1} \right)$", fontsize = 10)
        plt.tight_layout()