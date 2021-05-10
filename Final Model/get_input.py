# Define all parameters and read in all data inputs

import pandas as pd
import numpy as np
from define_parameters import *

# %% Set working directory:
if super_comp == 1:
    model_dir = '/home/anph/projects/HDV/Data/'
    load_folder = 'load_station/'
    solar_folder = 'solar/'
    trans_folder = 'transmission/'
    results_folder = '/home/anph/projects/HDV/Results/'
else:
    model_dir = 'C:\\Users\\atpha\\Documents\\Postdocs\\Projects\\HDV\\Data\\'
    load_folder = 'load_station\\'
    solar_folder = 'solar\\'
    trans_folder = 'transmission\\'
    results_folder = 'C:\\Users\\atpha\\Documents\\Postdocs\\Projects\\HDV\\Results\\'


# %% Define set:
if run_all_station == 0:
    no_station_to_run = 1
else:
    no_station_to_run = station

I = list(range(cap_class))          # Set of transmission capacity classes
S = list(range(no_station_to_run))  # Set of charging stations to run
T = list(range(hour))               # Set of hours to run
S_t = list(range(station))          # Set of total charging stations

# %% Import and calculate data input:
# Load station number:

# Load data:
load_data_raw = pd.read_csv(model_dir + load_folder + "TM2012_SPNation_TF-1_PT1_DPs1_BS1000_CP500_TD1_BD1_VW1_station.csv", header=None)
load_data_raw_2 = load_data_raw.iloc[:, 4:] / 1000
load_data_annual = pd.DataFrame(np.tile(load_data_raw_2, day))
d_E = {(r, c): load_data_annual.at[r, c] for r in S_t for c in T}

if run_all_station == 0:
    d_E_new = dict.fromkeys((range(hour)))
    for t in T:
        d_E_new[t] = d_E[station_no-1, t]
else:
    d_E = d_E

# Battery:
CRF_B = ir/(1-(1+ir)**(-life_B))        # Calculate battery capital recovery factor
p_BK = capex_B*CRF_B+fixed_OM_B         # Battery power rating cost
p_BC = p_BC_fixed*1000*8760             # Battery energy cost ($/MWh)

# H2:
p_HK = p_HK_fixed*1000*8760             # H2 power capacity cost
p_HC = p_HC_fixed*1000*8760             # H2 energy capacity cost

# SMR:
CRF_M = ir/(1-(1+ir)**(-life_M))
p_MK = capex_M*CRF_M+fixed_OM_M         # SMR capital cost
p_ME = var_OM_M + fuel_M                # SMR operating cost ($/MWh)

# Solar data:
solar_data = model_dir + solar_folder + "solar.xlsx"
f_P_annual = pd.read_excel(solar_data, "capacity factor")      # solar capacity factor (S x T)
f_P_annual_final = pd.DataFrame(np.tile(f_P_annual, 1))
f_P = {(r, c): f_P_annual_final.at[r, c] for r in S_t for c in T}

if run_all_station == 0:
    f_P_new = dict.fromkeys((range(hour)))
    for t in T:
        f_P_new[t] = f_P[station_no-1, t]
else:
    f_P = f_P


CRF_P = ir/(1-(1+ir)**(-life_P))
p_PK = capex_P*CRF_P+fixed_OM_P                             # solar annualized capacity cost

# Transmission data:
transmission_data = model_dir + trans_folder + "transmission.xlsx"
k_W_temp = pd.DataFrame(pd.read_excel(transmission_data, "transmission capacity")).to_dict()
k_W = k_W_temp['trans cap']                                # effective transmission capacity

p_WC_temp = pd.DataFrame(pd.read_excel(transmission_data, "conductor cost"))    # conductor cost
p_WC = {(r, c): p_WC_temp.at[r, c] for r in S_t for c in I}

l_W_temp = pd.DataFrame(pd.read_excel(transmission_data, "trans line length"))  # transmission line length
l_W = {(r, c): l_W_temp.at[r, c] for r in S_t for c in I}

if run_all_station == 0:
    p_WC_new = dict.fromkeys((range(cap_class)))
    l_W_new = dict.fromkeys((range(cap_class)))
    for i in I:
        p_WC_new[i] = p_WC[station_no - 1, i]
        l_W_new[i] = l_W[station_no - 1, i]
else:
    p_WC = p_WC
    l_W = l_W

p_WE_24 = pd.read_excel(transmission_data, "wholesale price")
p_WE_annual = pd.DataFrame(np.tile(p_WE_24, day))
p_WE = {(r, c): p_WE_annual.at[r, c] for r in S_t for c in T}                     # wholesale electricity cost

if run_all_station == 0:
    p_WE_new = dict.fromkeys((range(hour)))
    for t in T:
        p_WE_new[t] = p_WE[station_no-1, t]
else:
    p_WE = p_WE
