import pandas as pd
import numpy as np

# %% Switches:
super_comp = 0  # ==1: run on super computer, ==0: run on local laptop

# %% Set working/results directory:
if super_comp == 1:
    model_dir = '/storage/work/a/akp5369/Model_in_Python/HDV/Data/'
    load_folder = 'load_station/'
    solar_folder = 'solar/'
    trans_folder = 'transmission/'
    results_folder = '/storage/work/a/akp5369/Model_in_Python/HDV/Results/'
else:
    model_dir = 'C:\\Users\\atpha\\Documents\\Postdocs\\Projects\\HDV\\Data\\'
    load_folder = 'load_station\\'
    solar_folder = 'solar\\'
    trans_folder = 'transmission\\'
    results_folder = 'C:\\Users\\atpha\\Documents\\Postdocs\\Projects\\HDV\\Results\\'

# %% Model scope:
cap_class = 7  # number of transmission capacity classes
station = 170  # Number of charging stations
hour = 24  # Number of hours

# %% Define set:
I = list(range(cap_class))  # Set of transmission capacity classes (7 classes)
S = list(range(station))  # Set of charging stations (170 stations)
T = list(range(hour))  # Set of hours in a year (8760 hours)

# Load profile data:
load_data_raw = pd.read_csv(model_dir + load_folder + "TM2012_SPNation_TF-1_PT1_DPs1_BS1000_CP500_TD1_BD1_VW1_station.csv", header=None)
load_data_raw_2 = load_data_raw.iloc[:, 4:] / 1000
load_data_annual = pd.DataFrame(np.tile(load_data_raw_2, 1))
d_E = {(r, c): load_data_annual.at[r, c] for r in S for c in T}

# Storage data:
# Battery:
r_B = 0.2  # ramp rate for battery
p_BK = 90000  # battery power rating cost
p_BC = 600000  # battery capital cost
p_BE = 0  # battery operating cost

# H2:
r_H = 0.2  # ramp rate for H2
p_HK = 90000  # H2 capital cost
p_HC = 600000  # H2 capital cost
p_HE = 0  # H2 operating cost

# SMR data:
r_M = 0.2  # ramp rate for SMR (percentage of capacity)
k_M = 60  # SMR capacity per module
p_MK = 90000  # SMR capital cost
p_ME = 25  # SMR operating cost

# Solar data:
solar_data = model_dir + solar_folder + "solar.xlsx"
f_P_24 = pd.read_excel(solar_data, "capacity factor")  # solar capacity factor (S x T)
f_P_annual = pd.DataFrame(np.tile(f_P_24, 1))
f_P = {(r, c): f_P_annual.at[r, c] for r in S for c in T}

p_PK = 150000  # solar capacity cost
p_PE = 0  # solar operating cost

# Transmission data:
transmission_data = model_dir + trans_folder + "transmission.xlsx"
k_W_temp = pd.DataFrame(pd.read_excel(transmission_data, "transmission capacity")).to_dict()
k_W = k_W_temp['trans cap']  # effective transmission capacity (I)

p_WK = 100000  # transmission capacity cost
p_WO = 0.1  # transmission over-head
p_WI = 50000  # transmission infrastructure cost

p_WC_temp = pd.DataFrame(pd.read_excel(transmission_data, "conductor cost"))  # conductor cost (S x I)
p_WC = {(r, c): p_WC_temp.at[r, c] for r in S for c in I}

p_WL = 200000  # land cost (S x I)

l_W_temp = pd.DataFrame(pd.read_excel(transmission_data, "trans line length"))  # transmission line length
l_W = {(r, c): l_W_temp.at[r, c] for r in S for c in I}

p_WE_24 = pd.read_excel(transmission_data, "wholesale price")
p_WE_annual = pd.DataFrame(np.tile(p_WE_24, 1))
p_WE = {(r, c): p_WE_annual.at[r, c] for r in S for c in T}  # wholesale electricity cost (S x T)
