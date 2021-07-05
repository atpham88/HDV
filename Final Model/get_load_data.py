
import pandas as pd
import numpy as np

def load_data(model_dir, load_folder, S_t, T, hour, day, station_no, load_pr_case, inc_h2_demand):
    if load_pr_case == 1:
        load_case = 'station_load_profile_case1.xlsx'
        h2_load_case = 'h2_load_profile_case1.xlsx'
    elif load_pr_case == 2:
        load_case = 'station_load_profile_case2.xlsx'
        h2_load_case = 'h2_load_profile_case2.xlsx'
    elif load_pr_case == 3:
        load_case = 'station_load_profile_case3.xlsx'
        h2_load_case = 'h2_load_profile_case3.xlsx'

    # Total load (before substracting h2 demand):
    load_data_raw = pd.read_excel(model_dir + load_folder + load_case, header=None)
    load_data_raw_2 = load_data_raw.iloc[:, 4:] / 1000
    load_data_annual = pd.DataFrame(np.tile(load_data_raw_2, day))
    d_E_temp = {(r, c): load_data_annual.at[r, c] for r in S_t for c in T}

    d_E = dict.fromkeys((range(hour)))
    for t in T:
        d_E[t] = d_E_temp[station_no - 1, t]

    h2_load_data_raw = pd.read_excel(model_dir + load_folder + h2_load_case, header=None)
    h2_load_data_raw_2 = h2_load_data_raw.iloc[:, 4:] / 1000
    h2_load_data_annual = pd.DataFrame(np.tile(h2_load_data_raw_2, day))
    d_H_bar_temp = {(r, c): h2_load_data_annual.at[r, c] for r in S_t for c in T}

    d_H_bar = dict.fromkeys((range(hour)))
    if inc_h2_demand == 1:
        for t in T:
            d_H_bar[t] = d_H_bar_temp[station_no - 1, t]
    elif inc_h2_demand == 0:
        for t in T:
            d_H_bar[t] = d_H_bar_temp[station_no - 1, t]*0

    return d_E, d_H_bar
