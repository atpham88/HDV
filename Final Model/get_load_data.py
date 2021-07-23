
import pandas as pd
import numpy as np

def load_data(model_dir, load_folder, S_t, T, hour, day, station_no, load_pr_case, h2_demand_p, h2_convert, inc_h2_demand):
    if load_pr_case == 1:
        load_case = 'station_load_profile_case1.xlsx'
    elif load_pr_case == 2:
        load_case = 'station_load_profile_case2.xlsx'
    elif load_pr_case == 3:
        load_case = 'station_load_profile_case3.xlsx'

    # Total load (before subtracting h2 demand):
    load_data_raw = pd.read_excel(model_dir + load_folder + load_case, header=None)
    load_data_raw_2 = load_data_raw.iloc[:, 4:] / 1000
    load_data_annual = pd.DataFrame(np.tile(load_data_raw_2, day))
    d_E_temp = {(r, c): load_data_annual.at[r, c] for r in S_t for c in T}

    d_E_wH2 = dict.fromkeys((range(hour)))
    d_H_bar_MWh = dict.fromkeys((range(hour)))
    d_H_bar = dict.fromkeys((range(hour)))
    d_E = dict.fromkeys((range(hour)))
    for t in T:
        d_E_wH2[t] = d_E_temp[station_no - 1, t]
        if inc_h2_demand == 1:
            d_H_bar_MWh[t] = d_E_wH2[t] * h2_demand_p
        elif inc_h2_demand == 0:
            d_H_bar_MWh[t] = 0
        d_H_bar[t] = d_H_bar_MWh[t] * h2_convert
        d_E[t] = d_E_wH2[t] - d_H_bar_MWh[t]

    return d_E, d_H_bar
