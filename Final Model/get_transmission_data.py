
import pandas as pd
import numpy as np

def transmission_data(model_dir, trans_folder, S_t, T, I, cap_class, station_no, day, hour):
    transmission_data = model_dir + trans_folder + "transmission.xlsx"
    k_W_temp = pd.DataFrame(pd.read_excel(transmission_data, "transmission capacity")).to_dict()
    k_W = k_W_temp['trans cap']                                                     # effective transmission capacity

    p_WC_temp = pd.DataFrame(pd.read_excel(transmission_data, "conductor cost"))    # conductor cost
    p_WC_temp2 = {(r, c): p_WC_temp.at[r, c] for r in S_t for c in I}

    l_W_temp = pd.DataFrame(pd.read_excel(transmission_data, "trans line length"))  # transmission line length
    l_W_temp2 = {(r, c): l_W_temp.at[r, c] for r in S_t for c in I}

    p_WC = dict.fromkeys((range(cap_class)))
    l_W = dict.fromkeys((range(cap_class)))
    for i in I:
        p_WC[i] = p_WC_temp2[station_no - 1, i]
        l_W[i] = l_W_temp2[station_no - 1, i]

    p_WE_24 = pd.read_excel(transmission_data, "wholesale price")
    p_WE_annual = pd.DataFrame(np.tile(p_WE_24, day))
    p_WE_temp = {(r, c): p_WE_annual.at[r, c] for r in S_t for c in T}              # wholesale electricity cost

    p_WE = dict.fromkeys((range(hour)))
    for t in T:
        p_WE[t] = p_WE_temp[station_no - 1, t]

    return (l_W, k_W, p_WC, p_WE)