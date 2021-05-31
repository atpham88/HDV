
import pandas as pd
import numpy as np

def load_data(model_dir, load_folder, S_t, T, hour, day, station_no):

    load_data_raw = pd.read_csv(model_dir + load_folder + "TM2012_SPNation_TF-1_PT1_DPs1_BS1000_CP500_TD1_BD1_VW1_station.csv", header=None)
    load_data_raw_2 = load_data_raw.iloc[:, 4:] / 1000
    load_data_annual = pd.DataFrame(np.tile(load_data_raw_2, day))
    d_E_temp = {(r, c): load_data_annual.at[r, c] for r in S_t for c in T}

    d_E = dict.fromkeys((range(hour)))
    for t in T:
        d_E[t] = d_E_temp[station_no - 1, t]

    return d_E
