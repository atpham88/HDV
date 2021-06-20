
import pandas as pd
import numpy as np

def solar_data(model_dir, solar_folder, S_t, T, station_no, hour, ir, life_P, capex_P, fixed_OM_P, load_pr_case):
    if load_pr_case == 1:
        solar_data = model_dir + solar_folder + "solar_CF_case1.xlsx"
    elif load_pr_case == 2:
        solar_data = model_dir + solar_folder + "solar_CF_case2.xlsx"
    elif load_pr_case == 3:
        solar_data = model_dir + solar_folder + "solar_CF_case3.xlsx"

    f_P_annual = pd.read_excel(solar_data, header=None)  # solar capacity factor (S x T)
    f_P_annual_final = pd.DataFrame(np.tile(f_P_annual, 1))
    f_P_temp = {(r, c): f_P_annual_final.at[r, c] for r in S_t for c in T}

    f_P = dict.fromkeys((range(hour)))
    for t in T:
        f_P[t] = f_P_temp[station_no - 1, t]

    CRF_P = ir / (1 - (1 + ir) ** (-life_P))
    p_PK = capex_P * CRF_P + fixed_OM_P  # solar annualized capacity cost
    return f_P, p_PK
