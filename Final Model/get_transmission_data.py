
import pandas as pd
import numpy as np

def transmission_data(model_dir, trans_folder, charging_station_folder, S_t, T, I, cap_class,
                      station_no, day, hour, cf_W, load_pr_case, k_Double, trans_cap_util):
    station_case_data = model_dir + charging_station_folder + "charging_station_cases.xlsx"
    transmission_data = model_dir + trans_folder + "Transmission Cost Inputs.xlsx"

    # Read in transmission line capacity:
    k_W_temp = pd.read_excel(transmission_data, "Eff_Cap")
    k_W_temp.sort_index(inplace=True)

    # read in transmission line annualized capital cost:                                                            # effective transmission capacity
    p_WK_temp = pd.DataFrame(pd.read_excel(transmission_data, "Annualized_Capital_Cost"))    # conductor cost
    p_WK_temp.sort_index(inplace=True)

    # read in case scenario:
    scr_temp = pd.read_excel(station_case_data)
    scr_temp.sort_index(inplace=True)
    scr = scr_temp[['case 1', 'case 2', 'case 3']]

    if k_Double == 0:
        k_W_temp = k_W_temp[['<100', '100-161', '220-287', 345, '>500']]
        p_WK_temp = p_WK_temp[['<100', '100-161', '220-287', 345, '>500']]
    elif k_Double == 1:
        k_W_temp = k_W_temp[['<100', '<100 Double', '100-161', '161 Double', '220-287', '230 Double', 345, '345 Double', '>500', '500 Double']]
        p_WK_temp = p_WK_temp[['<100', '<100 Double', '100-161', '161 Double', '220-287', '230 Double', 345, '345 Double', '>500', '500 Double']]

    k_W_temp = pd.concat([k_W_temp, scr], axis=1)
    p_WK_temp = pd.concat([p_WK_temp, scr], axis=1)

    if load_pr_case == 1:
        k_W_temp = k_W_temp[(k_W_temp['case 1']!= 0)]
        p_WK_temp = p_WK_temp[(p_WK_temp['case 1'] != 0)]
    elif load_pr_case == 2:
        k_W_temp = k_W_temp[(k_W_temp['case 2'] != 0)]
        p_WK_temp = p_WK_temp[(p_WK_temp['case 2'] != 0)]
    elif load_pr_case == 3:
        k_W_temp = k_W_temp[(k_W_temp['case 3'] != 0)]
        p_WK_temp = p_WK_temp[(p_WK_temp['case 3'] != 0)]

    k_W_temp = k_W_temp.reset_index()
    p_WK_temp = p_WK_temp.reset_index()

    if k_Double == 0:
        k_W_temp = k_W_temp[['<100', '100-161', '220-287', 345, '>500']]
        p_WK_temp = p_WK_temp[['<100', '100-161', '220-287', 345, '>500']]
        k_W_temp = k_W_temp.rename(columns={'<100': 0, '100-161': 1, '220-287': 2, 345: 3, '>500': 4})
        p_WK_temp = p_WK_temp.rename(columns={'<100': 0, '100-161': 1, '220-287': 2, 345: 3, '>500': 4})
    elif k_Double == 1:
        k_W_temp = k_W_temp[['<100', '<100 Double', '100-161', '161 Double', '220-287', '230 Double', 345, '345 Double', '>500', '500 Double']]
        p_WK_temp = p_WK_temp[['<100', '<100 Double', '100-161', '161 Double', '220-287', '230 Double', 345, '345 Double', '>500', '500 Double']]
        k_W_temp = k_W_temp.rename(columns={'<100': 0, '<100 Double':1, '100-161': 2, '161 Double': 3, '220-287': 4, '230 Double': 5, 345: 6, '345 Double': 7, '>500': 8, '500 Double': 9})
        p_WK_temp = p_WK_temp.rename(columns={'<100': 0, '<100 Double':1, '100-161': 2, '161 Double': 3, '220-287': 4, '230 Double': 5, 345: 6, '345 Double': 7, '>500': 8, '500 Double': 9})

    k_W_temp = pd.DataFrame(k_W_temp).to_dict(orient='records')
    p_WK_temp = pd.DataFrame(p_WK_temp).to_dict(orient='records')

    k_W = dict.fromkeys((range(cap_class)))
    p_WK = dict.fromkeys((range(cap_class)))

    for i in I:
        k_W[i] = k_W_temp[station_no-1][i]*trans_cap_util
        p_WK[i] = p_WK_temp[station_no-1][i]

    if load_pr_case == 1:
        p_WE_annual_temp = pd.read_excel(model_dir + trans_folder + "lmp_2020_case1.xlsx")
    elif load_pr_case == 2:
        p_WE_annual_temp = pd.read_excel(model_dir + trans_folder + "lmp_2020_case2.xlsx")
    elif load_pr_case == 3:
        p_WE_annual_temp = pd.read_excel(model_dir + trans_folder + "lmp_2020_case3.xlsx")

    p_WE_annual_temp_2 = p_WE_annual_temp.iloc[:, 4:]
    p_WE_annual = pd.DataFrame(np.tile(p_WE_annual_temp_2, 1))
    p_WE_temp = {(r, c): p_WE_annual.at[r, c] for r in S_t for c in T}              # wholesale electricity cost

    p_WE = dict.fromkeys((range(hour)))
    for t in T:
        p_WE[t] = p_WE_temp[station_no - 1, t]

    return p_WK, k_W, p_WE
