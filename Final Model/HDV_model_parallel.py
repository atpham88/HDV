
import pandas as pd
import numpy as np
from pyomo.environ import *
from pyomo.opt import SolverFactory
import xlsxwriter as xw
import multiprocessing as mp
import time

start_time = time.time()

# %% Define Switches:
super_comp = 0          # ==1: run on super computer, ==0: run on local laptop
ramping_const = 1       # ==1: ramping constraints for SMR activated, ==0: no ramping constraints

# %% Specify model scope:
first_station, last_station = 3, 5    # Range of stations to run. Pick numbers between 1 and 170, first_station <= last_station
cap_class = 7           # Number of transmission capacity classes
station = 170           # Number of charging stations
hour = 8760             # Number of hours in a typical year
day = 365               # Number of days in a typical year

# %% Model main parameters:

# Interest rate:
ir = 0.015

# Battery:
capex_B = 1455000                # Assuming 4hr moderate battery storage ($/MW)
life_B = 15                      # Battery life time (years)
fixed_OM_B = 36370               # Fixed battery OM cost ($/MW-year)
p_BC_fixed = 0                   # Battery energy cost ($/kWh/h)
p_BE = 0                         # Battery operating cost ($/MW)
h_B = 4                          # Battery hours (hour)

# H2:
p_HK_fixed = 0.0148              # H2 capital cost ($/kW)
p_HC_fixed = 1.47 * 10 ** (-6)   # Dowling et al ($/kWh/h)
p_HE = 0                         # H2 operating cost

# SMR:
r_M = 0.4                        # Ramp rate for SMR (percentage of capacity)
k_M = 60                         # SMR capacity per module (MW)
life_M = 40                      # SMR life time (years)
capex_M = 2616000                # SMR capex ($/MW)
fixed_OM_M = 25000               # Fixed SMR OM cost ($/MW-year)
var_OM_M = 0.75                  # Variable OM cost ($/MWh)
fuel_M = 8.71                    # SMR fuel cost ($/MWh)
g_M_min = 30                     # SMR minimum stable load (MWh)

# Solar:
capex_P = 1354000                # Solar capex ($/MW)
life_P = 30                      # Solar PV life time (years)
fixed_OM_P = 19000               # Fixed solar OM cost ($/MW-year)
p_PE = 0                         # Solar operating cost

# Transmission:
p_WK = 100000                    # Transmission capacity cost ($/MW-year)
p_WO = 0.1                       # Transmission over-head cost (percentage)
p_WI = 50000                     # Transmission infrastructure cost
p_WL = 200000                    # Land cost

# %% Set working directory:
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

# %% Define set:
no_station_to_run = 1

I = list(range(cap_class))                              # Set of transmission capacity classes
S = list(range(no_station_to_run))                      # Set of charging stations to run
T = list(range(hour))                                   # Set of hours to run
S_t = list(range(station))                              # Set of total charging stations
S_r = list(range(first_station-1, last_station))        # Set of charging stations to run

# %% Import and calculate data input:
# Load station number:

def hdv_model(station_no):
    # %% Set model type - Concrete Model:
    model = ConcreteModel(name="HDV_model")

    # %% Define variables:

    # Power rating by technology:
    model.k_B = Var(S, within=NonNegativeReals)  # Battery power rating
    model.k_H = Var(S, within=NonNegativeReals)  # H2 power rating
    model.k_P = Var(S, within=NonNegativeReals)  # Solar capacity

    # Energy capacity by technology:
    model.e_B = Var(S, within=NonNegativeReals)  # Battery energy capacity
    model.e_H = Var(S, within=NonNegativeReals)  # H2 energy capacity

    # Generation by technology:
    model.g_B = Var(S, T, within=NonNegativeReals)  # Battery generation
    model.g_H = Var(S, T, within=NonNegativeReals)  # H2 generation
    model.g_P = Var(S, T, within=NonNegativeReals)  # Solar generation
    model.g_M = Var(S, T, within=NonNegativeReals)  # SMR generation
    model.g_W = Var(S, T, within=NonNegativeReals)  # Wholesale purchased generation

    # Inflow demands by technology:
    model.d_B = Var(S, T, within=NonNegativeReals)  # Battery inflow demand
    model.d_H = Var(S, T, within=NonNegativeReals)  # H2 inflow demand

    # States of charges:
    model.x_B = Var(S, T, within=NonNegativeReals)  # Battery state of charge
    model.x_H = Var(S, T, within=NonNegativeReals)  # H2 state of charge

    # Whole number variables:
    model.u_M = Var(S, within=NonNegativeIntegers)  # Number of SMR modules built
    model.u_W = Var(S, I, within=Binary)  # Whether a transmission line is built or not

    # Load data:
    load_data_raw = pd.read_csv(model_dir + load_folder + "TM2012_SPNation_TF-1_PT1_DPs1_BS1000_CP500_TD1_BD1_VW1_station.csv", header=None)
    load_data_raw_2 = load_data_raw.iloc[:, 4:] / 1000
    load_data_annual = pd.DataFrame(np.tile(load_data_raw_2, day))
    d_E_temp = {(r, c): load_data_annual.at[r, c] for r in S_t for c in T}

    d_E = dict.fromkeys((range(hour)))
    for t in T:
        d_E[t] = d_E_temp[station_no - 1, t]

    # Battery:
    CRF_B = ir / (1 - (1 + ir) ** (-life_B))    # Calculate battery capital recovery factor
    p_BK = capex_B * CRF_B + fixed_OM_B         # Battery power rating cost
    p_BC = p_BC_fixed * 1000 * 8760             # Battery energy cost ($/MWh)

    # H2:
    p_HK = p_HK_fixed * 1000 * 8760  # H2 power capacity cost
    p_HC = p_HC_fixed * 1000 * 8760  # H2 energy capacity cost

    # SMR:
    CRF_M = ir / (1 - (1 + ir) ** (-life_M))
    p_MK = capex_M * CRF_M + fixed_OM_M         # SMR capital cost
    p_ME = var_OM_M + fuel_M                    # SMR operating cost ($/MWh)

    # Solar data:
    solar_data = model_dir + solar_folder + "solar.xlsx"
    f_P_annual = pd.read_excel(solar_data, "capacity factor")  # solar capacity factor (S x T)
    f_P_annual_final = pd.DataFrame(np.tile(f_P_annual, 1))
    f_P_temp = {(r, c): f_P_annual_final.at[r, c] for r in S_t for c in T}

    f_P = dict.fromkeys((range(hour)))
    for t in T:
        f_P[t] = f_P_temp[station_no - 1, t]

    CRF_P = ir / (1 - (1 + ir) ** (-life_P))
    p_PK = capex_P * CRF_P + fixed_OM_P     # solar annualized capacity cost

    # Transmission data:
    transmission_data = model_dir + trans_folder + "transmission.xlsx"
    k_W_temp = pd.DataFrame(pd.read_excel(transmission_data, "transmission capacity")).to_dict()
    k_W = k_W_temp['trans cap']  # effective transmission capacity

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
    p_WE_temp = {(r, c): p_WE_annual.at[r, c] for r in S_t for c in T}  # wholesale electricity cost

    p_WE = dict.fromkeys((range(hour)))
    for t in T:
        p_WE[t] = p_WE_temp[station_no - 1, t]


    # %% Formulate constraints and  objective functions:

    # Constraints:
    # Battery constraints:
    def ub_d_battery(model, s, t):
        return model.d_B[s, t] <= model.k_B[s]
    model.ub_d_B = Constraint(S, T, rule=ub_d_battery)

    def ub_g_battery(model, s, t):
        return model.g_B[s, t] <= model.k_B[s]
    model.ub_g_B = Constraint(S, T, rule=ub_g_battery)

    def ub_g_x_battery(model, s, t):
        return model.g_B[s, t] <= model.x_B[s, t]
    model.ub_g_x_B = Constraint(S, T, rule=ub_g_x_battery)

    def ub_x_e_battery(model, s, t):
        return model.x_B[s, t] <= model.e_B[s]
    model.ub_x_e_B = Constraint(S, T, rule=ub_x_e_battery)

    def ub_e_battery(model, s):
        return model.e_B[s] == model.k_B[s]*h_B
    model.ub_e_B = Constraint(S, rule=ub_e_battery)

    def x_battery(model, s, t):
        if t == 0:
            return model.x_B[s, t] == 0.5 * model.e_B[s]  # here needs to specify initial SOC conditions
        return model.x_B[s, t] == model.x_B[s, t - 1] + model.d_B[s, t] - model.g_B[s, t]
    model.x_b = Constraint(S, T, rule=x_battery)

    # H2 constraints:
    def ub_d_hydrogen(model, s, t):
        return model.d_H[s, t] <= model.k_H[s]
    model.ub_d_H = Constraint(S, T, rule=ub_d_hydrogen)

    def ub_g_hydrogen(model, s, t):
        return model.g_H[s, t] <= model.k_H[s]
    model.ub_g_H = Constraint(S, T, rule=ub_g_hydrogen)

    def ub_g_x_hydrogen(model, s, t):
        return model.g_H[s, t] <= model.x_H[s, t]
    model.ub_g_x_H = Constraint(S, T, rule=ub_g_x_hydrogen)

    def ub_x_e_hydrogen(model, s, t):
        return model.x_H[s, t] <= model.e_H[s]
    model.ub_x_e_H = Constraint(S, T, rule=ub_x_e_hydrogen)

    def x_hydrogen(model, s, t):
        if t == 0:
            return model.x_H[s, t] == 0.5 * model.e_H[s]
        return model.x_H[s, t] == model.x_H[s, t - 1] + model.d_H[s, t] - model.g_H[s, t]
    model.x_h = Constraint(S, T, rule=x_hydrogen)

    # Solar PV constraint:
    def ub_g_solar(model, s, t):
        return model.g_P[s, t] <= f_P[t] * model.k_P[s]
    model.ub_g_P = Constraint(S, T, rule=ub_g_solar)

    # SMR constraints:
    def ub_g_smr(model, s, t):
        return model.g_M[s, t] <= model.u_M[s] * k_M
    model.ub_g_M = Constraint(S, T, rule=ub_g_smr)

    def g_min_smr(model, s, t):
        return model.g_M[s, t] >= model.u_M[s] * g_M_min
    model.ub_g_M_min = Constraint(S, T, rule=g_min_smr)

    if ramping_const == 1:
        def r_smr_up(model, s, t):
            if t == 0:
                return Constraint.Skip
            else:
                return model.g_M[s, t] - model.g_M[s, t - 1] <= r_M * k_M * model.u_M[s]
        model.r_m_up = Constraint(S, T, rule=r_smr_up)

        def r_smr_down(model, s, t):
            if t == 0:
                return Constraint.Skip
            else:
                return model.g_M[s, t - 1] - model.g_M[s, t] <= r_M * k_M * model.u_M[s]
        model.r_m_down = Constraint(S, T, rule=r_smr_down)

    # Wholesale power constraints:
    def ub_g_wholesale(model, s, t):
        return model.g_W[s, t] <= sum(model.u_W[s, i] * k_W[i] for i in I)
    model.ub_g_W = Constraint(S, T, rule=ub_g_wholesale)

    def trans_line_limit(model, s):
        return sum(model.u_W[s, i] for i in I) <= 1
    model.trans_const = Constraint(S, rule=trans_line_limit)

    # Market clearing condition:
    def market_clearing(model, s, t):
        return model.g_B[s, t] + model.g_H[s, t] + model.g_P[s, t] + model.g_M[s, t] \
               + model.g_W[s, t] >= d_E[t] + model.d_B[s, t] + model.d_H[s, t]
    model.mc_const = Constraint(S, T, rule=market_clearing)

    # Objective function:
    def obj_function(model):
        return sum(p_BK * model.k_B[s] + p_BC * model.e_B[s] + sum(p_BE * model.g_B[s, t] for t in T) for s in S) \
               + sum(p_HK * model.k_H[s] + p_HC * model.e_H[s] + sum(p_HE * model.g_H[s, t] for t in T) for s in S) \
               + sum(p_PK * model.k_P[s] + sum(p_PE * model.g_P[s, t] for t in T) for s in S) \
               + sum(p_MK * model.u_M[s] * k_M + sum(p_ME * model.g_M[s, t] for t in T) for s in S) \
               + sum(sum(p_WK * k_W[i] + (1 + p_WO) * (p_WI + p_WC[i] + p_WL) * l_W[i] * model.u_W[s, i] for i in I) + sum(p_WE[t] * model.g_W[s, t] for t in T) for s in S)
    model.obj_func = Objective(rule=obj_function)

    # %% Solve HDV model:
    # model.dual = Suffix(direction=Suffix.IMPORT_EXPORT)
    solver = SolverFactory('cplex')
    results = solver.solve(model, tee=True)
    # model.pprint()

    if (results.solver.status == SolverStatus.ok) and (results.solver.termination_condition == TerminationCondition.optimal):
        print('Solution is feasible')
    elif results.solver.termination_condition == TerminationCondition.infeasible:
        print('Solution is infeasible')
    else:
        # Something else is wrong
        print('Solver Status: ', results.solver.status)

    # %% Print variable outputs:
    total_cost = value(model.obj_func)

    k_B_star = np.zeros(station)
    k_H_star = np.zeros(station)
    k_P_star = np.zeros(station)
    e_B_star = np.zeros(station)
    e_H_star = np.zeros(station)
    u_M_star = np.zeros(station)

    g_B_star = np.zeros((station, hour))
    g_H_star = np.zeros((station, hour))
    g_P_star = np.zeros((station, hour))
    g_M_star = np.zeros((station, hour))
    g_W_star = np.zeros((station, hour))
    d_B_star = np.zeros((station, hour))
    d_H_star = np.zeros((station, hour))
    x_B_star = np.zeros((station, hour))
    x_H_star = np.zeros((station, hour))

    u_W_star = np.zeros((station, cap_class))

    for s in S:
        k_B_star[s] = value(model.k_B[s])
        k_H_star[s] = value(model.k_H[s])
        k_P_star[s] = value(model.k_P[s])
        e_B_star[s] = value(model.e_B[s])
        e_H_star[s] = value(model.e_H[s])
        u_M_star[s] = value(model.u_M[s])

    for s in S:
        for t in T:
            g_B_star[s, t] = value(model.g_B[s, t])
            g_H_star[s, t] = value(model.g_H[s, t])
            g_P_star[s, t] = value(model.g_P[s, t])
            g_M_star[s, t] = value(model.g_M[s, t])
            g_W_star[s, t] = value(model.g_W[s, t])
            d_B_star[s, t] = value(model.d_B[s, t])
            d_H_star[s, t] = value(model.d_H[s, t])
            x_B_star[s, t] = value(model.x_B[s, t])
            x_H_star[s, t] = value(model.x_H[s, t])

    for s in S:
        for i in I:
            u_W_star[s, i] = value(model.u_W[s, i])

    # write results:
    results_book = xw.Workbook(results_folder + 'HDV_model_results for station ' + str(station_no) + '.xlsx')
    result_sheet_pr = results_book.add_worksheet('power rating')
    result_sheet_tc = results_book.add_worksheet('transmission class')
    result_sheet_bg = results_book.add_worksheet('battery gen')
    result_sheet_hg = results_book.add_worksheet('h2 gen')
    result_sheet_mg = results_book.add_worksheet('smr gen')
    result_sheet_sg = results_book.add_worksheet('solar gen')
    result_sheet_wg = results_book.add_worksheet('purchased gen')
    result_sheet_ob = results_book.add_worksheet('total cost')
    result_sheet_bi = results_book.add_worksheet('battery inflow')
    result_sheet_hi = results_book.add_worksheet('h2 inflow')
    result_sheet_bs = results_book.add_worksheet('battery SOC')
    result_sheet_hs = results_book.add_worksheet('h2 SOC')

    station_number = [''] * station
    hour_number = [''] * hour
    class_number = [''] * cap_class

    for s in S:
        station_number[s] = "station " + str(station_no)

    for t in T:
        hour_number[t] = "hour " + str(T[t] + 1)

    for i in I:
        class_number[i] = "class " + str(I[i] + 1)

    # Write total cost result:
    result_sheet_ob.write("A1", "total cost (million $)")
    result_sheet_ob.write("B1", total_cost / 1000000)

    # Write power rating results:
    result_sheet_pr.write("B1", "battery power rating (MW)")
    result_sheet_pr.write("C1", "battery energy capacity (MWh)")
    result_sheet_pr.write("D1", "h2 power rating (MW)")
    result_sheet_pr.write("E1", "h2 energy capacity (MWh)")
    result_sheet_pr.write("F1", "solar capacity (MW)")
    result_sheet_pr.write("G1", "number of SMR modules")

    for item in S:
        result_sheet_pr.write(item + 1, 0, station_number[item])
        result_sheet_pr.write(item + 1, 1, k_B_star[item])
        result_sheet_pr.write(item + 1, 2, e_B_star[item])
        result_sheet_pr.write(item + 1, 3, k_H_star[item])
        result_sheet_pr.write(item + 1, 4, e_H_star[item])
        result_sheet_pr.write(item + 1, 5, k_P_star[item])
        result_sheet_pr.write(item + 1, 6, u_M_star[item])

    # write transmission class results:
    for item in S:
        result_sheet_tc.write(item + 1, 0, station_number[item])

    for item in I:
        result_sheet_tc.write(0, item + 1, class_number[item])

    for item_1 in S:
        for item_2 in I:
            result_sheet_tc.write(item_1 + 1, item_2 + 1, u_W_star[item_1, item_2])

    # write battery gen results:
    for item in S:
        result_sheet_bg.write(item + 1, 0, station_number[item])

    for item in T:
        result_sheet_bg.write(0, item + 1, hour_number[item])

    for item_1 in S:
        for item_2 in T:
            result_sheet_bg.write(item_1 + 1, item_2 + 1, g_B_star[item_1, item_2])

    # write h2 gen results:
    for item in S:
        result_sheet_hg.write(item + 1, 0, station_number[item])

    for item in T:
        result_sheet_hg.write(0, item + 1, hour_number[item])

    for item_1 in S:
        for item_2 in T:
            result_sheet_hg.write(item_1 + 1, item_2 + 1, g_H_star[item_1, item_2])

    # write smr gen results:
    for item in S:
        result_sheet_mg.write(item + 1, 0, station_number[item])

    for item in T:
        result_sheet_mg.write(0, item + 1, hour_number[item])

    for item_1 in S:
        for item_2 in T:
            result_sheet_mg.write(item_1 + 1, item_2 + 1, g_M_star[item_1, item_2])

    # write solar gen results:
    for item in S:
        result_sheet_sg.write(item + 1, 0, station_number[item])

    for item in T:
        result_sheet_sg.write(0, item + 1, hour_number[item])

    for item_1 in S:
        for item_2 in T:
            result_sheet_sg.write(item_1 + 1, item_2 + 1, g_P_star[item_1, item_2])

    # write purchased gen results:
    for item in S:
        result_sheet_wg.write(item + 1, 0, station_number[item])

    for item in T:
        result_sheet_wg.write(0, item + 1, hour_number[item])

    for item_1 in S:
        for item_2 in T:
            result_sheet_wg.write(item_1 + 1, item_2 + 1, g_W_star[item_1, item_2])

    # battery inflow results:
    for item in S:
        result_sheet_bi.write(item + 1, 0, station_number[item])

    for item in T:
        result_sheet_bi.write(0, item + 1, hour_number[item])

    for item_1 in S:
        for item_2 in T:
            result_sheet_bi.write(item_1 + 1, item_2 + 1, d_B_star[item_1, item_2])

    # h2 inflow results:
    for item in S:
        result_sheet_hi.write(item + 1, 0, station_number[item])

    for item in T:
        result_sheet_hi.write(0, item + 1, hour_number[item])

    for item_1 in S:
        for item_2 in T:
            result_sheet_hi.write(item_1 + 1, item_2 + 1, d_H_star[item_1, item_2])

    # battery SOC results:
    for item in S:
        result_sheet_bs.write(item + 1, 0, station_number[item])

    for item in T:
        result_sheet_bs.write(0, item + 1, hour_number[item])

    for item_1 in S:
        for item_2 in T:
            result_sheet_bs.write(item_1 + 1, item_2 + 1, x_B_star[item_1, item_2])

    # H2 SOC results:
    for item in S:
        result_sheet_hs.write(item + 1, 0, station_number[item])

    for item in T:
        result_sheet_hs.write(0, item + 1, hour_number[item])

    for item_1 in S:
        for item_2 in T:
            result_sheet_hs.write(item_1 + 1, item_2 + 1, x_H_star[item_1, item_2])

    results_book.close()
    del model

if __name__ == "__main__":
    station_no = [cs + 1 for cs in S_t]

    pool = mp.Pool(8)
    pool.map(hdv_model, S_t)
    pool.close()
    pool.join()

print("--- %s seconds ---" % (time.time() - start_time))

# py-spy record -o profile.svg -- python HDV_parallel.py