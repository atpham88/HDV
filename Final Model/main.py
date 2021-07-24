"""
An Pham
Updated 07/23/2021
"""

import numpy as np
from pyomo.environ import *
from pyomo.opt import SolverFactory
import time

from get_storage_data import storage_data
from get_SMR_data import SMR_data
from get_solar_data import solar_data
from get_load_data import load_data
from get_transmission_data import transmission_data
import xlsxwriter as xw

start_time = time.time()

# %% Main parameters:
def main_params():
    # Define Switches:
    super_comp = 0                          # ==1: run on super computer, ==0: run on local laptop
    ramping_const = 1                       # ==1: ramping constraints for SMR activated, ==0: no ramping constraints
    load_pr_case = 1                        # ==1: 170 stations, ==2: 161 stations, ==3: 152 stations
    inc_h2_demand = 1                       # ==1: include hydrogen demand, ==0: no hydrogen demand

    # Specify model scope:
    first_station, last_station = 148, 148      # Range of stations to run. Pick numbers between 1 and 170/161/152, first_station <= last_station
    hour = 8760                             # Number of hours in a typical year
    day = 365                               # Number of days in a typical year

    # Number of charging stations:
    if load_pr_case == 1:
        station = 170
    elif load_pr_case == 2:
        station = 161
    elif load_pr_case == 3:
        station = 152

    no_station_to_run = 1                   # Don't change this. We only run one station at a time

    # Model main parameters:
    ir = 0.015                              # Interest rate

    # Battery:
    capex_B = 1455000                       # Assuming 4hr moderate battery storage ($/MW)
    life_B = 15                             # Battery life time (years)
    fixed_OM_B = 36370                      # Fixed battery OM cost ($/MW-year)
    p_BC_fixed = 0                          # Battery energy cost ($/kWh/h)
    p_BE = 0                                # Battery operating cost ($/MW)
    h_B = 4                                 # Battery hours (hour)

    # Hydrogen:
    h2_demand_p = 0.2                       # Percentage of electricity demand that comes from H2
    h2_convert = 0.0493                     # Tianyi's number
    hhv = 39.4                              # Tianyi's slides (also same as Dowling et al.)
    p_HK_fixed = 0.0148                     # H2 capital cost ($/kW) from Dowling et al ($/kWh/h)
    p_HC_fixed = 1.47*10**(-6)              # H2 energy cost ($/kW) from Dowling et al ($/kWh/h)
    p_HE = 0                                # H2 operating cost

    # SMR:
    r_M = 0.4                               # Ramp rate for SMR (percentage of capacity)
    k_M = 60                                # SMR capacity per module (MW)
    life_M = 40                             # SMR life time (years)
    capex_M = 2616000                       # SMR capex ($/MW)
    fixed_OM_M = 25000                      # Fixed SMR OM cost ($/MW-year)
    var_OM_M = 0.75                         # Variable OM cost ($/MWh)
    fuel_M = 8.71                           # SMR fuel cost ($/MWh)
    g_M_min = 30                            # SMR minimum stable load (MWh)

    # Solar:
    capex_P = 1354000                       # Solar capex ($/MW)
    life_P = 30                             # Solar PV life time (years)
    fixed_OM_P = 19000                      # Fixed solar OM cost ($/MW-year)
    p_PE = 0                                # Solar operating cost

    # Transmission:
    cf_W = 1                                # Transmission capacity factor
    k_Double = 0                            # Include class double (==1) or not (==0)
    p_WO = 0.1                              # Transmission over-head cost (percentage)

    # Specify capacity class:
    if k_Double == 0:
        cap_class = 5                       # Number of transmission capacity classes
    elif k_Double == 1:
        cap_class = 10

    return (super_comp, ramping_const, load_pr_case, inc_h2_demand, first_station, last_station, cap_class, hour, day, station,
            no_station_to_run, ir, capex_B, life_B, fixed_OM_B, p_BC_fixed, p_BE, h_B, p_HK_fixed, p_HC_fixed, p_HE, r_M, k_M,
            life_M, capex_M, fixed_OM_M, var_OM_M, fuel_M, g_M_min, capex_P, life_P, fixed_OM_P, p_PE, p_WO, cf_W, h2_demand_p,
            hhv, h2_convert, k_Double)


def main_function():
    (super_comp, ramping_const, load_pr_case, inc_h2_demand, first_station, last_station, cap_class, hour, day, station,
     no_station_to_run, ir, capex_B, life_B, fixed_OM_B, p_BC_fixed, p_BE, h_B, p_HK_fixed, p_HC_fixed, p_HE, r_M, k_M,
     life_M, capex_M, fixed_OM_M, var_OM_M, fuel_M, g_M_min, capex_P, life_P, fixed_OM_P, p_PE, p_WO, cf_W, h2_demand_p,
     hhv, h2_convert, k_Double) = main_params()

    (model_dir, load_folder, solar_folder, trans_folder, results_folder, charging_station_folder) = working_directory(super_comp)

    (I, S, T, S_t, S_r) = main_sets(no_station_to_run, cap_class, hour, station, first_station, last_station)

    model_solve(S, S_r, S_t, T, I, model_dir, results_folder, load_folder, solar_folder, trans_folder,
                charging_station_folder, station, cap_class, load_pr_case, inc_h2_demand, hour, day,
                ir, capex_B, fixed_OM_B, p_BC_fixed, life_B, p_HK_fixed, p_HC_fixed, capex_M, life_M,
                fixed_OM_M, var_OM_M, fuel_M, life_P, capex_P, fixed_OM_P, ramping_const, k_M, g_M_min,
                h_B, r_M, p_WO, p_BE, p_HE, p_PE, cf_W, h2_demand_p, hhv, h2_convert, k_Double)


# %% Set working directory:
def working_directory(super_comp):
    if super_comp == 1:
        model_dir = '/home/anph/projects/HDV/Data/'
        load_folder = 'load_station/'
        solar_folder = 'solar/'
        trans_folder = 'transmission/'
        results_folder = '/home/anph/projects/HDV/Results/'
        charging_station_folder = 'charging station profiles/'
    else:
        model_dir = 'C:\\Users\\atpha\\Documents\\Postdocs\\Projects\\HDV\\Data\\'
        load_folder = 'load_station\\'
        solar_folder = 'solar\\'
        trans_folder = 'transmission\\'
        results_folder = 'C:\\Users\\atpha\\Documents\\Postdocs\\Projects\\HDV\\Results\\'
        charging_station_folder = 'charging station profiles\\'
    return model_dir, load_folder, solar_folder, trans_folder, results_folder, charging_station_folder


# %% Define set:
def main_sets(no_station_to_run, cap_class, hour, station, first_station, last_station):
    I = list(range(cap_class))                              # Set of transmission capacity classes
    S = list(range(no_station_to_run))                      # Set of charging stations to run
    T = list(range(hour))                                   # Set of hours to run
    S_t = list(range(station))                              # Set of total charging stations
    S_r = list(range(first_station-1, last_station))        # Set of charging stations to run
    return I, S, T, S_t, S_r


# %% Solving HDV model:
def model_solve(S, S_r, S_t, T, I, model_dir, results_folder, load_folder, solar_folder, trans_folder,
                charging_station_folder, station, cap_class, load_pr_case, inc_h2_demand, hour, day,
                ir, capex_B, fixed_OM_B, p_BC_fixed, life_B, p_HK_fixed, p_HC_fixed, capex_M, life_M,
                fixed_OM_M, var_OM_M, fuel_M, life_P, capex_P, fixed_OM_P, ramping_const, k_M, g_M_min,
                h_B, r_M, p_WO, p_BE, p_HE, p_PE, cf_W, h2_demand_p, hhv, h2_convert, k_Double):

    for cs in S_r:
        # %% Set model type - Concrete Model:
        model = ConcreteModel(name="HDV_model")
        station_no = cs + 1

        # Load data:
        (d_E, d_H_bar) = load_data(model_dir, load_folder, S_t, T, hour, day, station_no, load_pr_case, h2_demand_p, h2_convert, inc_h2_demand)

        # Storage:
        (p_BK, p_BC, p_HK, p_HC) = storage_data(ir, life_B, capex_B, fixed_OM_B, p_BC_fixed, p_HK_fixed, p_HC_fixed, hhv)

        # SMR:
        (p_MK, p_ME) = SMR_data(ir, life_M, capex_M, fixed_OM_M, var_OM_M, fuel_M)

        # Solar data:
        (f_P, p_PK) = solar_data(model_dir, solar_folder, S_t, T, station_no, hour, ir, life_P, capex_P, fixed_OM_P, load_pr_case)

        # Transmission data:
        (p_WK, k_W, p_WE) = transmission_data(model_dir, trans_folder, charging_station_folder, S_t, T, I, cap_class, station_no, day, hour, cf_W, load_pr_case, k_Double)

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
        if inc_h2_demand == 1:
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

            def d_hydrogen(model, s, t):
                return model.g_H[s, t] >= d_H_bar[t] + model.d_H[s, t]
            model.d_H_bar = Constraint(S, T, rule=d_hydrogen)

        # Solar PV constraint:
        def ub_g_solar(model, s, t):
            return model.g_P[s, t] <= f_P[t] * model.k_P[s]
        model.ub_g_P = Constraint(S, T, rule=ub_g_solar)

        # SMR constraints:
        def ub_g_smr(model, s, t):
            return model.g_M[s, t] <= model.u_M[s] * k_M
        model.ub_g_M = Constraint(S, T, rule=ub_g_smr)

        def g_min_smr(model, s, t):
            return model.g_M[s, t] >= model.u_M[s]*g_M_min
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
            return model.g_B[s, t] + model.g_P[s, t] + model.g_M[s, t] \
                       + model.g_W[s, t] >= d_E[t] + model.d_B[s, t]
        model.mc_const = Constraint(S, T, rule=market_clearing)

        # Objective function:
        def obj_function(model):
            return sum(p_BK * model.k_B[s] + p_BC * model.e_B[s] + sum(p_BE * model.g_B[s, t] for t in T) for s in S) \
                   + sum(p_HK * model.k_H[s] + p_HC * model.e_H[s] + sum(p_HE * model.g_H[s, t] for t in T) for s in S) \
                   + sum(p_PK * model.k_P[s] + sum(p_PE * model.g_P[s, t] for t in T) for s in S) \
                   + sum(p_MK * model.u_M[s] * k_M + sum(p_ME * model.g_M[s, t] for t in T) for s in S) \
                   + sum(sum((1 + p_WO) * p_WK[i] * k_W[i] * model.u_W[s, i] for i in I) + sum(p_WE[t] * model.g_W[s, t] for t in T) for s in S)
        model.obj_func = Objective(rule=obj_function)

        # Solve HDV model:
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

        # Print variable outputs:
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
                if inc_h2_demand == 1:
                    g_H_star[s, t] = value(model.g_H[s, t])
                    d_H_star[s, t] = value(model.d_H[s, t])
                    x_H_star[s, t] = value(model.x_H[s, t])
                g_P_star[s, t] = value(model.g_P[s, t])
                g_M_star[s, t] = value(model.g_M[s, t])
                g_W_star[s, t] = value(model.g_W[s, t])
                d_B_star[s, t] = value(model.d_B[s, t])
                x_B_star[s, t] = value(model.x_B[s, t])

        for s in S:
            for i in I:
                u_W_star[s, i] = value(model.u_W[s, i])

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

main_function()

print("--- %s seconds ---" % (time.time() - start_time))