# Formulate and solve HDV model in Pyomo using Cplex

# %% Import modules:
from pyomo.environ import *
from pyomo.opt import SolverFactory
import pandas as pd
import numpy as np
import time


start_time = time.time()


# %% Set model type - Concrete Model:
model = ConcreteModel(name="HDV_model")


# %% Define set:
I = list(range(7))      # Set of transmission capacity classes (7 classes)
S = list(range(170))    # Set of charging stations (170 stations)
T = list(range(8760))   # Set of hours in a year (8760 hours)


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

# %% Call in parameter inputs:

# Load profile data:
load_data_raw = pd.read_csv(r"C:\Users\atpha\Documents\Postdocs\Projects\HDV\Data\load_station\TM2012_SPNation_TF-1_PT1_DPs1_BS1000_CP500_TD1_BD1_VW1_station.csv", header=None)
load_data_raw_2 = load_data_raw.iloc[:, 4:]
load_data_annual = pd.DataFrame(np.tile(load_data_raw_2, 365))
d_E = {(r, c): load_data_annual.at[r, c] for r in S for c in T}

# Storage data:
# Battery:
battery_data = r"C:\Users\atpha\Documents\Postdocs\Projects\HDV\Data\storage\battery.xlsx"
r_B_temp = pd.DataFrame(pd.read_excel(battery_data, 'ramp rate')).to_dict()
r_B = r_B_temp['ramp rate']                                                         # ramp rate for battery (S)

p_BK_temp = pd.DataFrame(pd.read_excel(battery_data, "capital cost")).to_dict()     # battery capital cost (S)
p_BK = p_BK_temp['capital cost']

p_BC_temp = pd.DataFrame(pd.read_excel(battery_data, "energy cost")).to_dict()     # battery capital cost (S)
p_BC = p_BC_temp['energy cost']

p_BE_24 = pd.read_excel(battery_data, "operating cost")
p_BE_annual = pd.DataFrame(np.tile(p_BE_24, 365))
p_BE = {(r, c): p_BE_annual.at[r, c] for r in S for c in T}       # battery operating cost (S x T)

# H2:
hydrogen_data = r"C:\Users\atpha\Documents\Postdocs\Projects\HDV\Data\storage\hydrogen.xlsx"
r_H_temp = pd.read_excel(hydrogen_data, 'ramp rate')
r_H_temp = r_H_temp.to_dict()
r_H = r_H_temp['ramp rate']                                                         # ramp rate for H2 (S)

p_HK_temp = pd.DataFrame(pd.read_excel(hydrogen_data, "capital cost")).to_dict()     # H2 capital cost (S)
p_HK = p_HK_temp['capital cost']

p_HC_temp = pd.DataFrame(pd.read_excel(hydrogen_data, "energy cost")).to_dict()     # H2 capital cost (S)
p_HC = p_HC_temp['energy cost']

p_HE_24 = pd.read_excel(hydrogen_data, "operating cost")
p_HE_annual = pd.DataFrame(np.tile(p_HE_24, 365))
p_HE = {(r, c): p_HE_annual.at[r, c] for r in S for c in T}            # H2 operating cost (S x T)

# SMR data:
smr_data = r"C:\Users\atpha\Documents\Postdocs\Projects\HDV\Data\SMR\SMR.xlsx"
r_M_temp = pd.DataFrame(pd.read_excel(smr_data, "ramp rate")).to_dict()              # ramp rate for SMR (S)
r_M = r_M_temp['ramp rate']

k_M_temp = pd.DataFrame(pd.read_excel(smr_data, "module capacity")).to_dict()        # SMR capacity per module (S)
k_M = k_M_temp['capacity']

p_MK_temp = pd.DataFrame(pd.read_excel(smr_data, "capacity cost")).to_dict()         # SMR capital cost (S)
p_MK = p_MK_temp['capacity cost']

p_ME_24 = pd.read_excel(smr_data, "operating cost")
p_ME_annual = pd.DataFrame(np.tile(p_ME_24, 365))
p_ME = {(r, c): p_ME_annual.at[r, c] for r in S for c in T}                        # SMR operating cost (S x T)

# Solar data:
solar_data = r"C:\Users\atpha\Documents\Postdocs\Projects\HDV\Data\solar\solar.xlsx"
f_P_24 = pd.read_excel(solar_data, "capacity factor")                               # solar capacity factor (S x T)
f_P_annual = pd.DataFrame(np.tile(f_P_24, 365))
f_P = {(r, c): f_P_annual.at[r, c] for r in S for c in T}

p_PK_temp = pd.DataFrame(pd.read_excel(solar_data, "capacity cost")).to_dict()       # solar capacity cost (S)
p_PK = p_PK_temp['capacity cost']

p_PE_24 = pd.read_excel(solar_data, "operating cost")
p_PE_annual = pd.DataFrame(np.tile(p_PE_24, 365))
p_PE = {(r, c): p_PE_annual.at[r, c] for r in S for c in T}       # solar operating cost (S x T)

# Transmission data:
transmission_data = r"C:\Users\atpha\Documents\Postdocs\Projects\HDV\Data\transmission\transmission.xlsx"
k_W_temp = pd.DataFrame(pd.read_excel(transmission_data, "transmission capacity")).to_dict()  # effective transmission capacity (I)
k_W = k_W_temp['trans cap']

p_WK_temp = pd.read_excel(transmission_data, "capacity cost")   # transmission capacity cost (S x I)
p_WK = {(r, c): p_WK_temp.at[r, c] for r in S for c in I}

p_WO_temp = pd.DataFrame(pd.read_excel(transmission_data, "overhead")).to_dict()              # transmission over-head (S)
p_WO = p_WO_temp['overhead']

p_WI_temp = pd.DataFrame(pd.read_excel(transmission_data, "infrastructure cost"))     # transmission infrastructure cost (S x I)
p_WI = {(r, c): p_WI_temp.at[r, c] for r in S for c in I}

p_WC_temp = pd.DataFrame(pd.read_excel(transmission_data, "conductor cost"))           # conductor cost (S x I)
p_WC = {(r, c): p_WC_temp.at[r, c] for r in S for c in I}

p_WL_temp = pd.DataFrame(pd.read_excel(transmission_data, "land cost"))                     # land cost (S x I)
p_WL = {(r, c): p_WL_temp.at[r, c] for r in S for c in I}

l_W_temp = pd.DataFrame(pd.read_excel(transmission_data, "trans line length"))        # transmission line length (S x I)
l_W = {(r, c): l_W_temp.at[r, c] for r in S for c in I}

p_WE_24 = pd.read_excel(transmission_data, "wholesale price")
p_WE_annual = pd.DataFrame(np.tile(p_WE_24, 365))
p_WE = {(r, c): p_WE_annual.at[r, c] for r in S for c in T}                         # wholesale electricity cost (S x T)


# %% Formulate constraints and  objective functions:

# Constraints:
# Battery constraints:
def ub_d_battery(model, s, t):  # d_B <= k_B
    return model.d_B[s, t] <= model.k_B[s]
model.ub_d_B = Constraint(S, T, rule=ub_d_battery)

def ub_g_battery(model, s, t):  # g_B <= k_B
    return model.g_B[s, t] <= model.k_B[s]
model.ub_g_B = Constraint(S, T, rule=ub_g_battery)

def ub_g_x_battery(model, s, t):  # g_B <= x_B
    return model.g_B[s, t] <= model.x_B[s, t]
model.ub_g_x_B = Constraint(S, T, rule=ub_g_x_battery)

def ub_x_e_battery(model, s, t):  # x_B <= e_B
    return model.x_B[s, t] <= model.e_B[s]
model.ub_x_e_B = Constraint(S, T, rule=ub_x_e_battery)

def x_battery(model, s, t):    # state of charge constraints
    if t == 0:
        return model.x_B[s, t] == 0.5*model.e_B[s]      # here needs to specify initial SOC conditions
    return model.x_B[s, t] == model.x_B[s, t - 1] + model.d_B[s, t] - model.g_B[s, t]
model.x_b = Constraint(S, T, rule=x_battery)

def r_battery(model, s, t):    # ramping constraints
    if t == 0:
        return Constraint.Skip
    return abs(model.g_B[s, t] - model.g_B[s, t-1]) <= r_B[s]
model.r_b = Constraint(S, T, rule=r_battery)

# H2 constraints:
def ub_d_hydrogen(model, s, t):  # d_H <= k_H
    return model.d_H[s, t] <= model.k_H[s]
model.ub_d_H = Constraint(S, T, rule=ub_d_hydrogen)

def ub_g_hydrogen(model, s, t):  # g_H <= k_H
    return model.g_H[s, t] <= model.k_H[s]
model.ub_g_H = Constraint(S, T, rule=ub_g_hydrogen)

def ub_g_x_hydrogen(model, s, t):  # g_H <= x_H
    return model.g_H[s, t] <= model.x_H[s, t]
model.ub_g_x_H = Constraint(S, T, rule=ub_g_x_hydrogen)

def ub_x_e_hydrogen(model, s, t):  # x_H <= e_H
    return model.x_H[s, t] <= model.e_H[s]
model.ub_x_e_H = Constraint(S, T, rule=ub_x_e_hydrogen)

def x_hydrogen(model, s, t):    # state of charge constraints
    if t == 0:
        return model.x_H[s, t] == 0.5*model.e_H[s]      # here needs to specify initial SOC conditions
    return model.x_H[s, t] == model.x_H[s, t - 1] + model.d_H[s, t] - model.g_H[s, t]
model.x_h = Constraint(S, T, rule=x_hydrogen)

def r_hydrogen(model, s, t):    # ramping constraints
    if t == 0:
        return Constraint.Skip
    return abs(model.g_H[s, t] - model.g_H[s, t-1]) <= r_H[s]
model.r_h = Constraint(S, T, rule=r_hydrogen)

# Solar PV constraint:
def ub_g_solar(model, s, t):  # g_P <= f_P k_P
    return model.g_P[s, t] <= f_P[s, t]*model.k_P[s]
model.ub_g_P = Constraint(S, T, rule=ub_g_solar)

# SMR constraints:
def ub_g_smr(model, s, t):  # g_M <= u_M k_M
    return model.g_M[s, t] <= model.u_M[s]*k_M[s]
model.ub_g_M = Constraint(S, T, rule=ub_g_smr)

def r_smr(model, s, t):    # ramping constraints
    if t == 0:
        return Constraint.Skip                          # here needs to specify initial SOC conditions
    return abs(model.g_M[s, t] - model.g_M[s, t-1]) <= r_M[s]
model.r_m = Constraint(S, T, rule=r_smr)

# Wholesale power constraints:
def ub_g_wholesale(model, s, t):  # g_W <= u_W k_W
    return model.g_W[s, t] <= sum(model.u_W[s, i]*k_W[i] for i in I)
model.ub_g_W = Constraint(S, T, rule=ub_g_wholesale)

def trans_line_limit(model, s):  # only build 1 transmission line at one station
    return sum(model.u_W[s, i] for i in I) <= 1
model.trans_const = Constraint(S, rule=trans_line_limit)

# Market clearing condition:
def market_clearing(model, s, t):
    return model.g_B[s, t] + model.g_H[s, t] + model.g_P[s, t] + model.g_M[s, t] \
           + model.g_W[s, t] >= d_E[s, t] + model.d_B[s, t] + model.d_H[s, t]
model.mc_const = Constraint(S, T, rule=market_clearing)


# Objective function:
def obj_function(model):
    return sum(p_BK[s]*model.k_B[s] + p_BC[s]*model.e_B[s] + sum(p_BE[s, t]*model.g_B[s, t] for t in T) for s in S) \
           + sum(p_HK[s]*model.k_H[s] + p_HC[s]*model.e_H[s] + sum(p_HE[s, t]*model.g_H[s, t] for t in T) for s in S) \
           + sum(p_PK[s]*model.k_P[s] + sum(p_PE[s, t]*model.g_P[s, t] for t in T) for s in S) \
           + sum(p_MK[s]*model.u_M[s]*k_M[s] + sum(p_ME[s, t]*model.g_M[s, t] for t in T) for s in S) \
           + sum(sum(p_WK[s, i]*k_W[i] + (1+p_WO[s])*(p_WI[s, i]+p_WC[s, i] + p_WL[s, i])*l_W[s, i]*model.u_W[s, i] for i in I) + sum(p_WE[s, t]*model.g_W[s, t] for t in T) for s in S)
model.obj_func = Objective(rule=obj_function)


# %% Solve HDV model:
# model.dual = Suffix(direction=Suffix.IMPORT_EXPORT)
solver = SolverFactory('cplex')
results = solver.solve(model, tee=True)
model.pprint()

if (results.solver.status == SolverStatus.ok) and (results.solver.termination_condition == TerminationCondition.optimal):
    print('Solution is feasible')
elif results.solver.termination_condition == TerminationCondition.infeasible:
    print('Solution is infeasible')
else:
    # Something else is wrong
    print('Solver Status: ', results.solver.status)

print("--- %s seconds ---" % (time.time() - start_time))
