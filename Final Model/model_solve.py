# Formulate and solve HDV model in Pyomo using Cplex

# %% Import modules:
from pyomo.environ import *
from pyomo.opt import SolverFactory
import time
from get_input import *

start_time = time.time()

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


def x_battery(model, s, t):
    if t == 0:
        return model.x_B[s, t] == 0.5 * model.e_B[s]  # here needs to specify initial SOC conditions
    return model.x_B[s, t] == model.x_B[s, t - 1] + model.d_B[s, t] - model.g_B[s, t]


model.x_b = Constraint(S, T, rule=x_battery)


def r_battery_up(model, s, t):
    if t == 0:
        return Constraint.Skip
    else:
        return model.g_B[s, t] - model.g_B[s, t - 1] <= r_B * model.e_B[s]


model.r_b_up = Constraint(S, T, rule=r_battery_up)


def r_battery_down(model, s, t):
    if t == 0:
        return Constraint.Skip
    else:
        return model.g_B[s, t - 1] - model.g_B[s, t] <= r_B * model.e_B[s]


model.r_b_down = Constraint(S, T, rule=r_battery_down)


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


def r_hydrogen_up(model, s, t):
    if t == 0:
        return Constraint.Skip
    else:
        return model.g_H[s, t] - model.g_H[s, t - 1] <= r_H * model.e_H[s]


model.r_h_up = Constraint(S, T, rule=r_hydrogen_up)


def r_hydrogen_down(model, s, t):
    if t == 0:
        return Constraint.Skip
    else:
        return model.g_H[s, t - 1] - model.g_H[s, t] <= r_H * model.e_H[s]


model.r_h_down = Constraint(S, T, rule=r_hydrogen_down)


# Solar PV constraint:
def ub_g_solar(model, s, t):
    return model.g_P[s, t] <= f_P[s, t] * model.k_P[s]


model.ub_g_P = Constraint(S, T, rule=ub_g_solar)


# SMR constraints:
def ub_g_smr(model, s, t):
    return model.g_M[s, t] <= model.u_M[s] * k_M


model.ub_g_M = Constraint(S, T, rule=ub_g_smr)


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
           + model.g_W[s, t] >= d_E[s, t] + model.d_B[s, t] + model.d_H[s, t]


model.mc_const = Constraint(S, T, rule=market_clearing)


# Objective function:
def obj_function(model):
    return sum(p_BK * model.k_B[s] + p_BC * model.e_B[s] + sum(p_BE * model.g_B[s, t] for t in T) for s in S) \
           + sum(p_HK * model.k_H[s] + p_HC * model.e_H[s] + sum(p_HE * model.g_H[s, t] for t in T) for s in S) \
           + sum(p_PK * model.k_P[s] + sum(p_PE * model.g_P[s, t] for t in T) for s in S) \
           + sum(p_MK * model.u_M[s] * k_M + sum(p_ME * model.g_M[s, t] for t in T) for s in S) \
           + sum(sum(p_WK * k_W[i] + (1 + p_WO) * (p_WI + p_WC[s, i] + p_WL) * l_W[s, i] * model.u_W[s, i] for i in I) + sum(p_WE[s, t] * model.g_W[s, t] for t in T) for s in S)


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

# %% Print variable outputs:
print('Total Cost:', value(model.obj_func))

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
