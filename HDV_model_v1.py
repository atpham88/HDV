# HDV model v1
# Coded on April 21, 2021
# Language: Pyomo/Python

import pandas as pd
from pyomo.environ import *
from pyomo.opt import SolverFactory
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rc

# Set model type:
model = ConcreteModel(name="HDV_model")

## Read in data:
EGU_temp = pd.read_excel('C:/Users/atpha/Documents/Classes/Python workshop/Pyomo/'
                          'Power System Design/cost_uc.xlsx','EGUs')
demand_temp = pd.read_excel('C:/Users/atpha/Documents/Classes/Python workshop/Pyomo/'
                          'Power System Design/cost_uc.xlsx','demand')
wind_temp = pd.read_excel('C:/Users/atpha/Documents/Classes/Python workshop/Pyomo/'
                          'Power System Design/cost_uc.xlsx','wind')

# Calculate total variable cost:
EGU_temp['tvc'] = EGU_temp['fuelcost']*EGU_temp['heatrate']/1000 + EGU_temp['vom']

# Convert dataframe into dictionary:
EGU = EGU_temp.to_dict()
demand = demand_temp.to_dict()
wind = wind_temp.to_dict()

# Define set:
U = list(range(10))  # Set of charging stations
H = list(range(24))  # Set of hours

# Define parameters:
pmax = EGU['pmax']                       # Max gen
pmin = EGU['pmin']                       # Min gen
tvc = EGU['tvc']                         # Total variable cost
startup_cost = EGU['startupcost']        # Start-up cost
load = demand['load']                    # Load
wind_cf = wind['wind_cf']                # Wind capacity factor

# Define variables:
model.x = Var(U, H, within=NonNegativeReals)    # generation indexed by unit and hours
model.v = Var(U, H, within=Binary)              # binary variable, indicating whether the unit is on or off in a particular hour
model.vU = Var(U, H, within=Binary)             # binary variable, indicating whether the unit is started in an hour
model.vD = Var(U, H, within=Binary)             # binary variable, indicating whether the unit is shut down in an hour

# Define Objective function:
def obj_function(model):
    return sum(model.vU[u, h]*startup_cost[u] for u in U for h in H)\
           + sum(model.x[u, h]*tvc[u] for u in U for h in H)


model.obj_func = Objective(rule=obj_function)


# Define constraints:
def upperbound(model,u,h):
    return model.x[u,h] <= model.v[u, h]*pmax[u]


model.ub = Constraint(U,H, rule=upperbound)

def lowerbound(model,u,h):
    return model.x[u,h] >= model.v[u,h]*pmin[u]


model.lb = Constraint(U,H, rule=lowerbound)

# Market Clearing condition:
def market_clearing(model, h):
    return sum(model.x[u, h] for u in U) == load[h]


model.market_clearing_const = Constraint(H, rule=market_clearing)

# Turn on/shut down constraints:
def on_off(model,u,h):
    if h ==0:
        return Constraint.Skip
    return model.v[u, h] == model.v[u, h - 1] + model.vU[u, h] - model.vD[u, h]

model.onoff_const = Constraint(U,H, rule=on_off)

# Solve the model and report results:
# model.dual = Suffix(direction=Suffix.IMPORT_EXPORT)      # To get dual variables (challenging/not possible for MIP models)
solver = SolverFactory('cplex')
results = solver.solve(model, tee=True)
model.pprint()


if (results.solver.status == SolverStatus.ok) and (results.solver.termination_condition == TerminationCondition.optimal):
    print('Solution is feasible')
elif (results.solver.termination_condition == TerminationCondition.infeasible):
    print('Solution is infeasible')
else:
    # Something else is wrong
    print('Solver Status: ',  results.solver.status)
