# Specify all model parameters


# %% Define Switches:
super_comp = 0                           # ==1: run on super computer, ==0: run on local laptop
run_all_station = 0                      # ==1: run all stations at once, ==0: run only one station
station_no = 148                         # if only run 1 station, specify the station (enter number between 1 and 170)
ramping_const = 1                        # ==1: ramping constraints for SMR activated, ==0: no ramping constraints

# %% Specify model scope:
cap_class = 7                            # Number of transmission capacity classes
station = 170                            # Number of charging stations
hour = 8760                              # Number of hours in a typical year
day = 365                                # Number of days in a typical year

# %% Model main parameters:

# Interest rate:
ir = 0.015

# Battery:
capex_B = 1455000                       # Assuming 4hr moderate battery storage ($/MW)
life_B = 15                             # Battery life time (years)
fixed_OM_B = 36370                      # Fixed battery OM cost ($/MW-year)
p_BC_fixed = 0.004                      # Battery energy cost ($/kWh/h)
p_BE = 0                                # Battery operating cost ($/MW)

# H2:
p_HK_fixed = 0.0148                     # H2 capital cost ($/kW)
p_HC_fixed = 1.47*10**(-6)              # Dowling et al ($/kWh/h)
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
p_WK = 100000                           # Transmission capacity cost ($/MW-year)
p_WO = 0.1                              # Transmission over-head cost (percentage)
p_WI = 50000                            # Transmission infrastructure cost
p_WL = 200000                           # Land cost
