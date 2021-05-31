
def SMR_data(ir, life_M, capex_M, fixed_OM_M, var_OM_M, fuel_M):
    # SMR:
    CRF_M = ir / (1 - (1 + ir) ** (-life_M))
    p_MK = capex_M * CRF_M + fixed_OM_M         # SMR capital cost
    p_ME = var_OM_M + fuel_M                    # SMR operating cost ($/MWh)
    return p_MK, p_ME
