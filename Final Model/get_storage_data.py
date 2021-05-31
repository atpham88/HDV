

def storage_data(ir, life_B, capex_B, fixed_OM_B, p_BC_fixed, p_HK_fixed, p_HC_fixed):
    # Battery:
    CRF_B = ir / (1 - (1 + ir) ** (-life_B))  # Calculate battery capital recovery factor
    p_BK = capex_B * CRF_B + fixed_OM_B  # Battery power rating cost
    p_BC = p_BC_fixed * 1000 * 8760  # Battery energy cost ($/MWh)

    # H2:
    p_HK = p_HK_fixed * 1000 * 8760  # H2 power capacity cost
    p_HC = p_HC_fixed * 1000 * 8760  # H2 energy capacity cost
    return p_BK, p_BC, p_HK, p_HC
