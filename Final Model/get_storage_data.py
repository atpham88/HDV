

def storage_data(ir, life_B, capex_B, fixed_OM_B, p_BC_fixed, p_HK_fixed, p_HC_fixed, hhv, h2_convert, h2_convert_elec):
    # Battery:
    CRF_B = ir / (1 - (1 + ir) ** (-life_B))    # Calculate battery capital recovery factor
    p_BK = capex_B * CRF_B + fixed_OM_B         # Battery power rating cost
    p_BC = p_BC_fixed * 1000 * 8760             # Battery energy cost ($/MWh)

    # H2:
    CRF_H = 0.0806
    c_H = h2_convert_elec*1000                  # kg to MWh
    p_HK = p_HK_fixed * 1000 * 8760             # $/MW-year
    p_HC = p_HC_fixed * CRF_H *8760             # ($/kg)
    return p_BK, p_BC, p_HK, p_HC, c_H
