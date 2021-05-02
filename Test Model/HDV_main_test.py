# Main HDV model script

import xlsxwriter as xw
from model_solve_test import *


# %% Check if model's constraints are satisfied:
def check_constraint():
    pass


# %% Write results into excel file:
results_book = xw.Workbook(results_folder + 'HDV_model_results.xlsx')
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
    station_number[s] = "station " + str(S[s] + 1)

for t in T:
    hour_number[t] = "hour " + str(T[t] + 1)

for i in I:
    class_number[i] = "class " + str(I[i] + 1)

# Write power rating results:
result_sheet_pr.write("B1", "battery power rating")
result_sheet_pr.write("C1", "battery energy capacity")
result_sheet_pr.write("D1", "h2 power rating")
result_sheet_pr.write("E1", "h2 energy capacity")
result_sheet_pr.write("F1", "solar capacity")
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
