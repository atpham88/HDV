"""
An Pham
Updated 05/12/2021
Clean solar CF after downloaded from NREL.
Combine all downloaded solar CF files into one master file

"""
import pandas as pd
import geopandas as gpd
from geopandas import GeoDataFrame
import matplotlib.pyplot as plt
from shapely.geometry import Point
import folium

results_cat = 0             # == 1: with h2, ==0: without h2

if results_cat == 0:
    m_dir = "C:\\Users\\atpha\\Documents\\Postdocs\\Projects\\HDV\Results\\no h2 results\\"
elif results_cat == 1:
    m_dir = "C:\\Users\\atpha\\Documents\\Postdocs\\Projects\\HDV\Results\\with h2 results\\"

stations_to_exclude = [3, 163, 166, 169]
stations_to_exclude = [x - 1 for x in stations_to_exclude]

power_rating_results = "2021-08-13-135948%5Cpower rating all stations"
power_rating = pd.read_excel(m_dir + power_rating_results + ".xlsx")
power_rating = power_rating.sort_values(['station no'], ignore_index=True)

lon_lat = pd.read_excel("C:\\Users\\atpha\\Documents\\Postdocs\\Projects\\HDV\\Data\\load_station\\station_load_profile_case1.xlsx", header=None)
lon_lat = lon_lat.iloc[:,2:4]
lon_lat.columns = ['lon', 'lat']
lon_lat.reset_index()
lon_lat = lon_lat.drop(stations_to_exclude)
lon_lat.reset_index(drop=True, inplace=True)

power_rating_final = pd.concat([lon_lat,power_rating], axis=1)

# Visualize the charging station locations on map:
# Read in the US States shape file:
us_states = gpd.read_file('C:/Users/atpha/Documents/Research/PJM Maps/tl_2016_us_state/tl_2016_us_state.shp')
us_states_in_NA = us_states.drop(labels=[31, 34, 35, 36, 40, 41, 49], axis=0)

for i in range(0,len(power_rating_final)):
   folium.Circle(
      location=[power_rating_final.iloc[i]['lat'], power_rating_final.iloc[i]['lon']],
      popup=power_rating_final.iloc[i]['number of SMR modules'],
      radius=float(power_rating_final.iloc[i]['number of SMR modules'])*20000,
      color='crimson',
      fill=True,
      fill_color='crimson'
   ).add_to(us_states_in_NA)



# Spatial join station data with states:
geometry = [Point(xy) for xy in zip(power_rating_final['lon'], power_rating_final['lat'])]
station_location_frame = GeoDataFrame(power_rating_final, crs='epsg:4269', geometry=geometry)

station_with_states = gpd.sjoin(station_location_frame, us_states_in_NA, how="inner", op='intersects')

station_with_states[["number of SMR modules", "solar capacity (MW)"]] = station_with_states[["number of SMR modules", "solar capacity (MW)"]].apply(pd.to_numeric)
station_with_states = station_with_states[station_with_states['number of SMR modules'] > 0]


fig, ax = plt.subplots()
ax.set_aspect('equal')
us_states_in_NA.plot(ax=ax, color='white', figsize=(200, 100), edgecolors='black', linewidth=0.3)
station_with_states.plot(ax=ax, color='grey', figsize=(200, 100), marker='o', markersize=station_with_states['number of SMR modules'],alpha=0.7, categorical=False, legend=True)
plt.axis('off')
minx, miny, maxx, maxy = us_states_in_NA.total_bounds
ax.set_xlim(minx, maxx)
ax.set_ylim(miny, maxy)
plt.title('HDV Charging Stations and SMR Investments', fontsize=14)
plt.show()
fig.savefig(m_dir + 'SMR investment map', dpi=300)
