"""
An Pham
Updated 05/12/2021
Clean solar CF after downloaded from NREL.
Combine all downloaded solar CF files into one master file

"""
import os
import pandas as pd
import geopandas as gpd
from geopandas import GeoDataFrame
import matplotlib.pyplot as plt
from shapely.geometry import Point
import glob
from timezonefinder import TimezoneFinder

m_dir = 'C:\\Users\\atpha\\Documents\\Postdocs\\Projects\\HDV\\powGen-wtk-nsrdb-main\\powGen-wtk-nsrdb-main\\output\\solar\\'

# Combine all solar capacity factors from all files:
cwd = os.path.dirname(m_dir)
all_files = glob.glob(m_dir + "solar*.csv")

li = []
for filename in all_files:
    df = pd.read_csv(filename, usecols=range(1, 2))
    li.append(df)
frame = pd.concat(li, axis=1)
frame_T = frame.T

frame_T_index = frame_T.reset_index()

frame_T.insert(0, 'lat', 0)
frame_T.insert(0, 'lon', 0)

frame_T_index.rename(columns={'index': 'coordinates'}, inplace=True)
frame_split = frame_T_index["coordinates"].str.split(" ", n=1, expand=True)

frame_T_index["lat"] = frame_split[0]
frame_T_index["lon"] = frame_split[1]
frame_T_index.drop("coordinates", axis=1, inplace=True)

cols = frame_T_index.columns.tolist()
cols = cols[-1:] + cols[:-1]
cols = cols[0:1] + cols[-1:] + cols[1:-1]

frame_T_index = frame_T_index[cols]
frame_T_index["lon"] = pd.to_numeric(frame_T_index["lon"], downcast="float")
frame_T_index["lat"] = pd.to_numeric(frame_T_index["lat"], downcast="float")
frame_T_index = frame_T_index.sort_values(['lon', 'lat'], ignore_index=True)

# Create cross-walk between station numbers and coordinates:
station_location = frame_T_index[['lon', 'lat']]

# Print time zone for each long/lat:
tf = TimezoneFinder()
timezone = [''] * 219
for s in range(219):
    longitude, latitude = station_location.iloc[s, 0], station_location.iloc[s, 1]
    timezone[s] = tf.timezone_at(lng=longitude, lat=latitude)

timezone_frame = pd.DataFrame(timezone)

frame_T_index["time zone"] = timezone_frame
cols = frame_T_index.columns.tolist()
cols = cols[-1:] + cols[:-1]
frame_T_index = frame_T_index[cols]

# for s in range(170):
# if frame_T_index["time zone"][s] == 'America/Los Angeles':
# frame_T_index.shift(periods=7, axis="columns")
# elif frame_T_index["time zone"][s] == 'America/Boise':
# frame_T_index.shift(periods=6, axis="columns")

# Save to csv:
station_location.to_csv('C:\\Users\\atpha\\Documents\\Postdocs\\Projects\\HDV\\Data\\solar\\' + 'station_crosswalk.csv')
frame_T_index.to_csv('C:\\Users\\atpha\\Documents\\Postdocs\\Projects\\HDV\\Data\\solar\\' + 'solar_cf_by_point.csv')

# Visualize the charging station locations on map:
# Read in the US States shape file:
us_states = gpd.read_file('C:/Users/atpha/Documents/Research/PJM Maps/tl_2016_us_state/tl_2016_us_state.shp')
us_states_in_NA = us_states.drop(labels=[31, 34, 35, 36, 40, 41, 49], axis=0)

# Spatial join station data with states:
geometry = [Point(xy) for xy in zip(frame_T_index['lon'], frame_T_index['lat'])]
station_location_frame = GeoDataFrame(frame_T_index, crs='epsg:4269', geometry=geometry)

station_with_states = gpd.sjoin(station_location_frame, us_states_in_NA, how="inner", op='intersects')

fig, ax = plt.subplots()
ax.set_aspect('equal')
us_states_in_NA.plot(ax=ax, color='white', figsize=(200, 100), edgecolors='black', linewidth=0.3)
station_with_states.plot(ax=ax, color='red', figsize=(200, 100), marker='o', markersize=5)
plt.axis('off')
minx, miny, maxx, maxy = us_states_in_NA.total_bounds
ax.set_xlim(minx, maxx)
ax.set_ylim(miny, maxy)
plt.title('HDV Charging Stations', fontsize=16)
plt.show()
fig.savefig('C:/Users/atpha/Documents/Postdocs/Projects/HDV/HDV_charging_stations.png', dpi=300)
