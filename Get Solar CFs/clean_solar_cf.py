'''
Clean solar CF after downloaded from NREL.
Combine all downloaded solar CF files into one master file

'''
import os
import pandas as pd
import csv
import glob

m_dir = 'C:\\Users\\atpha\\Documents\\Postdocs\\Projects\\HDV\\powGen-wtk-nsrdb-main\\powGen-wtk-nsrdb-main\\output\\solar\\'

cwd = os.path.dirname(m_dir)
all_files = glob.glob(m_dir + "solar*.csv")

li = []
for filename in all_files:
    df = pd.read_csv(filename, usecols=range(1,2))
    li.append(df)
frame = pd.concat(li, axis=1)
frame_T = frame.T

frame_T_index = frame_T.reset_index()

frame_T.insert(0, 'lat', 0)
frame_T.insert(0, 'lon', 0)

frame_T_index.rename(columns={'index':'coordinates'}, inplace=True)
frame_split = frame_T_index["coordinates"].str.split(" ", n = 1, expand = True)

frame_T_index["lat"] = frame_split[0]
frame_T_index["lon"] = frame_split[1]
frame_T_index.drop("coordinates", axis=1, inplace=True)

cols = frame_T_index.columns.tolist()
cols = cols[-1:] + cols[:-1]
cols = cols[0:1] + cols[-1:]  + cols[1:-1]

frame_T_index = frame_T_index[cols]
frame_T_index["lon"] = pd.to_numeric(frame_T_index["lon"], downcast="float")
frame_T_index["lat"] = pd.to_numeric(frame_T_index["lat"], downcast="float")
frame_T_index = frame_T_index.sort_values(['lon', 'lat'], ignore_index=True)

# Create cross-walk between station numbers and coordinates:
station_location = frame_T_index[['lon', 'lat']]

# Save to csv:
station_location.to_csv('C:\\Users\\atpha\\Documents\\Postdocs\\Projects\\HDV\\Data\\solar\\' + 'station_crosswalk.csv')
frame_T_index.to_csv('C:\\Users\\atpha\\Documents\\Postdocs\\Projects\\HDV\\Data\\solar\\' + 'solar_cf_by_point.csv')

