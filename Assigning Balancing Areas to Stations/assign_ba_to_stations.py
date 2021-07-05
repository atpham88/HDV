"""
An Pham
Updated 06/30/2021
Assign stations with no BA assignment to BA associated to closest lon/lat
"""

import pandas as pd
import geopandas as gpd
from geopandas import GeoDataFrame
import matplotlib.pyplot as plt
from shapely.geometry import Point
from shapely.ops import nearest_points

unassigned_stations = pd.read_excel("C:\\Users\\atpha\\Documents\\Postdocs\\Projects\\HDV\\Data\\charging station profiles\\unassigned_station.xlsx")
assigned_stations = pd.read_excel("C:\\Users\\atpha\\Documents\\Postdocs\\Projects\\HDV\\Data\\charging station profiles\\assigned_station.xlsx")

df_unassigned = gpd.GeoDataFrame(unassigned_stations)
df_assigned = gpd.GeoDataFrame(assigned_stations)

geometry_unassigned = [Point(xy) for xy in zip(df_unassigned.lon, df_unassigned.lat)]
gdf_unassigned = GeoDataFrame(df_unassigned, crs="EPSG:4326", geometry=geometry_unassigned)

geometry_assigned = [Point(xy) for xy in zip(df_assigned.lon, df_assigned.lat)]
gdf_assigned = GeoDataFrame(df_assigned, crs="EPSG:4326", geometry=geometry_assigned)
gdf_unassigned.insert(4, 'nearest_geometry', None)

for index, row in gdf_unassigned.iterrows():
    point = row.geometry
    multipoint = gdf_assigned.drop(index, axis=0).geometry.unary_union
    queried_geom, nearest_geom = nearest_points(point, multipoint)
    gdf_unassigned.loc[index, 'nearest_geometry'] = nearest_geom

gdf_unassigned.to_excel('C:\\Users\\atpha\\Documents\\Postdocs\\Projects\\HDV\\Data\\charging station profiles\\unassigned_station_ba.xlsx')
