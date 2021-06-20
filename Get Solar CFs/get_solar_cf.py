'''
ijbd
3/25/2021

Download solar and wind resource using the NREL API. 
Resource data is pulled from the NSRDB and WIND Toolkit API, then converted
to power generation with PySAM.

Revised by An Pham May 30 to retrieve solar CF by lon/lat for HDV project

'''
import argparse
import os
import numpy as np
import pandas as pd
import PySAM.ResourceTools as tools  
import PySAM.Pvwattsv7 as pv
import PySAM.Windpower as wp
from urllib.error import HTTPError
import geopandas as gpd
from shapely.geometry import Point
import matplotlib.pyplot as plt
import datetime
import time

# relative location 
local_path = os.path.dirname('C:\\Users\\atpha\\Documents\\Postdocs\\Projects\\HDV\\powGen-wtk-nsrdb-main\\powGen-wtk-nsrdb-main\\')
model_dir = 'C:\\Users\\atpha\\Documents\\Postdocs\\Projects\\HDV\\Data\\'
load_folder = 'charging station profiles\\'
station_location_raw = pd.read_excel(model_dir + load_folder + "charging_station_cases.xlsx")
geo_location = station_location_raw.iloc[:, 1:3]
station = 219

year = 2014
api_key = 'hXhxQrP4wSBTLtdkZfuOf3S3zGPPN6tcrbcw0PcJ'
email = 'anph@umich.edu'
geo_option = 'point'


for i in list(range(station)):
    lon = float(geo_location.iloc[i,0])
    lat = float(geo_location.iloc[i,1])

    # CLI
    parser = argparse.ArgumentParser(description='Download wind and solar resource data, then use PySAM to convert to hourly capacity factors.')
    parser.add_argument('--year', type=int, choices=[2007, 2008, 2009, 2010, 2011, 2012, 2013, 2014], help='Data year. Must be in 2007-2014 (inclusive).',required=True)
    parser.add_argument('--api-key', type=str, help='NREL API Key. Sign up @ https://developer.nrel.gov/signup/',required=True)
    parser.add_argument('--email', type=str, help='Email address.', required=True)
    parser.add_argument('--geometry', type=str, help='Option for choosing sites.', choices=['point','grid','state'], required=True)
    parser.add_argument('--save_resource', help='Save resource data in addition to generation data, THIS COULD TAKE A LOT OF DISK SPACE',action='store_true')
    parser.add_argument('--verbose', action='store_true')
    parser.add_argument('--lat', type=float, help='Required if geometry=point')
    parser.add_argument('--lon', type=float, help='Required if geometry=point')
    parser.add_argument('--min_lat', type=float)
    parser.add_argument('--max_lat', type=float)
    parser.add_argument('--min_lon', type=float)
    parser.add_argument('--max_lon', type=float)
    parser.add_argument('--states', nargs='+', type=str, help='Required if geometry=state, e.g. \'PA OH NY\'')
    parser.add_argument('--deg_resolution', type=float, default=.04, help='Approximate km resolution of grid. Used for geometry=state or geometry=grid, default .04')

    args = parser.parse_args(['--year', str(year), '--api-key', api_key,
                              '--email', email, '--geometry', geo_option, '--lat', str(lat), '--lon', str(lon)])


    def getSolarResourceData(year, lat, lon):
        solar_resource_filename = os.path.join(local_path,'resourceData/{lat}_{lon}_nsrdb.csv'.format(lat=lat,lon=lon))

        # check if download already complete
        if os.path.exists(solar_resource_filename):
            solarResource = pd.read_csv(solar_resource_filename)
        else:
            nsrdb_url = 'https://developer.nrel.gov/api/nsrdb/v2/solar/psm3-download.csv'

            params = { 'api_key' : args.api_key,
                    'email' : args.email,
                    'wkt' : 'POINT({lon}+{lat})'.format(lon=lon, lat=lat),
                    'names' : year,
                    'utc' : 'true'
                    }

            params_str = '&'.join(['{key}={value}'.format(key=key,value=params[key]) for key in params])
            download_url = '{nsrdb_url}?{params_str}'.format(nsrdb_url=nsrdb_url,params_str=params_str)

            solarResource = pd.read_csv(download_url)

            # Save
            if args.save_resource:
                solarResource.to_csv(solar_resource_filename,index=False)

        # Process for SAM
        solarResourceDescription = solarResource.head(1)

        # Get site of acutal lat/lon
        nsrdbLat = solarResourceDescription.at[0,'Latitude']
        nsrdbLon = solarResourceDescription.at[0,'Longitude']
        tz = solarResourceDescription.at[0,'Time Zone']
        elev = solarResourceDescription.at[0,'Elevation']


        solarResourceVariables = solarResource.iloc[1]
        solarResource.drop([0,1],inplace=True)
        solarResource.rename(columns=solarResourceVariables,inplace=True)


        solarResourceData = { 'tz' : float(tz),
                              'elev' : float(elev),
                              'lat' : float(nsrdbLat),
                              'lon' : float(nsrdbLon),
                              'year' : solarResource['Year'].values.astype(float).tolist(),
                              'month' : solarResource['Month'].values.astype(float).tolist(),
                              'day' : solarResource['Day'].values.astype(float).tolist(),
                              'hour' : solarResource['Hour'].values.astype(float).tolist(),
                              'minute' : solarResource['Minute'].values.astype(float).tolist(),
                              'dn' : solarResource['DNI'].values.astype(float).tolist(),
                              'df' : solarResource['DHI'].values.astype(float).tolist(),
                              'gh' : solarResource['GHI'].values.astype(float).tolist(),
                              'wspd' : solarResource['Wind Speed'].values.astype(float).tolist(),
                              'tdry' : solarResource['Temperature'].values.astype(float).tolist()
                            }
        return solarResourceData

    def getSolarCF(solarResourceData):
        s = pv.default("PVWattsNone")

        ##### Parameters #######
        s.SolarResource.solar_resource_data=solarResourceData
        s.SystemDesign.array_type = 0
        s.SystemDesign.azimuth = 180
        s.SystemDesign.tilt = abs(solarResourceData['lat'])
        nameplate_capacity = 1000 #kw
        s.SystemDesign.system_capacity = nameplate_capacity   # System Capacity (kW)
        s.SystemDesign.dc_ac_ratio = 1.1 #DC to AC ratio
        s.SystemDesign.inv_eff = 96 #default inverter eff @ rated power (%)
        s.SystemDesign.losses = 14 #other DC losses (%) (14% is default from documentation)
        ########################

        s.execute()
        solarCF = np.array(s.Outputs.ac) / (nameplate_capacity * 1000) #convert AC generation (w) to capacity factor

        if args.verbose:
            print('\t','Average Solar CF = {cf}'.format(cf=round(np.average(solarCF),2)))

        return solarCF

    def getWindSRW(year, lat, lon): # switch to srw

        windSRW = os.path.join(local_path,'resourceData/{lat}_{lon}_wtk.srw'.format(lat=lat,lon=lon))

        if not os.path.exists(windSRW):

            wtk_url = 'https://developer.nrel.gov/api/wind-toolkit/v2/wind/wtk-srw-download'

            params = { 'api_key' : args.api_key,
                    'email' : args.email,
                    'lat' : lat,
                    'lon' : lon,
                    'hubheight' : 100,
                    'year' : year,
                    'utc' : 'true'
                    }

            params_str = '&'.join(['{key}={value}'.format(key=key,value=params[key]) for key in params])
            download_url = '{wtk_url}?{params_str}'.format(wtk_url=wtk_url,params_str=params_str)

            windResource = pd.read_csv(download_url)

            # Process for SAM
            windResourceDescription = windResource.columns.values.tolist()

            # save srw
            windResource.to_csv(windSRW,index=False)

        # Find wind speed (for iec wind class)
        windSpeed100 = np.median(pd.read_csv(windSRW,skiprows=[0,1,3,4],usecols=['Speed']).values)

        # IEC wind classes
        if windSpeed100 >= 9:
            iecClass = 1
        elif windSpeed100 >= 8:
            iecClass = 2
        else:
            iecClass = 3

        return windSRW, iecClass

    def getWindCF(windSRW, iecClass, powerCurve):
        d = wp.default("WindPowerNone")

        powerout = powerCurve["Composite IEC Class {iecClass}".format(iecClass=iecClass)]
        speed = powerCurve["Wind Speed"]

        ##### Parameters #######
        d.Resource.wind_resource_filename = windSRW
        d.Resource.wind_resource_model_choice = 0
        d.Turbine.wind_turbine_powercurve_powerout = powerout
        d.Turbine.wind_turbine_powercurve_windspeeds = speed
        d.Turbine.wind_turbine_rotor_diameter = 90
        d.Turbine.wind_turbine_hub_ht = 100
        nameplate_capacity = 1500 #kw
        d.Farm.system_capacity = nameplate_capacity # System Capacity (kW)
        d.Farm.wind_farm_wake_model = 0
        d.Farm.wind_farm_xCoordinates = np.array([0]) # Lone turbine (centered at position 0,0 in farm)
        d.Farm.wind_farm_yCoordinates = np.array([0])
        ########################

        d.execute()
        windCF = np.array(d.Outputs.gen) / nameplate_capacity #convert AC generation (kw) to capacity factor

        if args.verbose:
            print('\t','Average Wind CF = {cf}'.format(cf=round(np.average(windCF),2)))

        if not args.save_resource:
            os.remove(windSRW)

        return windCF

    def getCoordinateList():
        if args.geometry == 'point':
            return [(args.lat,args.lon)]

        if args.geometry == 'grid':
            coordinates = []

            lat = args.min_lat
            while lat <= args.max_lat:
                lon= args.min_lon
                while lon <= args.max_lon:
                    coordinates.append((lat,lon))
                    lon += args.deg_resolution
                lat += args.deg_resolution
            return coordinates

        if args.geometry == 'state':
            states = args.states
            if 'CONTINENTAL' in args.states:
                states = ['AL','AZ','AR','CA','CO','CT','DE','DC','FL','GA','ID','IL',
                               'IN','IA','KS','KY','LA','ME','MD','MA','MI','MN','MS','MO',
                               'MT','NE','NV','NH','NH','NM','NY','NC','ND','OH','OK','OR',
                               'PA','RI','SC','SD','TN','TX','UT','VT','VA','WA','WV','WI',
                               'WY']
            # get outer bounds
            usShp = gpd.read_file(os.path.join(local_path,'states/s_11au16.shp'))
            statesShp = usShp[usShp['STATE'].isin(states)]
            bounds = statesShp.total_bounds

            coordinates = []

            bounds = statesShp.total_bounds
            min_lon = round(bounds[0],2)
            min_lat = round(bounds[1],2)
            max_lon = round(bounds[2],2)
            max_lat = round(bounds[3],2)

            lat = min_lat
            while lat <= max_lat:
                lon = min_lon
                while lon <= max_lon:
                    if statesShp.contains(Point(lon,lat)).any():
                        coordinates.append((lat,lon))
                    lon += args.deg_resolution
                lat += args.deg_resolution

            return coordinates

    def checkArgs():
        if args.geometry == 'point':
            if args.lat is None or args.lon is None:
                raise RuntimeError('For \'point\' geometry, please include --lat and --lon')
        if args.geometry == 'grid':
            if args.min_lat is None or args.max_lat is None or args.min_lon is None or args.max_lon is None:
                raise RuntimeError('For \'grid\' geometry, please include --min_lat, --max_lat, --min_lon and --max_lon')
        if args.deg_resolution < .04:
            raise RuntimeError('Please choose a resolution greater than .04 degrees')
        if not os.path.exists(os.path.join(local_path,'output')):
            os.mkdir(os.path.join(local_path,'output'))

    def dbgPlotCoords(coords):
        fig, ax = plt.subplots()
        lats = [t[0] for t in coords]
        lons = [t[1] for t in coords]
        ax.scatter(lons,lats,s=1)
        ax.set_xlim(-125,-65)
        ax.set_ylim(25,50)

        title = "{st} sites @ {res} degs ({ns} sites)".format(st=' '.join(args.states), res=args.deg_resolution, ns=len(lats))

        ax.set_title(title)

        filename = os.path.join(local_path,"{st}-sites-{res}-res-tmp.png".format(st='-'.join(args.states), res=args.deg_resolution))

        plt.savefig(filename)

    def getFilenames():
        # save
        if args.geometry == 'point':
            solar_filename = os.path.join(local_path,'output/solar_cf_{lat}_{lon}_{res}_{year}.csv'.format(lat=args.lat,lon=args.lon,res=args.deg_resolution,year=args.year))
            wind_filename = os.path.join(local_path,'output/wind_cf_{lat}_{lon}_{res}_{year}.csv'.format(lat=args.lat,lon=args.lon,res=args.deg_resolution,year=args.year))
        elif args.geometry == 'grid':
            solar_filename = os.path.join(local_path,'output/solar_cf_{minlat}_{minlon}_{maxlat}_{maxlon}_{res}_{year}.csv'.format(minlat=args.minlat,minlon=args.minlon,maxlat=args.maxlat,maxlon=args.maxlon,res=args.deg_resolution,year=args.year))
            wind_filename = os.path.join(local_path,'output/wind_cf_{minlat}_{minlon}_{maxlat}_{maxlon}_{year}.csv'.format(minlat=args.minlat,minlon=args.minlon,maxlat=args.maxlat,maxlon=args.maxlon,res=args.deg_resolution,year=args.year))
        else:
            solar_filename = os.path.join(local_path,'output/solar_cf_{st}_{res}_{year}.csv'.format(st='_'.join(args.states),res=args.deg_resolution,year=args.year))
            wind_filename = os.path.join(local_path,'output/wind_cf_{st}_{res}_{year}.csv'.format(st='_'.join(args.states),res=args.deg_resolution,year=args.year))

        return solar_filename, wind_filename

    def main():

        start_time = datetime.datetime.now()

        #############

        # check dependent arguments
        checkArgs()

        year = args.year

        solarGen = pd.DataFrame()
        windGen = pd.DataFrame()

        # get desired coordinates base on geometry
        coords = getCoordinateList()
        #dbgPlotCoords(coords)

        print('{nc} coordinates found...'.format(nc=len(coords)))

        # check for existing file
        '''
        This next if-statement checks to see if the same geometry/year has been run before.
        In case the program exceeds the API request limit, or fails for someother reason in the middle of a large job,
        this will allow it to pick up where it left off
        '''
        solar_filename, wind_filename = getFilenames()

        if os.path.exists(solar_filename) and os.path.exists(wind_filename): #if it found an existing job
            solarGen = pd.read_csv(solar_filename,index_col=0)
            windGen = pd.read_csv(wind_filename,index_col=0)

            completeSolarCoords = [(float(col.split(' ')[0]), float(col.split(' ')[1])) for col in solarGen.columns[:].values]
            completeWindCoords = [(float(col.split(' ')[0]), float(col.split(' ')[1])) for col in windGen.columns[:].values]

            # print summary
            print('Existing job with similar parameters found...')
            print('\t','year:',args.year)
            print('\t','geometry:',args.geometry)
            print('\t','deg_resolution:',args.deg_resolution)
            print('\t','{nc} of {nic} completed'.format(nc=min(len(completeSolarCoords),len(completeWindCoords)),nic=len(coords)))

            # only complete remaining coordinates
            coords = [coord for coord in coords if coord not in completeSolarCoords or coord not in completeWindCoords]

            if(len(coords) == 0):
                print('Nothing else to do... Exiting')
                return
        #endif

        powerCurve = pd.read_csv(os.path.join(local_path,'powerCurves/powerCurves.csv'))


        for i in range(len(coords)):

            lat = coords[i][0]
            lon = coords[i][1]

            if args.verbose:
                print('Running ({lat}, {lon})...'.format(lat=lat,lon=lon))

            try:
                solarResourceData = getSolarResourceData(year,lat,lon)
                solarGen['{lat} {lon}'.format(lat=lat, lon=lon)] = getSolarCF(solarResourceData)
                windSRW, iecClass = getWindSRW(year,lat,lon)
                windGen['{lat} {lon}'.format(lat=lat, lon=lon)] = getWindCF(windSRW,iecClass,powerCurve)
            except HTTPError as err:
                print('\t','Invalid coordinate, error code:',err.code)
                if err.code == 429:
                    solarGen.to_csv(solar_filename)
                    windGen.to_csv(wind_filename)
                    print('\t','Too many requests... (exiting)')
                    return
                time.sleep(2)


            # save progress
            if i%100 == 99:
                solarGen.to_csv(solar_filename)
                windGen.to_csv(wind_filename)

        # save
        solarGen.to_csv(solar_filename)
        windGen.to_csv(wind_filename)

        print('Solar capacity factors saved to csv: {sf}'.format(sf=solar_filename))
        print('Wind capacity factors saved to csv: {wf}'.format(wf=wind_filename))

        #########
        if args.verbose:
            end_time = datetime.datetime.now()
            print("Program start time...\t",start_time)
            print("Program end time...\t",end_time)
            print("Program run time... \t",end_time - start_time)
            print("Avg. coord run time...\t", (end_time-start_time)/len(coords))


    if __name__ == '__main__':
        main()