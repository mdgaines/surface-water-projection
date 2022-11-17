
import xarray as xr
import rioxarray as riox
from glob import glob
import geopandas as gpd
from shapely.geometry import mapping, Polygon
# import rasterio as rio
import numpy as np
import pandas as pd
# from rasterstats import zonal_stats
import regionmask
import matplotlib.pyplot as plt

import os
import sys
# import pyproj

from cli import parse_clip_nc


def check_inputs():
    '''
        Check user inputs to make sure they are valid.
    '''
    args = parse_clip_nc()

    cutline = args.cutline
    if not os.path.exists(cutline):
        print(f'Sorry, {cutline} does not exist. Check your path!')
        sys.exit()
    
    data_set = args.data_set
    if data_set == 'all':
        FileList = glob('../data/ClimateData/GFDL-ESM2M_macav2livneh/*/*.nc') + \
                   glob('../data/ClimateData/HadGEM2-ES365_macav2livneh/*/*.nc') + \
                   glob('../data/ClimateData/IPSL-CM5A-LR_macav2livneh/*/*.nc') + \
                   glob('../data/ClimateData/MIROC5_macav2livneh/*/*.nc') + \
                   glob('../data/ClimateData/NorESM1-M_macav2livneh/*/*.nc')
    elif os.path.exists(data_set) and data_set.endswith('.nc'):
        FileList = glob(data_set)
    else:
        print(f'Sorry, {data_set} does not exist. Check your path!')
        sys.exit()

    if args.seasonal_avg:
        shp = args.shapefile
        if not os.path.exists(shp):
            print(f'Sorry, {shp} does not exist. Check your path!')
            sys.exit()
    else:
        shp=None

    return(cutline, FileList, args.seasonal_avg, shp)


def clip_NC(nc_path, cutline):
    '''
        Clip NC data to a study area.
    '''
    with xr.open_dataset(nc_path) as dta:
        ds = dta

    model = nc_path.split('_')[3]   # set which model we are assessing
    rcpX = nc_path.split('\\')[1]   # set which RCP path we are assessing
    var = nc_path.split('_')[2]     # set which variable we are assessing

    print('starting', rcpX, var)

    if var == 'pr':
        var_long = 'precipitation'
        var_short = 'PRECIP'
    elif var == 'tasmax':
        var_long = 'air_temperature'
        var_short = 'MAX_TEMP'
    elif var == 'tasmin':
        var_long = 'air_temperature'
        var_short = 'MIN_TEMP'
    else:
        print(f'Climate variable for {nc_path} not precip or temp, please check.')
        # break

    # sets crs to espg:4326 for all layers (same as what was in the metadata)
    ds_set_crs = ds[var_long].rio.write_crs("epsg:4326", inplace=True)
    # clipps all air_temperature layers to 
    clipped = ds_set_crs.rio.clip(cutline.geometry.apply(mapping), \
        cutline.crs, drop=False, invert=False)     # clips everything - all nan
    # clipped[0].plot() # plot first year-month

    return(clipped, model, rcpX, var_long, var_short)


def save_studyArea_avg(clipped, model, rcpX, var_short):
    '''
        Save the average of each date/season layer
    '''

    filename = ';'.join([model, rcpX, var_short+'.csv'])
    filename = filename.replace('_','-')
    filename = filename.replace(';','_')

    outpath = f'../data/ClimateData/macav2livneh_studyArea_avgs/{filename}'

    if not os.path.exists(outpath):

        sa_monthly_dict = {}

        for i in range(len(clipped.time)):
            date = str(clipped[i].time.values).split(' ')[0]
            sa_mean = np.nanmean(clipped[i])

            sa_monthly_dict[date] = sa_mean
        
        df = pd.DataFrame(sa_monthly_dict.values(), index=sa_monthly_dict.keys()).reset_index()
        df.columns = ['DATE', var_short + '_AVG']

        df.to_csv(outpath)

    return()


def save_zonal_avg(clipped, idx, huc_bounds, huc8_reproj, huc_mask, var_long, var_short, nc_path, model, rcpX):



    yr = int(str(clipped[idx].time.values).split('-')[0])
    mnth = int(str(clipped[idx].time.values).split('-')[1])
    # print(yr, mnth)

    if mnth == 3:
        szn = 'SPRING'
    elif mnth == 6:
        szn = 'SUMMER'
    elif mnth == 9:
        szn = 'FALL'
    elif mnth == 12:
        szn = 'WINTER'
    else:
        print('error')

    output = f'{os.path.dirname(nc_path)}/zonal_avg/{model}_{rcpX}_{yr}_{szn}_{var_short}_AVG.csv'

    if not os.path.exists(output):
        if not os.path.exists(f'{os.path.dirname(nc_path)}/zonal_avg'):
            os.mkdir(f'{os.path.dirname(nc_path)}/zonal_avg')

        if var_short == 'PRECIP': # get seasonal sum for precip (NC data already monthly sum)
            szn_avg = clipped[idx] + clipped[idx+1] + clipped[idx+2]
        else:
            # get seasonal average of climate variable
            szn_avg = (clipped[idx] + clipped[idx+1] + clipped[idx+2]) / 3
        
        # Subset the data - this is now a dataarray rather than a DataSet
        clipped_sub = szn_avg.sel(
            lon=slice(huc_bounds["lon"][0], huc_bounds["lon"][1]),
            lat=slice(huc_bounds["lat"][0], huc_bounds["lat"][1])).where(huc_mask)

        zs = clipped_sub.groupby("region").mean(["lat", "lon"]).to_dataframe()
        zs['huc8'] = huc8_reproj['huc8']
        zs['YEAR'] = yr
        zs['SEASON'] = szn
        zs.rename(columns={var_long:'AVG_'+var_short}, inplace=True)

        zs.to_csv(output)
        print(f'{model}_{rcpX}_{yr}_{szn}_{var_short}_AVG.csv saved.')

    return()  


# Helper Function to extract AOI
def get_aoi(shp, world=True):
    """
    Takes a geopandas object and converts it to a lat/ lon
    extent 
    
    from: https://www.earthdatascience.org/courses/use-data-open-source-python/hierarchical-data-formats-hdf/subset-netcdf4-climate-data-spatially-aoi/
    """

    lon_lat = {}
    # Get lat min, max
    aoi_lat = [float(shp.total_bounds[1]), float(shp.total_bounds[3])]
    aoi_lon = [float(shp.total_bounds[0]), float(shp.total_bounds[2])]

    # Handle the 0-360 lon values
    if world:
        aoi_lon[0] = aoi_lon[0] + 360
        aoi_lon[1] = aoi_lon[1] + 360
    lon_lat["lon"] = aoi_lon
    lon_lat["lat"] = aoi_lat
    return lon_lat


def main():

    cutline, FileList, seasonal_avg, shp_path = check_inputs()

    cutline_shp = gpd.read_file(cutline)
    # sets coords to 360 scale
    row=next(cutline_shp.iterrows())[1]
    coords=np.array(row['geometry'].exterior.coords)
    coords[:,0]=coords[:,0]+360.
    newpoly=Polygon(coords)
    cutline_shp_360 = gpd.GeoDataFrame(index=[0], crs='epsg:4326', geometry=[newpoly])

    ### Loop through NC files and clip them by cutline
    for j in range(len(FileList)):
        # read in netcdf

        nc_path = FileList[j]

        clipped, model, rcpX, var_long, var_short = clip_NC(nc_path, cutline=cutline_shp_360)

        save_studyArea_avg(clipped, model, rcpX, var_short)

        if seasonal_avg and j == 0:
            shp = gpd.read_file(shp_path)
            huc8_reproj = shp.to_crs(crs='epsg:4326')
            del(shp)

            #######################################
            # adapted from https://www.earthdatascience.org/courses/use-data-open-source-python/hierarchical-data-formats-hdf/subset-netcdf4-climate-data-spatially-aoi/

            huc_mask = regionmask.mask_3D_geopandas(huc8_reproj, clipped.lon, clipped.lat)

            huc_bounds = get_aoi(huc8_reproj)

            #######################################

        # Loop through climate data in 3-month seasons starting in Spring (i.e., March 1)
        # for all years (2006-2100)
        for i in range(2, len(clipped.time), 3):
            # break on 2099-12
            if i == len(clipped.time) - 1:
                break
        
            save_zonal_avg(clipped, i, huc_bounds, huc8_reproj, huc_mask, \
                var_long, var_short, nc_path, model, rcpX)

            # break

if __name__ == '__main__':
    main()

        
    



# d = '../data/ClimateData/GFDL-ESM2M_macav2livneh/RCP85/macav2livneh_tasmax_GFDL-ESM2M_r1i1p1_rcp85_2006_2099_CONUS_monthly_aggregated.nc'
# ds = xr.open_dataset(d)
# ds.head

# ds.air_temperature[0]

# plt.figure()
# ds.air_temperature[0].plot()

        # # sets coords to 360 scale
        # lst = [[0,0]]*310
        # for i in range(len(huc8_reproj)):
        #     row=huc8_reproj.iloc[i]
        #     coords=np.array(row['geometry'].exterior.coords)
        #     coords[:,0]=coords[:,0]+360.
        #     newpoly=Polygon(coords)
        #     lst[i] = [newpoly, row['huc8']]
        
        # huc8_reproj_360 = gpd.GeoDataFrame(pd.DataFrame(lst)[1], crs='epsg:4326',\
        #      geometry = pd.DataFrame(lst)[0])
        # huc8_reproj_360.rename(columns={1:'huc8'},inplace=True)

        # huc8 = huc8_reproj[['huc8','geometry']]

    # # set empty df
    # huc8_df = pd.DataFrame(huc8['huc8'])
    # huc8_df['YEAR'] = 0
    # huc8_df['SEASON'] = ''
    # huc8_df['AVG_'+var_short] = 0


        # # Run Zonal Stats to get avg of avg climate variable
        # affine = szn_avg.rio.transform()
        # affine[2] = affine[2] - 360
        # affine[4] = affine[4] * -1

        # # using all_touhched = True because we are taking the precipitation or max temperature mean
        # # and some edge HUCs are limited in their projected precip/max temp values

        # #### Problem running Zonal Stats with the 360 longitude
        # zs = zonal_stats(huc8, szn_avg.values, affine=affine, stats=['mean', 'count'], nodata=0, all_touched = True)
        # zs