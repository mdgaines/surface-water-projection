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
# import pyproj


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



huc_outline = gpd.read_file('../data/Shapefiles/HUC08_outline_clean.shp')

# sets coords to 360 scale
row=next(huc_outline.iterrows())[1]
coords=np.array(row['geometry'].exterior.coords)
coords[:,0]=coords[:,0]+360.
newpoly=Polygon(coords)
huc_outline_360 = gpd.GeoDataFrame(index=[0], crs='epsg:4326', geometry=[newpoly])


# List filepaths for all bands in the scence
FileList = glob(os.path.join(r'../data/ClimateData/GFDL-ESM2M_macav2livneh/*','*.nc'))


for j in range(len(FileList)):
    # read in netcdf
    with xr.open_dataset(FileList[j]) as dta:
        ds = dta

    rcpX = FileList[j].split('\\')[1]   # set which RCP path we are looking at
    var = FileList[j].split('_')[2]     # set which variable we are assessing

    print('starting', rcpX, var)

    if var == 'pr':
        var_long = 'precipitation'
        var_short = 'PRECIP'
    elif var == 'tasmax':
        var_long = 'air_temperature'
        var_short = 'MAX_TEMP'
    else:
        print('Climate variable for {} not precip or temp, please check.'.format(FileList[j]))
        break

    # sets crs to espg:4326 for all layers (same as what was in the metadata)
    ds_set_crs = ds[var_long].rio.write_crs("epsg:4326", inplace=True)
    # clipps all air_temperature layers to 
    clipped = ds_set_crs.rio.clip(huc_outline_360.geometry.apply(mapping), \
        huc_outline_360.crs, drop=False, invert=False)     # clips everything - all nan
    # clipped[0].plot() # plot first year-month

    if j == 0:
        shp = gpd.read_file('../data/Shapefiles/HUC08_trimmed.shp')
        huc8_reproj = shp.to_crs(crs='epsg:4326')
        # point_reproj = pt_shp.to_crs(dataset.crs.to_dict())
        del(shp)

        #######################################
        # adapted from https://www.earthdatascience.org/courses/use-data-open-source-python/hierarchical-data-formats-hdf/subset-netcdf4-climate-data-spatially-aoi/

        huc_mask = regionmask.mask_3D_geopandas(huc8_reproj, clipped.lon, clipped.lat)

        huc_bounds = get_aoi(huc8_reproj)

        #######################################


    # Loop through climate data in 3-month seasons starting in Spring (i.e., March 1)
    # for all years (2006-2100)
    for i in range(2, 1128, 3):
        # break on 2099-12
        if i == 1127:
            break

        yr = int(str(clipped[i].time.values).split('-')[0])
        mnth = int(str(clipped[i].time.values).split('-')[1])
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

        # get seasonal average of climate variable
        szn_avg = (clipped[i] + clipped[i+1] + clipped[i+2]) / 3
        
        # Subset the data - this is now a dataarray rather than a DataSet
        clipped_sub = szn_avg.sel(
            lon=slice(huc_bounds["lon"][0], huc_bounds["lon"][1]),
            lat=slice(huc_bounds["lat"][0], huc_bounds["lat"][1])).where(huc_mask)

        zs = clipped_sub.groupby("region").mean(["lat", "lon"]).to_dataframe()
        zs['huc8'] = huc8_reproj['huc8']
        zs['YEAR'] = yr
        zs['SEASON'] = szn
        zs.rename(columns={var_long:'AVG_'+var_short}, inplace=True)

        output = '../data/ClimateData/GFDL-ESM2M_macav2livneh/{0}/zonal_avg/{0}_{1}_{2}_{3}_AVG.csv'.format(rcpX, var_short, yr, szn)

        if not os.path.exists(output):
            zs.to_csv(output)
            print('{0}_{1}_{2}_{3}_AVG.csv'.format(rcpX, var_short, yr, szn), "saved.")
    
        # break
        
    



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