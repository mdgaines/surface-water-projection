import concurrent.futures
import multiprocessing
import threading

import os
import sys
import tarfile
import shutil
import time
import zipfile
import rasterio
import numpy as np
import numpy.ma as ma
import pandas as pd
from glob import glob
from rasterstats import zonal_stats
import matplotlib.pyplot as plt
from rasterio.plot import plotting_extent
import geopandas as gpd
import earthpy as et
import earthpy.spatial as es
import earthpy.plot as ep

from itertools import islice

CHUNK = 100


def print_time(start, end, process = ""):
    """
        prints message with elapsed time of process
    """
    elapsed = end - start
    time_str = time.strftime("%H hr %M min %S sec", time.gmtime(elapsed))
    print(process, "completed in", time_str)
    return


def untar(in_dir):
    ''' Decompress DSWE tarfiles, extract INWM file,
        return list of raster objects ready to be mosaicked.'''

    print("Beginning Decompression")

    seasons = {'03':'Spring', '04':'Spring', '05':'Spring', \
               '06':'Summer', '07':'Summer', '08':'Summer', \
               '09':'Fall', '10':'Fall', '11':'Fall',\
               '12':'Winter', '01':'Winter', '02':'Winter'}

    # Get list of all tar files in our input directory
    tar_lst = glob(in_dir + '/*.tar') #[t for t in os.listdir(in_dir) if t.endswith(".tar")]

    for i in range(len(tar_lst)):
        tarfilename = os.path.basename(tar_lst[i])

        yr = tarfilename[15:19]       # index file name to get year of collection
        mnth = tarfilename[19:21]      # index to get month of collection
        szn = seasons[mnth]

        if mnth == '01' or mnth == '02':   # this will make sure our year winter
            yr = str(int(yr) - 1)          # makes sense (Dec18, Jan19, Feb19) for the ./2018/Winter folder
        
        out_dir = os.path.join('../data/DSWE_SE/raw_tifs',yr,szn)
            
        inwm_file = tarfilename.replace("SW.tar", "INWM.tif")
        tarfile_path = os.path.join(in_dir, tarfilename)

        if os.path.exists(os.path.join(out_dir, inwm_file)): # if the file already exists
            continue                                         # no need to untar it again

        try:
            tar = tarfile.open(tarfile_path)
            tar.extract(inwm_file, path=out_dir)
            tar.close()
        except:
            try:
                print('Hit an error with tarfile {} and INWM {}'.format(tarfilename, inwm_file))
                print('Unexpected error:', sys.exc_info()[0])
                tar.close()
            except UnboundLocalError:
                print('Hit another weird error. Apparently we could not declare the tar variable:', sys.exc_info()[0])

        if not ((i+1) % 200): # most folders with tar files have 2,000 so we should get updates every 10%
            print('{}% complete.'.format((i+1)/len(tar_lst) * 100))


    print("Decompression Complete")
    return

def composite(path, window):
    """
        docstring
    """

    with rasterio.open(path) as src:
        # print(f"Processing data: window={window}")
        array = src.read(window=window)

    high_conf_sum = np.count_nonzero(array==1,axis=0)
    high_conf_sum = high_conf_sum.astype('uint8')

    non_water_sum = np.count_nonzero(array==0,axis=0)
    non_water_sum = non_water_sum.astype('uint8')
    
    cloud_sum = np.count_nonzero(array==9,axis=0)
    cloud_sum = cloud_sum.astype('uint8')

    non_cloud_sum = np.count_nonzero(array<=4,axis=0)
    non_cloud_sum = non_cloud_sum.astype('uint8')

    all_water_sum = non_cloud_sum - non_water_sum
    all_water_sum = all_water_sum.astype('uint8')

    out = np.zeros(non_cloud_sum.shape)
    p_high_conf = np.round(np.divide(high_conf_sum, non_cloud_sum, out=out, where=non_cloud_sum!=0) * 100,0)
    p_high_conf = p_high_conf.astype('uint8')

    del(non_cloud_sum)
    
    all_stack = np.stack((high_conf_sum, all_water_sum, non_water_sum, cloud_sum, p_high_conf))
    
    return(window, all_stack)


def tileProcess(tile, out_fldr, sub_fldr, fldr, szn_num):
    '''
        Processes each full tile and sends it through the composite() function
        Writes output raster
    '''
    start_time_tile = time.time()
    print(tile, 'in process', os.getpid())

    # make output file path
    out_fl_path = os.path.join(out_fldr, fldr[-4:] + szn_num + '_' + tile + '_composite.tif')

    # if composite file does not exist
    if not os.path.exists(out_fl_path):
        rstr_fl_lst = glob(os.path.join(sub_fldr, '*_' + tile + '_*.tif'))
        print(len(rstr_fl_lst), "rasters for tile", tile)

        if not len(rstr_fl_lst):
            print('No rasters for tile', tile, "- moving to next tile")
            i += 1
            return()
        
        if len(rstr_fl_lst) == 1:
            stack_path = rstr_fl_lst[0]
        else:
            #### write a fully stacked raster for the year_szn_tile
            # read in rasters as stack
            stack_path = 'C:/Users/mdgaines/Documents/Research/temp/'+fldr[-4:]+szn_num+'_'+tile+'.tif' #'../temp/'+fldr[-4:]+szn_num+'_'+tile+'.tif' #
            array, raster_prof = es.stack(rstr_fl_lst, out_path = stack_path)
            del(array)
            del(raster_prof)

        ## Launch ThreadPoolExecutor
        with concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor:
            #### open newly made, stacked raster
            with rasterio.open(stack_path) as src:

                profile = src.profile
                profile.update(count=5, blockxsize=1024, blockysize=1024, tiled=True)

                start_time_composite = time.time()
                # write composite raster via concurrent processing
                with rasterio.open(out_fl_path, "w", **profile) as dst:
                    windows = (window for ij, window in dst.block_windows())
                    group = islice(windows, CHUNK)
                    futures = {executor.submit(composite, stack_path, window) for window in group}

                    while futures:

                        done, futures = concurrent.futures.wait(
                            futures, return_when=concurrent.futures.FIRST_COMPLETED
                        )

                        for future in done:
                            window, data = future.result()
                            # print(f"Writing data: window={window}")
                            dst.write(data, window=window)

                            dst.set_band_description(1,"COUNT HIGH CONFIDENCE WATER")
                            dst.set_band_description(2,"COUNT ALL WATER")
                            dst.set_band_description(3,"COUNT NON WATER")
                            dst.set_band_description(4,"COUNT CLOUD")
                            dst.set_band_description(5,"PERCENT HIGH CONFIDENCE WATER")

                        group = islice(windows, CHUNK)

                        for window in group:
                            futures.add(executor.submit(composite, stack_path, window))

        end_time_composite = time.time()
        print("Raster",os.path.basename(out_fl_path),"written successfully.")
        
        print_time(start_time_composite, end_time_composite, "composite")
        if os.path.exists(stack_path):
            os.remove(stack_path)
    else:
        print(os.path.basename(out_fl_path), "already exists.")
        return()
    
    end_time_tile = time.time()
    print_time(start_time_tile, end_time_tile, "tile")

    # import sys
    # sys.exit()

    return()


def param_wrapper(p):
    return tileProcess(*p)



def process(fldr):
    start_fldr = time.time()

    szn = os.path.basename(fldr)[6:]
    yr = os.path.basename(os.path.dirname(fldr))
    out_path = os.path.join('../data/DSWE_SE/huc_stats_p2',yr+'_'+szn+'.csv')

    print(yr, szn, 'in process', os.getpid())

    if os.path.exists(out_path):
        print(out_path, 'already exists.')
        return

    tif_paths = glob(os.path.join(fldr,'*.tif'))

    for i in range(len(tif_paths)):
        start_tif = time.time()
        if i == 0:
            src = rasterio.open(tif_paths[i])
            shp = gpd.read_file('../data/Shapefiles/HUC08/HUC08_paper2/HUC08_paper2.shp')
            huc8_reproj = shp.to_crs(crs=src.crs.to_dict())
            # point_reproj = pt_shp.to_crs(dataset.crs.to_dict())
            del(shp)
            huc8 = huc8_reproj[['huc8','geometry']]
            del(huc8_reproj)
            huc8_df = pd.DataFrame(huc8['huc8'])
            huc8_df['total_water'] = 0
            huc8_df['total_pixels'] = 0


        else:
            src = rasterio.open(tif_paths[i])

        array = src.read(5)
        affine = src.transform

        water_array = (array > 24).astype('uint8')
        del(array)

        # water_arr = ma.masked_where(array >= 25, array)
        # del(array)
        # water_array = water_arr.mask * 1
        # water_array = water_array.astype('uint8')

        zs = zonal_stats(huc8, water_array, affine=affine, stats=['sum', 'count'], nodata=0, all_touched = False)

        del(water_array)

        zs_df = pd.DataFrame(zs)
        zs_df = zs_df.fillna(0)

        huc8_df['total_water'] += zs_df['sum']
        huc8_df['total_pixels'] += zs_df['count']
        # print(szn, os.path.basename(tif_paths[i]), 'completed')
        # print(huc8_df.head())

        src.close()
        end_tif = time.time()
        print_time(start_tif, end_tif, szn + os.path.basename(tif_paths[i]))

    huc8_df.to_csv(out_path, index=False)
    end_fldr = time.time()
    print_time(start_fldr, end_fldr, yr+szn+'FINISHED!')
    # print(yr,szn,'FINISHED!')
    return


##### 
def main():

    # untar files (if needed)
    tar_lst = ['../DSWE_SE/paper2_018_013-017_allYears', '../DSWE_SE/paper2_020-021_017_allYears', \
               '../DSWE_SE/paper2_019012_allYears', '../DSWE_SE/paper2_019017_allYears']
    
    # for fl in tar_lst:
    #     untar(fl)

    # get list of ARD tile names (all start with 0) in the study area
    tiles = pd.read_csv("../data/DSWE_SE/tiles_p2.csv")
    tiles['Unique_Tiles'] = tiles['tile'].apply(lambda x: str(x).zfill(6))

    # data_path = 'C:/Users/mdgaines/Documents/Research'
    raw_tif_path = '../data/DSWE_SE/raw_tifs' #os.path.join(data_path,'data/DSWE_SE/raw_tifs')

    # get list of all year directories in raw data folder
    raw_yr_paths = [os.path.join(raw_tif_path,x) for x in os.listdir(raw_tif_path) if '.ini' not in x]
    # loop through all raw data directories
    for fldr in raw_yr_paths:
        start_time_yr = time.time()

        yr_szn = []
        if os.path.exists(os.path.join(fldr,'Fall')):
            yr_szn.append(os.path.join(fldr,'Fall'))
        if os.path.exists(os.path.join(fldr,'Spring')):
            yr_szn.append(os.path.join(fldr,'Spring'))
        if os.path.exists(os.path.join(fldr,'Summer')):
            yr_szn.append(os.path.join(fldr,'Summer'))
        if os.path.exists(os.path.join(fldr,'Winter')):
            yr_szn.append(os.path.join(fldr,'Winter'))
        print(yr_szn)
        # loop through seasonal subfolders for the year we're looking at
        for sub_fldr in yr_szn:
            start_time_szn = time.time()

            if os.path.basename(sub_fldr) == 'Spring':
                szn_num = '0'
            elif os.path.basename(sub_fldr) == 'Summer':
                szn_num = '1'
            elif os.path.basename(sub_fldr) == 'Fall':
                szn_num = '2'
            elif os.path.basename(sub_fldr) == 'Winter':
                szn_num = '3'

            out_fldr = os.path.join('../data/DSWE_SE/processed_tifs',fldr[-4:],'tiles_'+os.path.basename(sub_fldr))

            print(out_fldr)
            if not os.path.exists(out_fldr):
                if not os.path.exists(os.path.dirname(out_fldr)):
                    os.mkdir(os.path.dirname(out_fldr))
                os.mkdir(out_fldr)

            tile_lst = tiles.Unique_Tiles

            to_proc_tile_lst = []

            for tile in tile_lst:
                out_fl_path = os.path.join(out_fldr, fldr[-4:] + szn_num + '_' + tile + '_composite.tif')
                if not os.path.exists(out_fl_path):
                    to_proc_tile_lst.append(tile)
                else:
                    print(os.path.basename(out_fl_path), "already exists.")

            params = ((t, out_fldr, sub_fldr, fldr, szn_num) for t in to_proc_tile_lst)

            # can use ProcessPoolExecutor because we are reading and writing different files in each core
            with concurrent.futures.ProcessPoolExecutor(
                max_workers=7
            ) as executor:
                executor.map(param_wrapper, params)

            print(out_fldr, "complete.")
            end_time_szn = time.time()
            print_time(start_time_szn, end_time_szn)

        print(os.path.basename(fldr), "complete.")
        end_time_yr = time.time()
        print_time(start_time_yr, end_time_yr)

        # if os.path.exists(fldr):
        #     shutil.rmtree(fldr)
        #     print(os.path.basename(fldr), "deleted.")


    # proc_tif_paths = glob('../data/DSWE_SE/processed_tifs/*/*/*.tif')

    yr_szn_lst = glob('../data/DSWE_SE/processed_tifs/*/*[!.ini]')

    # check if csv's have already been made
    for yr_szn in yr_szn_lst:
        szn = os.path.basename(yr_szn)[6:]
        yr = os.path.basename(os.path.dirname(yr_szn))
        out_path = os.path.join('../data/DSWE_SE/huc_stats_p2',yr+'_'+szn+'.csv')

        # if csv's exist, delete the folder from temp
        if os.path.exists(out_path):
            # shutil.rmtree(yr_szn)
            print("{0} {1} already processed.".format(yr, szn))
            continue

    # run zonal stats in 4 processes
    with concurrent.futures.ProcessPoolExecutor(
        max_workers=4
    ) as executor:
        executor.map(process, yr_szn_lst)
    
    print('finished', yr_szn_lst)



if __name__ == '__main__':
    main()

