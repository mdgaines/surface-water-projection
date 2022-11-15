

from importlib.resources import path
import os
import pandas as pd
import numpy as np
from glob import glob

##########################################################
##### Get 30+ year avgs and stdevs into seaonal csvs #####
##########################################################

sznAvgStdev_path_lst = glob('../data/ClimateData/GRIDMET_AVG_STDEV/*HUC*.csv')

sznAvgStdev_path_lst.sort()

for i in range(0, 16):
    path_element_lst = sznAvgStdev_path_lst[i].split('_')
    szn = path_element_lst[-2]
    climate_var = path_element_lst[-1][:-4]

    df = pd.read_csv(sznAvgStdev_path_lst[i])

    if i % 4 == 0:
        new_df = pd.DataFrame(df[['huc8','mean']])

        if szn == 'Sp':
            szn = 'SPRING'
        elif szn == 'Su':
            szn = 'SUMMER'
        elif szn == 'Fa':
            szn = 'FALL'
        elif szn == 'Wi':
            szn = 'WINTER'
        else:
            print('Something went wrong. szn = ', szn)
        
        new_df['SEASON'] = szn

        print('starting new season:', szn)

    else:
        new_df = new_df.merge(df[['huc8','mean']], on='huc8')

    new_df.rename(columns={'mean':climate_var.upper()},inplace=True)

    if (i+1) % 4 == 0:
        outpath = '../data/ClimateData/GRIDMET_AVG_STDEV/GRIDMET_'+np.unique(new_df['SEASON'])[0]+'_AvgStdev.csv'
        
        if os.path.exists(outpath):
            print(outpath, 'exists.')
            continue

        print('saving seasaon:', np.unique(new_df['SEASON']))
        new_df.to_csv(outpath)



##########################################################
#####  Get standardized seaonal anomalies into csvs  #####
##########################################################

spring_long_avgStdev = pd.read_csv(glob('../data/ClimateData/GRIDMET_AVG_STDEV/*SPRING*.csv')[0], index_col=0)
summer_long_avgStdev = pd.read_csv(glob('../data/ClimateData/GRIDMET_AVG_STDEV/*SUMMER*.csv')[0], index_col=0)
fall_long_avgStdev = pd.read_csv(glob('../data/ClimateData/GRIDMET_AVG_STDEV/*FALL*.csv')[0], index_col=0)
winter_long_avgStdev = pd.read_csv(glob('../data/ClimateData/GRIDMET_AVG_STDEV/*WINTER*.csv')[0], index_col=0)

rcp_sznAvg_path_lst = glob('../data/ClimateData/GFDL-ESM2M_macav2livneh/RCP*/zonal_avg/*.csv')
# rcp85_sznAvg_path_lst = glob('../data/ClimateData/GFDL-ESM2M_macav2livneh/RCP85/zonal_avg/*.csv')

szn_denoms = {'SPRING':((31+30+31)/3),\
              'SUMMER':((30+31+31)/3),\
              'FALL':((30+31+30)/3),\
              'WINTER':((31+31+28)/3)}

for i in range(len(rcp_sznAvg_path_lst)):

    rcp_sznAvg_df = pd.read_csv(rcp_sznAvg_path_lst[i])

    path_element_lst = rcp_sznAvg_path_lst[i].split('_')
    szn = path_element_lst[-2]
    yr = path_element_lst[-3]
    climate_var = path_element_lst[-4]

    outpath = os.path.join(os.path.dirname(os.path.dirname(rcp_sznAvg_path_lst[i])),'zonal_anom', \
        os.path.basename(rcp_sznAvg_path_lst[i])[:-7]+'ANOM.csv')

    if os.path.exists(outpath):
        print(os.path.basename(outpath), 'exists.')
        continue

    if climate_var == 'PRECIP':
        cl_avg, cl_stdev = 'AVGPR', 'STDEVPR'
        szn_avg = 'AVG_PRECIP'
        denom = szn_denoms[szn]
    elif climate_var == 'TEMP':
        cl_avg, cl_stdev = 'AVGMXTEMP', 'STDEVMXTEMP'
        szn_avg = 'AVG_MAX_TEMP'
        denom = 1

    if szn == 'SPRING':
        new_df = rcp_sznAvg_df[[szn_avg,'huc8','YEAR','SEASON']].merge(\
            spring_long_avgStdev[['huc8', cl_avg, cl_stdev]], on='huc8')

        new_df[szn_avg] = new_df[szn_avg] / denom

        new_df[climate_var+'_ANOM'] = (new_df[szn_avg] - new_df[cl_avg]) / new_df[cl_stdev]

    elif szn == 'SUMMER':
        new_df = rcp_sznAvg_df[[szn_avg,'huc8','YEAR','SEASON']].merge(\
            summer_long_avgStdev[['huc8', cl_avg, cl_stdev]], on='huc8')

        new_df[szn_avg] = new_df[szn_avg] / denom

        new_df[climate_var+'_ANOM'] = (new_df[szn_avg] - new_df[cl_avg]) / new_df[cl_stdev]

    elif szn == 'FALL':
        new_df = rcp_sznAvg_df[[szn_avg,'huc8','YEAR','SEASON']].merge(\
            fall_long_avgStdev[['huc8', cl_avg, cl_stdev]], on='huc8')

        new_df[szn_avg] = new_df[szn_avg] / denom

        new_df[climate_var+'_ANOM'] = (new_df[szn_avg] - new_df[cl_avg]) / new_df[cl_stdev]

    elif szn == 'WINTER':
        new_df = rcp_sznAvg_df[[szn_avg,'huc8','YEAR','SEASON']].merge(\
            winter_long_avgStdev[['huc8', cl_avg, cl_stdev]], on='huc8')

        new_df[szn_avg] = new_df[szn_avg] / denom

        new_df[climate_var+'_ANOM'] = (new_df[szn_avg] - new_df[cl_avg]) / new_df[cl_stdev]

    else:
        print('Something went wrong. szn = ', szn)

    new_df.to_csv(outpath)

    print(os.path.basename(outpath), 'saved.')

    del(new_df)
