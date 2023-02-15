

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



############################################
############ AGU 2022 ####################
##########################################

# Get CSVs of mean and stdev of GRIDMET climate vars from 2000-2018
# this is an avg and stdev of a seasonal sum for precip and a seasonal avg for mx temp (latter is okay, i think)
# we need the annual sum for the table
szn_dict = {'Sp':'SPRING', 'Su':'SUMMER', 'Fa':'FALL', 'Wi':'WINTER'}
var_dict = {'Pr': 'PRECIP', 'maxTemp':'MAX_TEMP'}

for szn in ['Sp', 'Su', 'Fa', 'Wi']:
    for var in ['Pr', 'maxTemp']:

        outpath = f'../data/ClimateData/GRIDMET_AVG_STDEV/1979_2008_{szn_dict[szn]}_{var_dict[var]}_AVG_STDV.csv'

        if not os.path.exists(outpath):
            
            cVar_lst = glob(f'../data/ClimateData/GRIDMET_AVG_STDEV/GRIDMET_YR_AVG/*{szn}_{var}*.csv')
            cVar_lst.sort()
            cVar_lst = cVar_lst[0:30] # 0-30 is 1979-2008
            for i in range(len(cVar_lst)):
                cVar_df = pd.read_csv(cVar_lst[i], usecols=['huc8', 'mean'])
                if i == 0:
                    df = cVar_df
                else:
                    df = df.join(cVar_df.set_index('huc8'), on='huc8', lsuffix=cVar_lst[i].split('\\')[-1][:4])

            out_df = pd.DataFrame({'huc8':df.set_index('huc8').T.mean().index,'mean':df.set_index('huc8').T.mean().values, 'std':df.set_index('huc8').T.std().values})
            out_df.to_csv(outpath)



##########################################
############ Study Area Table ############
##########################################

gridmet_mnth_lst = glob('../data/ClimateData/GRIDMET_AVG_STDEV/*MNTH*.csv')

for j in range(len(gridmet_mnth_lst)):

    yr = gridmet_mnth_lst[j].split('\\')[1].split('_')[1]

    if int(yr) < 2009:
        df = pd.read_csv(gridmet_mnth_lst[j])

        lst = list(df.columns)
        lst = [i for i in lst if '-' in i]
        yr_cols = [i for i in lst if yr in i]
        df = df[yr_cols]
        lst_2 = ['-'.join([i.split('-')[0],i.split('-')[1].zfill(2)]) for i in lst if yr + '-' in i]

        col_dict = dict(zip(yr_cols,lst_2))

        df = df.rename(col_dict, axis='columns')
        # df = df[lst_2]
        df = df.reindex(sorted(df.columns), axis=1)
        df = df.T

        df = df[0].apply(lambda x: pd.Series(x.split(',')))
        df['AVG_PRECIP'] = df[0].apply(lambda x: float(x[1:]))
        df['AVG_MAX_TEMP'] = df[1].apply(lambda x: float(x[:-1]))
        df = df.drop([0,1], axis=1)

        if j == 0:
            df_1979_2008 = df
        else:
            df_1979_2008 = pd.concat([df_1979_2008, df])

mean_ann_precip_1979_2008 = df_1979_2008['AVG_PRECIP'].sum() / 30
mean_ann_mxtemp_1979_2008 = df_1979_2008['AVG_MAX_TEMP'].mean() - 273.15    # put it in C for the paper

