import os
import numpy as np 
import pandas as pd 
import geopandas as gpd

import math

import seaborn as sns
from matplotlib import cm
from matplotlib.colors import ListedColormap
import matplotlib.pyplot as plt

from glob import glob

plt.rcParams['font.family'] = 'Times New Roman'
plt.rcParams['font.size'] = '16'

def clean_df(df, huc=False, pred=True):

    # df['HUC_SEASON'] = df[huc].astype('str') + '_' + df['SEASON']
    df.loc[df.SEASON == 'Spring', 'YR_SZN'] = df.YEAR * 100 + 0
    df.loc[df.SEASON == 'Summer', 'YR_SZN'] = df.YEAR * 100 + 25
    df.loc[df.SEASON == 'Fall', 'YR_SZN'] = df.YEAR * 100 + 50
    df.loc[df.SEASON == 'Winter', 'YR_SZN'] = df.YEAR * 100 + 75
    df = df.sort_values(by=['YR_SZN'])
    
    # if pred:
    #     df['PR_WATER'] = np.exp(df['PRED_LOG_WATER']) - 10e-6
    # else:
    #     df['OBS_WATER'] = df['PR_WATER']

    # if huc:
    #     df = df[(df[huc]==3020201)&(df['YEAR']>2005)]

    return(df)



shp = gpd.read_file('../data/Shapefiles/HUC08/HUC08_paper2/HUC08_paper2.shp')
shp = shp[['huc8','areasqkm', 'geometry']]
shp['HUC08'] = shp['huc8'].apply(lambda x : int(x))
shp['AREA'] = shp['areasqkm'].apply(lambda x : float(x))

dswe_obs_lst = glob('../data/DSWE_SE/huc_stats_p2/*.csv')
dswe_06_21_fl_lst = [fl for fl in dswe_obs_lst if int(os.path.basename(fl)[0:4]) in range(2006, 2019)]
dswe_06_21_fl_lst.sort()

df_full = pd.DataFrame()

for fl in dswe_06_21_fl_lst:
    yr = int(os.path.basename(fl)[0:4])
    szn = os.path.basename(fl).split('.')[0][5:]
    df = pd.read_csv(fl, usecols=[0,1])
    df['YEAR'] = yr
    df['SEASON'] = szn

    df_full = df_full.append(df)

df_full = df_full.merge(shp[['HUC08', 'AREA']], left_on='huc8', right_on='HUC08')
df_full['obs_water_sqkm'] = df_full['total_water'] * 0.0009
df_full['OBS_PR_WATER'] = df_full['obs_water_sqkm'] / df_full['AREA']

df_full = clean_df(df_full)


# Set up for loop variables
scn_lst = ['A1B', 'A2', 'B1', 'B2']
rcp_lst = ['RCP45', 'RCP85']
scn_rcp_lst = [(scn, rcp) for scn in scn_lst for rcp in rcp_lst]

overall_min = 0
overall_max = 0

min_max_dict = {}

for scn, rcp in scn_rcp_lst:
    print(f'{scn} {rcp}')
    inpath_merf = f'../data/FutureData/GCM_FORESCE_CSVs/HUC_CI/MULTIMODEL_{rcp}_{scn}_MC_HUC_CI95.csv'
    merf_df = pd.read_csv(inpath_merf, index_col=0)
    merf_df = merf_df.loc[merf_df['YR_SZN'] < 201900]

    comp_df = df_full[['HUC08', 'OBS_PR_WATER', 'YR_SZN', 'YEAR', 'SEASON']].merge(merf_df[['HUC08', 'YR_SZN', 'MEAN']], on=['HUC08','YR_SZN'])
    comp_df['MEAN_OBS_DIFF'] = comp_df['MEAN'] - comp_df['OBS_PR_WATER']
    comp_df['PER_ERROR'] = (comp_df['OBS_PR_WATER'] - comp_df['MEAN']) / comp_df['OBS_PR_WATER'] * 100

    # SEASONAL MPE
    comp_df.boxplot(column='PER_ERROR', by='SEASON', figsize=(12,10))

    comp_shp = shp.merge(comp_df.groupby('HUC08').sum()['PER_ERROR']/52 , on='HUC08')

    # SEASONAL-HUC MPE
    plt.figure(figsize=(15, 5))

    for szn in ['Spring', 'Summer', 'Fall', 'Winter', 'all']:
        if szn == 'all':
            all_min = comp_shp['PER_ERROR'].min()
            all_max = comp_shp['PER_ERROR'].max()
            
            if all_min < overall_min:
                min_max_dict['overall_min'] = (all_min, scn+'_'+rcp)
            if all_max > overall_max:
                min_max_dict['overall_max'] = (all_max, scn+'_'+rcp)

            # comp_shp.plot('PER_ERROR', legend=True, title=f'{scn}_{rcp}')
            print(f"All\n\tmin: {all_min}\n\tmed: {comp_shp['PER_ERROR'].median()}\n\tmax: {all_max}\n")
        else:
            # create subplot axes in a 3x3 grid
            dct = {'Spring': 0, 'Summer': 1, 'Fall': 2, 'Winter': 3}
            ax = plt.subplot(1, 4, dct[szn] + 1) # nrows, ncols, axes position
            # plot the continent on these axes
            shp.merge(comp_df.loc[comp_df['SEASON']==szn].groupby('HUC08').sum()['PER_ERROR']/13 , on='HUC08').plot('PER_ERROR', legend=True, ax=ax)
            # set the title
            ax.set_title(szn)
            # set the aspect
            # adjustable datalim ensure that the plots have the same axes size
            ax.set_aspect('equal', adjustable='datalim')

            # shp.merge(comp_df.loc[comp_df['SEASON']==szn].groupby('HUC08').sum()['PER_ERROR']/13 , on='HUC08').plot('PER_ERROR', legend=True)
            print(f"{szn}\n\tmin: {(comp_df.loc[comp_df['SEASON']==szn].groupby('HUC08').sum()['PER_ERROR']/13).min()}\
                  \n\tmed: {(comp_df.loc[comp_df['SEASON']==szn].groupby('HUC08').sum()['PER_ERROR']/13).median()}\
                  \n\tmax: {(comp_df.loc[comp_df['SEASON']==szn].groupby('HUC08').sum()['PER_ERROR']/13).max()}")

    plt.tight_layout()
    plt.show()

# MERF projections are overestimating a lot 
# but they are based on projected climate and 
# LCLU data. We should also look at the mean
# percent error we get for our input data
# during this time (max temp, precip, INTS, FORE, AGRI)
# for each proj as compared to the observed
# This means the MULTIMODEL MEAN of each of these variables.

# We should check this because when training/testing the MERF 
# predictions from the training/testing with OBS data, the
# model is very accurate. Plot this too?


def mk_climate_lclu_multimodel_csv():

    return


def uncenter_unstandardize(df, obs_df, var):

    df[var] = df[var] * np.std(obs_df[var]) + np.mean(obs_df[var])

    return(df)

def unnormalize_climate_vars(df=pd.DataFrame, var=str):

    if var == 'MAX_TMP':
        var2 = 'MAX_TEMP'
    else:
        var2 = var
    df_new = pd.DataFrame()

    for szn in ['SPRING', 'SUMMER', 'FALL', 'WINTER']:

        avgStd_df = pd.read_csv(f'../data/ClimateData/GRIDMET_AVG_STDEV/1979_2008_{szn}_{var2}_AVG_STDV.csv', index_col=0)
        avgStd_df = avgStd_df.sort_values('huc8')

        temp_df = df.loc[df['SEASON']==szn[0]+szn[1:].lower()].merge(avgStd_df, left_on='HUC08', right_on='huc8')

        temp_df[var] = temp_df[var] * temp_df['std'] + temp_df['mean']

        df_new = pd.concat([df_new, temp_df.iloc[:,:-3]] , ignore_index=True)


    return(df_new)



######## main():

shp = gpd.read_file('../data/Shapefiles/HUC08/HUC08_paper2/HUC08_paper2.shp')
shp = shp[['huc8','areasqkm']]
shp['HUC08'] = shp['huc8'].apply(lambda x : int(x))
shp['AREA'] = shp['areasqkm'].apply(lambda x : float(x))

dswe = pd.read_csv('../data/all_data_0118_p2.csv', index_col=0)

rcp_lst = ['RCP45', 'RCP85']
foresce_lst = ['A1B', 'A2', 'B1', 'B2']
gcm_lst = ['GFDL', 'HadGEM2', 'IPSL', 'MIROC5', 'NorESM1']
var_lst = ['PRECIP', 'MAX_TMP', 'PR_AG', 'PR_INT', 'PR_NAT']


for rcp in rcp_lst:
    for foresce in foresce_lst:
        outpath = f'../data/FutureData/GCM_FORESCE_CSVs/VAR_MEANS/MULTIMODEL_{rcp}_{foresce}_VAR_MEAN.csv'
        if os.path.exists(outpath):
            print(f'{os.path.basename(outpath)} exists.')
            continue
        # full_MC_ci_info_dict = {}
        # gcm_df = pd.DataFrame()
        for gcm in gcm_lst:
            print(f'\nstarting {gcm}')

            full_MC_path_lst = glob(f'../data/FutureData/GCM_FORESCE_CSVs/{gcm}/{gcm}_{rcp}_{foresce}/{gcm}_{rcp}_{foresce}_*.csv')
            full_MC_path_lst.sort()
            count = 0
            for fl in full_MC_path_lst:
                full_MC_df = pd.read_csv(fl)#,  usecols=['HUC08', 'YEAR', 'SEASON', var])

                for var in var_lst:
                    # print(f'\nstarting {var}')

                    szn_MC_df = full_MC_df[['HUC08', 'YEAR', 'SEASON', var]].loc[full_MC_df['YEAR'] <= 2018]

                    szn_MC_df = uncenter_unstandardize(szn_MC_df, dswe, var)
                    if var in ['PRECIP', 'MAX_TMP']:
                        szn_MC_df = unnormalize_climate_vars(szn_MC_df, var)
                    
                    if count == 0:
                        if var == 'PRECIP':
                            precip_df = szn_MC_df
                        elif var == 'MAX_TMP':
                            maxtemp_df = szn_MC_df
                        elif var == 'PR_AG':
                            ag_df = szn_MC_df
                        elif var == 'PR_INT':
                            int_df = szn_MC_df                    
                        elif var == 'PR_NAT':
                            nat_df = szn_MC_df
                    else:
                        if var == 'PRECIP':
                            precip_df = precip_df.merge(szn_MC_df, on=['YEAR','SEASON', 'HUC08'], suffixes=[None, f'_{count}'])
                            precip_df['RUNNING_SUM_PRECIP'] = precip_df.iloc[:,3:].sum(axis = 1)
                            precip_df = precip_df[['HUC08', 'YEAR', 'SEASON', 'RUNNING_SUM_PRECIP']]
                        elif var == 'MAX_TMP':
                            maxtemp_df = maxtemp_df.merge(szn_MC_df, on=['YEAR','SEASON', 'HUC08'], suffixes=[None, f'_{count}'])
                            maxtemp_df['RUNNING_SUM_MAX_TMP'] = maxtemp_df.iloc[:,3:].sum(axis = 1)
                            maxtemp_df = maxtemp_df[['HUC08', 'YEAR', 'SEASON', 'RUNNING_SUM_MAX_TMP']]
                        elif var == 'PR_AG':
                            ag_df = ag_df.merge(szn_MC_df, on=['YEAR','SEASON', 'HUC08'], suffixes=[None, f'_{count}'])
                            ag_df['RUNNING_SUM_PR_AG'] = ag_df.iloc[:,3:].sum(axis = 1)
                            ag_df = ag_df[['HUC08', 'YEAR', 'SEASON', 'RUNNING_SUM_PR_AG']]
                        elif var == 'PR_INT':
                            int_df = int_df.merge(szn_MC_df, on=['YEAR','SEASON', 'HUC08'], suffixes=[None, f'_{count}'])                    
                            int_df['RUNNING_SUM_PR_INT'] = int_df.iloc[:,3:].sum(axis = 1)
                            int_df = int_df[['HUC08', 'YEAR', 'SEASON', 'RUNNING_SUM_PR_INT']]
                        elif var == 'PR_NAT':
                            nat_df = nat_df.merge(szn_MC_df, on=['YEAR','SEASON', 'HUC08'], suffixes=[None, f'_{count}'])                    
                            nat_df['RUNNING_SUM_PR_NAT'] = nat_df.iloc[:,3:].sum(axis = 1)
                            nat_df = nat_df[['HUC08', 'YEAR', 'SEASON', 'RUNNING_SUM_PR_NAT']]

                if count % 50 == 0:
                    print(f'{count} files completed.')
                count += 1


            if gcm == 'GFDL':
                multi_df = precip_df.merge(maxtemp_df, on=['YEAR','SEASON', 'HUC08'])
            else:
                multi_df = multi_df.merge(precip_df, on=['YEAR','SEASON', 'HUC08'])
                multi_df = multi_df.merge(maxtemp_df, on=['YEAR','SEASON', 'HUC08'])
                
            multi_df = multi_df.merge(ag_df, on=['YEAR','SEASON', 'HUC08'])
            multi_df = multi_df.merge(int_df, on=['YEAR','SEASON', 'HUC08'])
            multi_df = multi_df.merge(nat_df, on=['YEAR','SEASON', 'HUC08'])
            # else:

            multi_df[['RUNNING_SUM_PRECIP',
                    'RUNNING_SUM_MAX_TMP',
                    'RUNNING_SUM_PR_AG',
                    'RUNNING_SUM_PR_INT',
                    'RUNNING_SUM_PR_NAT']] = multi_df[['RUNNING_SUM_PRECIP',
                                                        'RUNNING_SUM_MAX_TMP',
                                                        'RUNNING_SUM_PR_AG',
                                                        'RUNNING_SUM_PR_INT',
                                                        'RUNNING_SUM_PR_NAT']].apply(lambda x: x / 1000, axis=1)
            
            multi_df = multi_df.rename(
                columns={'RUNNING_SUM_PRECIP': f'MEAN_{gcm}_PRECIP',
                            'RUNNING_SUM_MAX_TMP': f'MEAN_{gcm}_MAX_TMP',
                            'RUNNING_SUM_PR_AG': f'MEAN_{gcm}_PR_AG',
                            'RUNNING_SUM_PR_INT': f'MEAN_{gcm}_PR_INT',
                             'RUNNING_SUM_PR_NAT': f'MEAN_{gcm}_PR_NAT'})

        standard_cols = ['HUC08', 'YEAR', 'SEASON']
        for var in var_lst:
            var_cols = [col for col in multi_df.columns if var in col]
            multi_df[f'MULTIMODEL_MEAN_{var}'] = multi_df[var_cols].mean(axis=1)

        precip_cols = [col for col in multi_df.columns if 'PRECIP' in col]
        maxtemp_cols = [col for col in multi_df.columns if 'MAX_TMP' in col]
        ag_cols = [col for col in multi_df.columns if 'PR_AG' in col]
        int_cols = [col for col in multi_df.columns if 'PR_INT' in col]
        nat_cols = [col for col in multi_df.columns if 'PR_NAT' in col]

        multi_df = multi_df[standard_cols + precip_cols + maxtemp_cols + ag_cols + int_cols + nat_cols]

        multi_df.to_csv(outpath)



