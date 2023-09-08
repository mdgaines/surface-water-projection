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