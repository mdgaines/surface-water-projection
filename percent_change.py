import os
import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt
import geopandas as gpd

import pymannkendall as mk

def clean_df(df, huc=False, pred=True):
    
    # df['HUC_SEASON'] = df[huc].astype('str') + '_' + df['SEASON']
    df.loc[df.SEASON == 'Spring', 'YR_SZN'] = df.loc[df.SEASON == 'Spring', 'YEAR'] * 100 + 0
    df.loc[df.SEASON == 'Summer', 'YR_SZN'] = df.loc[df.SEASON == 'Summer', 'YEAR'] * 100 + 25
    df.loc[df.SEASON == 'Fall', 'YR_SZN'] = df.loc[df.SEASON == 'Fall', 'YEAR'] * 100 + 50
    df.loc[df.SEASON == 'Winter', 'YR_SZN'] = df.loc[df.SEASON == 'Winter', 'YEAR'] * 100 + 75
    df = df.sort_values(by=['YR_SZN'])
    
    if pred:
        df['PR_WATER'] = np.exp(df['PRED_LOG_WATER']) - 10e-6
    else:
        df['OBS_WATER'] = df['PR_WATER']

    if huc:
        df = df[(df[huc]==3020201)&(df['YEAR']>2005)]

    return(df)


def per_change_info(obs_huc_df:pd.DataFrame, CI_df:pd.DataFrame):

    lst = [i for i in range(72) if i%4 == 0]
    mean_obs_huc_lst = [['Spring', obs_huc_df.iloc[lst][['OBS_WATER']].mean().values[0]]]
    lst = [i+1 for i in range(72) if i%4 == 0]
    mean_obs_huc_lst.append(['Summer', obs_huc_df.iloc[lst][['OBS_WATER']].mean().values[0]])
    lst = [i+2 for i in range(72) if i%4 == 0]
    mean_obs_huc_lst.append(['Fall', obs_huc_df.iloc[lst][['OBS_WATER']].mean().values[0]])
    lst = [i+3 for i in range(72) if i%4 == 0]
    mean_obs_huc_lst.append(['Winter', obs_huc_df.iloc[lst][['OBS_WATER']].mean().values[0]])

    mean_obs_huc_df = pd.DataFrame(mean_obs_huc_lst, columns=['SEASON','MEAN_OBS_WATER'])

    lst = [i for i in range(375) if i % 4 == 0]
    # spring_CI_df = (CI_df.iloc[lst][['MEAN', 'LOWER_95_CI', 'UPPER_95_CI']] - mean_obs_huc_df.iloc[0].MEAN_OBS_WATER) * 100
    spring_CI_df = (CI_df.iloc[lst][['MEAN', '90th', '10th']] - mean_obs_huc_df.iloc[0].MEAN_OBS_WATER) / mean_obs_huc_df.iloc[0].MEAN_OBS_WATER * 100
    lst = [i + 1 for i in range(375) if i % 4 == 0]
    # summer_CI_df = (CI_df.iloc[lst][['MEAN', 'LOWER_95_CI', 'UPPER_95_CI']] - mean_obs_huc_df.iloc[1].MEAN_OBS_WATER) / mean_obs_huc_df.iloc[1].MEAN_OBS_WATER * 100
    summer_CI_df = (CI_df.iloc[lst][['MEAN', '90th', '10th']] - mean_obs_huc_df.iloc[1].MEAN_OBS_WATER) / mean_obs_huc_df.iloc[1].MEAN_OBS_WATER * 100
    lst = [i + 2 for i in range(375) if i % 4 == 0]
    # fall_CI_df = (CI_df.iloc[lst][['MEAN', 'LOWER_95_CI', 'UPPER_95_CI']] - mean_obs_huc_df.iloc[2].MEAN_OBS_WATER) / mean_obs_huc_df.iloc[2].MEAN_OBS_WATER * 100
    fall_CI_df = (CI_df.iloc[lst][['MEAN', '90th', '10th']] - mean_obs_huc_df.iloc[2].MEAN_OBS_WATER) / mean_obs_huc_df.iloc[2].MEAN_OBS_WATER * 100
    lst = [i + 3 for i in range(375) if i % 4 == 0][:-1]
    # winter_CI_df = (CI_df.iloc[lst][['MEAN', 'LOWER_95_CI', 'UPPER_95_CI']] - mean_obs_huc_df.iloc[3].MEAN_OBS_WATER) / mean_obs_huc_df.iloc[3].MEAN_OBS_WATER * 100
    winter_CI_df = (CI_df.iloc[lst][['MEAN', '90th', '10th']] - mean_obs_huc_df.iloc[3].MEAN_OBS_WATER) / mean_obs_huc_df.iloc[3].MEAN_OBS_WATER * 100

    CI_df = pd.concat([spring_CI_df, summer_CI_df, fall_CI_df, winter_CI_df])
    CI_df = CI_df.sort_index()

    return CI_df


def get_plus_minus(df):

    plus = (df['90th'] - df['MEAN']) * 100
    minus = (df['MEAN'] - df['10th']) * 100

    print(f'mean: {round(df.MEAN * 100, 2)}, plus: {round(plus, 2)}, minus: {round(minus, 2)}')


scn_ci_dict = {}
scn_lst = ['RCP45_B1', 'RCP85_B1', \
           'RCP45_B2', 'RCP85_B2', \
           'RCP45_A1B', 'RCP85_A1B', \
           'RCP45_A2', 'RCP85_A2']
for scn in scn_lst:
    multi_output_ci = f'../data/FutureData/GCM_FORESCE_CSVs/StudyAreaCI/MULTIMODEL_{scn}_MC_CI95.csv'
    ci_df = pd.read_csv(multi_output_ci, index_col=0)
    scn_ci_dict[scn] = ci_df / 1121571.58 # study area in kmsq
    scn_ci_dict[scn] = scn_ci_dict[scn].reset_index()


# huc02 = gpd.read_file('../data/Shapefiles/HUC02/HUC02_clean_paper2/HUC02_clean_paper2.shp')

##########################################
######   Comp to Spring 2018 #############
##########################################

shp = gpd.read_file('../data/Shapefiles/HUC08/HUC08_paper2/HUC08_paper2.shp')
shp = shp[['huc8','areasqkm', 'geometry']]
shp['HUC08'] = shp['huc8'].apply(lambda x : int(x))
shp['AREA'] = shp['areasqkm'].apply(lambda x : float(x))

obs_df = pd.read_csv('../data/all_data_0118_p2.csv', index_col=0)
obs_df = clean_df(obs_df, pred=False)
obs_2018_spring = obs_df[(obs_df["YR_SZN"]==201800)]

shp = shp.merge(obs_2018_spring[["HUC08", "OBS_WATER"]], on="HUC08")

shp['OBS_WATER_AREA'] = shp['AREA'] * shp['OBS_WATER']

obs_2018_spring = shp['OBS_WATER_AREA'].sum() / shp['AREA'].sum()

################################################
######  Comp to 2001-2018 Seasonal Avgs  #######
################################################

shp2 = gpd.read_file('../data/Shapefiles/HUC08/HUC08_paper2/HUC08_paper2.shp')
shp2 = shp2[['huc8','areasqkm', 'geometry']]
shp2['HUC08'] = shp2['huc8'].apply(lambda x : int(x))
shp2['AREA'] = shp2['areasqkm'].apply(lambda x : float(x))

obs_df = pd.read_csv('../data/all_data_0118_p2.csv', index_col=0)
obs_df = clean_df(obs_df, pred=False)
obs_2018_spring = obs_df[(obs_df["YR_SZN"]==201800)]

shp2_spring2018 = shp2.merge(obs_2018_spring[["HUC08", "SEASON", "YEAR", "OBS_WATER"]], on="HUC08")
shp2_spring2018['OBS_WATER_AREA'] = shp2_spring2018['AREA'] * shp2_spring2018['OBS_WATER']
obs_2018_spring = shp2_spring2018['OBS_WATER_AREA'].sum() / shp2_spring2018['AREA'].sum()


shp2 = shp2.merge(obs_df[["HUC08", "SEASON", "YEAR", "OBS_WATER"]], on="HUC08")

shp2['OBS_WATER_AREA'] = shp2['AREA'] * shp2['OBS_WATER']

shp2_spring = shp2[shp2['SEASON']=='Spring']

studyArea_obs_0118_avg = (shp2_spring.groupby('YEAR').sum()['OBS_WATER_AREA'] / shp2_spring.groupby('YEAR').sum()['AREA']).mean()


####################################################################
### Percent change (study area from spring 2018) in each scenario for spring 2099 ###
####################################################################

pdiff_dict = {}

for scn in scn_lst:
    for yr in [204000, 207000, 209900]:

        df = scn_ci_dict[scn]

        percent_diff = (df[df['YR_SZN'] == yr].iloc[0]['MEAN'] - obs_2018_spring) / obs_2018_spring * 100
        percent_diff_90 = (df[df['YR_SZN'] == yr].iloc[0]['90th'] - obs_2018_spring) / obs_2018_spring * 100
        percent_diff_10 = (df[df['YR_SZN'] == yr].iloc[0]['10th'] - obs_2018_spring) / obs_2018_spring * 100


        pdiff_dict[f'{scn} - {yr}'] = [yr, percent_diff, percent_diff_90, percent_diff_10]

pdiff_df = pd.DataFrame.from_dict(pdiff_dict, orient='index')
pdiff_df = pdiff_df.rename(columns={0:'YEAR', 1:'MEAN', 2:'90th', 3:'10th'})

################################


#######################################################################################################
### Percent change (study area from spring 2001-2018) avg in focal scns for spring 2040, 2070, 2099 ###
#######################################################################################################

pdiff_0118_dict = {}

for scn in scn_lst:
    for yr in [204000, 207000, 209900]:

        df = scn_ci_dict[scn]

        percent_diff = (df[df['YR_SZN'] == yr].iloc[0]['MEAN'] - studyArea_obs_0118_avg) / studyArea_obs_0118_avg * 100
        percent_diff_90 = (df[df['YR_SZN'] == yr].iloc[0]['90th'] - studyArea_obs_0118_avg) / studyArea_obs_0118_avg * 100
        percent_diff_10 = (df[df['YR_SZN'] == yr].iloc[0]['10th'] - studyArea_obs_0118_avg) / studyArea_obs_0118_avg * 100


        pdiff_0118_dict[f'{scn} - {yr}'] = [yr, percent_diff, percent_diff_90, percent_diff_10]

pdiff_0118_df = pd.DataFrame.from_dict(pdiff_0118_dict, orient='index')
pdiff_0118_df = pdiff_0118_df.rename(columns={0:'YEAR', 1:'MEAN', 2:'90th', 3:'10th'})

for index in pdiff_0118_df.index[0:3]:
    print(f'\n{index}')
    get_plus_minus(pdiff_0118_df.loc[index])

################################


################################
### MK trend for study area ###
################################

scn_mk_dict = {}
for scn in scn_lst:

    df = scn_ci_dict[scn]

    # mk_res_lst = [0] * len(np.unique(df.HUC08))
    # huc_lst = df['HUC08'].unique()

    # for i in range(len(mk_res_lst)):
        
    df = df.sort_values('YR_SZN')
    df = df.set_index('YR_SZN')

    res = mk.seasonal_test(df['MEAN'], period=4)

    mk_res_lst = [res.trend, res.h, res.p, res.z, res.Tau, res.s, res.var_s, res.slope, res.intercept]

        # mk_savepng(np.asarray(temp_df['PR_WATER']), res, huc_lst[i])

    # mk_res_df.to_csv(outpath)

    scn_mk_dict[scn] = mk_res_lst

pd.DataFrame(scn_mk_dict)

################################




scn_huc_ci_dict = {}
scn_lst = ['RCP45_B1', 'RCP85_B1', \
           'RCP45_B2', 'RCP85_B2', \
           'RCP45_A1B', 'RCP85_A1B', \
           'RCP45_A2', 'RCP85_A2']
for scn in scn_lst:
    multi_output_ci = f'../data/FutureData/GCM_FORESCE_CSVs/HUC_CI/MULTIMODEL_{scn}_MC_HUC_CI95.csv'
    ci_df = pd.read_csv(multi_output_ci, index_col=0)
    scn_huc_ci_dict[scn] = ci_df 

obs_df = pd.read_csv('../data/all_data_0118_p2.csv', index_col=0)
obs_df = clean_df(obs_df, pred=False)
obs_2018_spring = obs_df[(obs_df["YR_SZN"]==201800)]

####################################################################
### Percent change (HUCs) in each scenario for spring 2099 ###
####################################################################

huc_pdiff_dict = {}

for scn in scn_lst:

    df = scn_huc_ci_dict[scn]
    # df_2099 = df[df['YR_SZN']==209900].merge(obs_2018_spring[['HUC08','OBS_WATER']], on=['HUC08'])
    # df_2099['PER_CHANGE'] = (df_2099['MEAN'] - df_2099['OBS_WATER']) / df_2099['OBS_WATER'] * 100

    # # percent_diff = (df[df['YR_SZN']==209900].iloc[0]['MEAN'] - obs_2018_spring) / obs_2018_spring * 100

    # huc_pdiff_dict[scn] = df_2099.mean()['PER_CHANGE']

    
    for yr in [204000, 207000, 209900]:

        # df = scn_ci_dict[scn]
        df_yr = df[df['YR_SZN']==yr].merge(obs_2018_spring[['HUC08','OBS_WATER']], on=['HUC08'])

        percent_diff = (df_yr['MEAN'] - df_yr['OBS_WATER']) / df_yr['OBS_WATER'] * 100
        percent_diff_90 = (df_yr['90th'] - df_yr['OBS_WATER']) / df_yr['OBS_WATER'] * 100
        percent_diff_10 = (df_yr['10th'] - df_yr['OBS_WATER']) / df_yr['OBS_WATER'] * 100


        huc_pdiff_dict[f'{scn} - {yr}'] = [yr, percent_diff.mean(), percent_diff_90.mean(), percent_diff_10.mean()]

huc_pdiff_df = pd.DataFrame.from_dict(huc_pdiff_dict, orient='index')
huc_pdiff_df = huc_pdiff_df.rename(columns={0:'YEAR', 1:'MEAN', 2:'90th', 3:'10th'})


################################
