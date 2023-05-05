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



scn_ci_dict = {}
scn_lst = ['RCP45_B1', 'RCP85_B2', 'RCP85_A2']
for scn in scn_lst:
    multi_output_ci = f'../data/FutureData/GCM_FORESCE_CSVs/HUC_CI/MULTIMODEL_{scn}_MC_HUC_CI95.csv'
    ci_df = pd.read_csv(multi_output_ci, index_col=0)
    scn_ci_dict[scn] = ci_df


huc_lst = [3130001, 3020201, 3100205, 6010201, 8090203]
huc_names = ['Upper Chattahoochee', 'Upper Neuse', 'Hillsborough', 'Watts Bar Lake', 'Eastern Louisiana Coastal']
# for i in range(len(huc_lst)):
#     plot_gcm_ci(scn_ci_dict, huc_lst[i], huc_names[i])


df = pd.read_csv('../data/WASSI/ANNUALWaSSI/ANNUALWaSSI_US_GFDL-ESM2M_RCP45_917f5706936a02c1183cad2ea9fad887.TXT', usecols=[0,1,2])

df = df[df['CELL']==3130001]

huc = 3130001
obs_df = pd.read_csv('../data/all_data_0118_p2.csv', index_col=0)
obs_df = clean_df(obs_df, pred=False)
obs_huc_df = obs_df[obs_df['HUC08'] == huc].sort_values('YR_SZN')
obs_huc_df_yrAvg = obs_huc_df.groupby('YEAR').mean()

# create figure and axis objects with subplots()
fig,ax = plt.subplots()
# make a plot
ax.plot(df.iloc[40:60].YEAR,
        df.iloc[40:60].SWS_MGD,
        color="red", 
        marker="o")
# set x-axis label
ax.set_xlabel("year", fontsize = 14)
# set y-axis label
ax.set_ylabel("SWS_MGD",
              color="red",
              fontsize=14)

# twin object for two different y-axis on the sample plot
ax2=ax.twinx()
# make a plot with different y-axis using second axis object
ax2.plot(obs_huc_df_yrAvg.index, obs_huc_df_yrAvg.OBS_WATER,color="blue",marker="o")
ax2.set_ylabel("OBS_WATER",color="blue",fontsize=14)
plt.show()


huc = 3130001
obs_df = pd.read_csv('../data/all_data_0118_p2.csv', index_col=0)
obs_df = clean_df(obs_df, pred=False)
obs_huc_df = obs_df[obs_df['HUC08'] == huc].sort_values('YR_SZN')
obs_huc_df_yrAvg = obs_huc_df.groupby('YEAR').mean()

key = 'RCP45_B1'
if key == 'RCP45_B1':
    col = '#005a32' # green
    lab = 'RCP 4.5 - B1'
elif key == 'RCP85_B2':
    col = '#084594' # blue
    lab = 'RCP 8.5 - B2'
elif key == 'RCP85_A2':
    col = '#91003f' # pink
    lab = 'RCP 8.5 - A2'

CI_df_full = ci_dict[key]
CI_df = CI_df_full[CI_df_full['HUC08'] == huc].sort_values('YR_SZN')
CI_df['YEAR'] = CI_df['YR_SZN'] // 100
CI_df = CI_df.groupby('YEAR').mean()

# create figure and axis objects with subplots()
fig,ax = plt.subplots()
# make a plot
ax.plot(df.iloc[40:60].YEAR,
        df.iloc[40:60].SWS_MGD,
        color="red", 
        marker="o")
# set x-axis label
ax.set_xlabel("year", fontsize = 14)
# set y-axis label
ax.set_ylabel("SWS_MGD",
              color="red",
              fontsize=14)

# twin object for two different y-axis on the sample plot
ax2=ax.twinx()
# make a plot with different y-axis using second axis object
ax2.plot(obs_huc_df_yrAvg.index, obs_huc_df_yrAvg.OBS_WATER,color="black",marker="o")
ax2.set_ylabel("OBS_WATER",color="black",fontsize=14)

ax2.plot(CI_df.iloc[0:15].index, np.asarray(CI_df.iloc[0:15]['MEAN']), color = col, linewidth=1 , label=lab, marker="o")
ax2.fill_between(CI_df.iloc[0:15].index, CI_df.iloc[0:15]['LOWER_95_CI'], CI_df.iloc[0:15]['UPPER_95_CI'], color = col, alpha=.3, label='95% CI')

plt.show()


huc = 3130001
obs_df = pd.read_csv('../data/all_data_0118_p2.csv', index_col=0)
obs_df = clean_df(obs_df, pred=False)
obs_huc_df = obs_df[obs_df['HUC08'] == huc].sort_values('YR_SZN')
obs_huc_df_yrAvg = obs_huc_df.groupby('YEAR').sum()

key = 'RCP45_B1'
if key == 'RCP45_B1':
    col = '#005a32' # green
    lab = 'RCP 4.5 - B1'
elif key == 'RCP85_B2':
    col = '#084594' # blue
    lab = 'RCP 8.5 - B2'
elif key == 'RCP85_A2':
    col = '#91003f' # pink
    lab = 'RCP 8.5 - A2'

CI_df_full = ci_dict[key]
CI_df = CI_df_full[CI_df_full['HUC08'] == huc].sort_values('YR_SZN')
CI_df['YEAR'] = CI_df['YR_SZN'] // 100
CI_df = CI_df.groupby('YEAR').sum()

# create figure and axis objects with subplots()
fig,ax = plt.subplots()
# make a plot
ax.plot(df.iloc[45:].YEAR,
        df.iloc[45:].SWS_MGD,
        color="red", 
        marker="o")
# set x-axis label
ax.set_xlabel("year", fontsize = 14)
# set y-axis label
ax.set_ylabel("SWS_MGD",
              color="red",
              fontsize=14)

# twin object for two different y-axis on the sample plot
ax2=ax.twinx()
# make a plot with different y-axis using second axis object
# ax2.plot(obs_huc_df_yrAvg.index, obs_huc_df_yrAvg.OBS_WATER,color="black",marker="o")
# ax2.set_ylabel("OBS_WATER",color="black",fontsize=14)

ax2.plot(CI_df.index, np.asarray(CI_df['MEAN']), color = col, linewidth=1 , label=lab)#, marker="o")
ax2.fill_between(CI_df.index, CI_df['LOWER_95_CI'], CI_df['UPPER_95_CI'], color = col, alpha=.3, label='95% CI')

plt.show()