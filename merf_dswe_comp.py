import os
import numpy as np 
import pandas as pd 
import geopandas as gpd

import math

import seaborn as sns
import matplotlib
from matplotlib import cm
from matplotlib.colors import ListedColormap
from matplotlib.colors import BoundaryNorm
import matplotlib.pyplot as plt

from glob import glob

plt.rcParams['font.family'] = 'Times New Roman'
plt.rcParams['font.size'] = '16'

MY_PALLET = {'RCP45_B1_MPE':'#008607', 'RCP85_B1_MPE':'#005a01',
            'RCP45_B2_MPE':'#009ffa', 'RCP85_B2_MPE':'#00489e',
            'RCP45_A1B_MPE':'#ff3cfe', 'RCP85_A1B_MPE':'#65019f',
            'RCP45_A2_MPE':'#ef0096', 'RCP85_A2_MPE':'#790149'}

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


def plot_var_pError(outpath:str, comp_shp:gpd.GeoDataFrame, var:str, min_max_var_dict:dict, rcp:str, scn:str):

    # colors = ['#c51b7d','#de77ae','#f1b6da','#fde0ef','#e6f5d0','#b8e186','#7fbc41','#4d9221']

    if var == 'PRECIP':
        colors = ['#6e0038','#9d2160','#c04f85','#db7ea9','#efaecc','#fde0ef','#e6f5d0','#b8e186']
        bounds = [-50, -40, -30, -20, -10, 0, 10]
    elif var == 'MAX_TMP':
        colors = ['#6e0038','#a72c6a','#cf6697','#eba2c3','#fde0ef','#e6f5d0','#b8e186','#7fbc41']
        bounds = [-2, -1.5, -1, -0.5, 0, 0.5, 1]
    elif var == 'PR_AG':
        colors = ['#6e0038','#9d2160','#c04f85','#db7ea9','#efaecc','#fde0ef','#e6f5d0','#b8e186']
        bounds = [-100000, -50000, -5000, -1000, -500, 0, 50]
    elif var == 'PR_INT':
        colors = ['#cf6697', '#eba2c3', '#fde0ef','#e6f5d0', '#a2cb77', '#659f3b', '#327216', '#004600']
        bounds = [-100, -50, 0, 50, 100, 150, 200]
    elif var == 'PR_NAT':
        bounds = [-750, -50, -25, 0, 15, 30, 50]
        colors = ['#6e0038','#c51b7d','#de77ae','#fde0ef','#e6f5d0','#b8e186','#7fbc41','#4d9221']

    cmap=ListedColormap(colors)
    norm = BoundaryNorm(bounds, len(colors), extend='both')


    if var in ['PR_AG', 'PR_INT', 'PR_NAT']:

        all_min = comp_shp[f'{var}_PER_ERROR'].min()
        all_max = comp_shp[f'{var}_PER_ERROR'].max()
        
        try:
            if all_min < min_max_var_dict[f'overall_{var}_min'][0]:
                min_max_var_dict[f'overall_{var}_min'] = (all_min, scn+'_'+rcp)
        except KeyError:
            min_max_var_dict[f'overall_{var}_min'] = (all_min, scn+'_'+rcp)
        try:
            if all_max > min_max_var_dict[f'overall_{var}_max'][0]:
                min_max_var_dict[f'overall_{var}_max'] = (all_max, scn+'_'+rcp)
        except KeyError:
            min_max_var_dict[f'overall_{var}_max'] = (all_max, scn+'_'+rcp)

        # comp_shp.plot(f'{var}_PER_ERROR', legend=True, title=f'{scn}_{rcp}')
        print(f"All\n\tmin: {all_min}\n\tmed: {comp_shp[f'{var}_PER_ERROR'].median()}\n\tmax: {all_max}\n")


        plt.figure(figsize=(15, 10))

        ax = plt.subplot(1, 1, 1) #dct[szn] + 1) # nrows, ncols, axes position
        # plot the continent on these axes
        shp.merge(comp_df.loc[comp_df['SEASON']=='Spring'].groupby('HUC08').sum()[f'{var}_PER_ERROR']/13 , on='HUC08').plot(
            f'{var}_PER_ERROR', legend=True, ax=ax, cmap=cmap, norm=norm)
        # set the title
        ax.set_title(f'{var} {rcp}-{scn}')
        ax.set_facecolor('darkgray')
        # set the aspect
        # adjustable datalim ensure that the plots have the same axes size
        ax.set_aspect('equal', adjustable='datalim')

        plt.tight_layout()
        # plt.show()
        plt.savefig(outpath, facecolor='w', edgecolor='w', transparent=False)

    else:
        
        # SEASONAL-HUC MPE
        plt.figure(figsize=(20, 5))

        for szn in ['Spring', 'Summer', 'Fall', 'Winter', 'all']:
            if szn == 'all':
                all_min = comp_shp[f'{var}_PER_ERROR'].min()
                all_max = comp_shp[f'{var}_PER_ERROR'].max()
                
                try:
                    if all_min < min_max_var_dict[f'overall_{var}_min'][0]:
                        min_max_var_dict[f'overall_{var}_min'] = (all_min, scn+'_'+rcp)
                except KeyError:
                    min_max_var_dict[f'overall_{var}_min'] = (all_min, scn+'_'+rcp)
                try:
                    if all_max > min_max_var_dict[f'overall_{var}_max'][0]:
                        min_max_var_dict[f'overall_{var}_max'] = (all_max, scn+'_'+rcp)
                except KeyError:
                    min_max_var_dict[f'overall_{var}_max'] = (all_max, scn+'_'+rcp)

                # comp_shp.plot(f'{var}_PER_ERROR', legend=True, title=f'{scn}_{rcp}')
                print(f"All\n\tmin: {all_min}\n\tmed: {comp_shp[f'{var}_PER_ERROR'].median()}\n\tmax: {all_max}\n")

            else:
                # create subplot axes in a 3x3 grid
                dct = {'Spring': 0, 'Summer': 1, 'Fall': 2, 'Winter': 3}
                ax = plt.subplot(1, 4, dct[szn] + 1) # nrows, ncols, axes position
                # plot the continent on these axes
                shp.merge(comp_df.loc[comp_df['SEASON']==szn].groupby('HUC08').sum()[f'{var}_PER_ERROR']/13 , on='HUC08').plot(
                    f'{var}_PER_ERROR', legend=True, ax=ax, cmap=cmap, norm=norm)
                # set the title
                ax.set_title(szn)
                ax.set_facecolor('darkgray')
                # set the aspect
                # adjustable datalim ensure that the plots have the same axes size
                ax.set_aspect('equal', adjustable='datalim')

                # shp.merge(comp_df.loc[comp_df['SEASON']==szn].groupby('HUC08').sum()[f'{var}_PER_ERROR']/13 , on='HUC08').plot(f'{var}_PER_ERROR', legend=True)
                # print(f"{szn}\n\tmin: {(comp_df.loc[comp_df['SEASON']==szn].groupby('HUC08').sum()[f'{var}_PER_ERROR']/13).min()}\
                #     \n\tmed: {(comp_df.loc[comp_df['SEASON']==szn].groupby('HUC08').sum()[f'{var}_PER_ERROR']/13).median()}\
                #     \n\tmax: {(comp_df.loc[comp_df['SEASON']==szn].groupby('HUC08').sum()[f'{var}_PER_ERROR']/13).max()}")

        plt.tight_layout()
        # plt.show()
        plt.savefig(outpath, facecolor='w', edgecolor='w', transparent=False)
    
    plt.close()

    return


def plot_HUC_MPE(mpe_huc_df:pd.DataFrame):
    mpe_huc_long = pd.melt(mpe_huc_df).rename(columns={'variable':'model', 'value':'mpe'})

    sns.set(font_scale=2)

    fig, ax = plt.subplots(figsize=(12,12))
    sns.boxplot(y='mpe',x='model', data=mpe_huc_long, showfliers=False, palette=MY_PALLET, ax=ax, showmeans=True,\
        meanprops={"marker":"o",
                    "markerfacecolor":"white", 
                    "markeredgecolor":"black",
                    "markersize":"12"})
    sns.stripplot(y='mpe',x='model', data=mpe_huc_long, color=".25", ax=ax, alpha=0.2)

    # ax.set_yticklabels(['Climate', 'Anthro.', 'Combo.', 'Climate', 'Anthro.', 'Combo.', 'Climate', 'Anthro.', 'Combo.'])
    # ax.set_ylabel('MERF                     RF                      LMM', size=30)
    ax.set_xlabel('Mean Percent Error', size=30)
    # ax.legend(handles=legend_elements, loc='upper left')
    # ax.set_xlim([-55,55])
    ax.axhline(0, color='black', linewidth=3, linestyle='--')

    # define patch area
    # rect = matplotlib.patches.Rectangle(
    #     xy=(55, 2.5),  # lower left corner of box: beginning of x-axis range & y coord)
    #     width=ax.get_xlim()[0]-ax.get_xlim()[1] -20,  # width from x-axis range
    #     height=3,
    #     #color='grey',
    #     #alpha=0.4, 
    #     edgecolor='darkgrey',
    #     linewidth=2,
    #     facecolor='none'
    # )
    # ax.add_patch(rect)

    # define patch area2
    # rect2 = matplotlib.patches.Rectangle(
    #     xy=(-1.5, -0.47),  # lower left corner of box: beginning of x-axis range & y coord)
    #     width=3,  # width from x-axis range
    #     height=8.95,
    #     #color='grey',
    #     #alpha=0.4, 
    #     edgecolor='red',
    #     linewidth=2,
    #     facecolor='none'
    # )
    # ax.add_patch(rect2)


def boxplot_var_mpe(mpe_huc_var_df, var):
    mpe_huc_long = pd.melt(mpe_huc_var_df).rename(columns={'variable':'model', 'value':'mpe'})
    mpe_huc_long['model'] = mpe_huc_long['model'].replace(f'_{var}','', regex=True)

    sns.set(font_scale=2)

    fig, ax = plt.subplots(figsize=(12,12))
    sns.boxplot(y='mpe',x='model', data=mpe_huc_long, showfliers=False, palette=MY_PALLET, ax=ax, showmeans=True,\
        meanprops={"marker":"o",
                    "markerfacecolor":"white", 
                    "markeredgecolor":"black",
                    "markersize":"12"})
    sns.stripplot(y='mpe',x='model', data=mpe_huc_long, color=".25", ax=ax, alpha=0.2)

    # ax.set_yticklabels(['Climate', 'Anthro.', 'Combo.', 'Climate', 'Anthro.', 'Combo.', 'Climate', 'Anthro.', 'Combo.'])
    # ax.set_ylabel('MERF                     RF                      LMM', size=30)
    # ax.set_xlabel('Mean Percent Error', size=30)
    ax.set_title(var)
    # ax.legend(handles=legend_elements, loc='upper left')
    # ax.set_ylim([-5500,200])
    ax.axhline(0, color='black', linewidth=3, linestyle='--')


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

colors = ['#8e0152', '#b01467', '#cb337e', '#e15396', '#f174ae', '#fb97c6', '#ffbadc', '#ffddf0',
          '#d4eec2', '#acdc8e', '#88c763', '#68b142']
bounds = [-3000, -2000, -1000, -500, -100, -50, -10, 0, 10, 50, 100]
cmap=ListedColormap(colors)
norm = BoundaryNorm(bounds, len(colors), extend='both')

# all_scn_perError_df = pd.DataFrame()
init = True

for scn, rcp in scn_rcp_lst:
    print(f'{scn} {rcp}')
    inpath_merf = f'../data/FutureData/GCM_FORESCE_CSVs/HUC_CI/MULTIMODEL_{rcp}_{scn}_MC_HUC_CI95.csv'
    merf_df = pd.read_csv(inpath_merf, index_col=0)
    merf_df = merf_df.loc[merf_df['YR_SZN'] < 201900]

    comp_df = df_full[['HUC08', 'OBS_PR_WATER', 'YR_SZN', 'YEAR', 'SEASON']].merge(merf_df[['HUC08', 'YR_SZN', 'MEAN']], on=['HUC08','YR_SZN'])
    comp_df['MEAN_OBS_DIFF'] = comp_df['MEAN'] - comp_df['OBS_PR_WATER']
    comp_df['PER_ERROR'] = (comp_df['OBS_PR_WATER'] - comp_df['MEAN']) / comp_df['OBS_PR_WATER'] * 100

    if init:
        all_scn_perError_df = comp_df[['HUC08', 'PER_ERROR']].groupby('HUC08').mean().reset_index()
        init = False
    else:
        all_scn_perError_df = all_scn_perError_df.merge(comp_df[['HUC08', 'PER_ERROR']].groupby('HUC08').mean().reset_index(), on='HUC08')
    all_scn_perError_df = all_scn_perError_df.rename(columns={'PER_ERROR':f'{rcp}_{scn}_MPE'})


    # SEASONAL MPE
    # comp_df.boxplot(column='PER_ERROR', by='SEASON', figsize=(12,10))

    comp_shp = shp.merge(comp_df.groupby('HUC08').sum()['PER_ERROR']/52 , on='HUC08')

    # SEASONAL-HUC MPE
    outpath = f'../imgs/Paper2/Results/proj_dswe_mpe_maps/MULTIMODEL_{rcp}_{scn}__DSWE_MPE.png'
    plt.figure(figsize=(40, 10))

    for szn in ['Spring', 'Summer', 'Fall', 'Winter', 'all']:
        if szn == 'all':
            all_min = comp_shp['PER_ERROR'].min()
            all_max = comp_shp['PER_ERROR'].max()
            
            if all_min < overall_min:
                overall_min = all_min
                min_max_dict['overall_min'] = (all_min, scn+'_'+rcp)
            if all_max > overall_max:
                overall_max = all_max
                min_max_dict['overall_max'] = (all_max, scn+'_'+rcp)

            # comp_shp.plot('PER_ERROR', legend=True, title=f'{scn}_{rcp}')
            # print(f"All\n\tmin: {all_min}\n\tmed: {comp_shp['PER_ERROR'].median()}\n\tmax: {all_max}\n")
        else:
            # create subplot axes in a 3x3 grid
            dct = {'Spring': 0, 'Summer': 1, 'Fall': 2, 'Winter': 3}
            ax = plt.subplot(1, 4, dct[szn] + 1) # nrows, ncols, axes position
            # plot the continent on these axes
            shp.merge(comp_df.loc[comp_df['SEASON']==szn].groupby('HUC08').sum()['PER_ERROR']/13 , 
                      on='HUC08').plot('PER_ERROR', legend=True, ax=ax, cmap=cmap, norm=norm)
            ax.set_facecolor('darkgray')
            # set the title
            ax.set_title(szn)
            # set the aspect
            # adjustable datalim ensure that the plots have the same axes size
            ax.set_aspect('equal', adjustable='datalim')

            # shp.merge(comp_df.loc[comp_df['SEASON']==szn].groupby('HUC08').sum()['PER_ERROR']/13 , on='HUC08').plot('PER_ERROR', legend=True)
            # print(f"{szn}\n\tmin: {(comp_df.loc[comp_df['SEASON']==szn].groupby('HUC08').sum()['PER_ERROR']/13).min()}\
            #       \n\tmed: {(comp_df.loc[comp_df['SEASON']==szn].groupby('HUC08').sum()['PER_ERROR']/13).median()}\
            #       \n\tmax: {(comp_df.loc[comp_df['SEASON']==szn].groupby('HUC08').sum()['PER_ERROR']/13).max()}")

    plt.suptitle(f'{scn}-{rcp} Seasonal-HUC MPE (proj-obs)')
    plt.tight_layout()
    # plt.show()
    plt.savefig(outpath, facecolor='w', edgecolor='w', transparent=False)


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

def mk_climate_lclu_multimodel_csv():

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

    return




######## main():

mk_climate_lclu_multimodel_csv()



shp = gpd.read_file('../data/Shapefiles/HUC08/HUC08_paper2/HUC08_paper2.shp')
shp = shp[['huc8','areasqkm', 'geometry']]
shp['HUC08'] = shp['huc8'].apply(lambda x : int(x))
shp['AREA'] = shp['areasqkm'].apply(lambda x : float(x))

dswe = pd.read_csv('../data/all_data_0118_p2.csv', index_col=0)
dswe = dswe.loc[dswe['YEAR'] >= 2006]
dswe = dswe.reset_index().iloc[:,1:]
dswe = clean_df(dswe)

standard_cols = ['HUC08', 'YR_SZN']

# Set up for loop variables
scn_lst = ['A1B', 'A2', 'B1', 'B2']
rcp_lst = ['RCP45', 'RCP85']
scn_rcp_lst = [(scn, rcp) for scn in scn_lst for rcp in rcp_lst]
var_lst = ['PRECIP', 'MAX_TMP', 'PR_AG', 'PR_INT', 'PR_NAT']

min_max_var_dict = {}
init = True

for scn, rcp in scn_rcp_lst:
    print(f'{scn} {rcp}')
    inpath_fixed_vars = f'../data/FutureData/GCM_FORESCE_CSVs/VAR_MEANS/MULTIMODEL_{rcp}_{scn}_VAR_MEAN.csv'
    fixed_var_df = pd.read_csv(inpath_fixed_vars, index_col=0)
    fixed_var_df = clean_df(fixed_var_df)
    fixed_var_df = fixed_var_df.loc[fixed_var_df['YR_SZN'] < 201900]

    fixed_multi_cols = standard_cols + [col for col in fixed_var_df.columns if 'MULTIMODEL' in col]
    dswe_obs_cols = [col for col in dswe.columns if 'WATER' not in col]    

    comp_df = dswe[dswe_obs_cols].merge(fixed_var_df[fixed_multi_cols], on=['HUC08','YR_SZN'])
    # comp_cols = [col for col in comp_df.columns if 'YEAR' not in col and 'SEASON' not in col]
    # comp_df = comp_df[comp_cols]
    for var in var_lst:
        print(f'{var}')
        # outpath = f'../imgs/Paper2/var_projections/MC_mean_maps/MULTIMODEL_{rcp}_{scn}_{var}_MEAN.png'
        # if os.path.exists(outpath):
        #     print(f'{os.path.basename(outpath)} exists.')
        #     continue

        comp_df[f'{var}_OBS_DIFF'] = comp_df[f'MULTIMODEL_MEAN_{var}'] - comp_df[var]
        comp_df[f'{var}_PER_ERROR'] = (comp_df[var] - comp_df[f'MULTIMODEL_MEAN_{var}']) / comp_df[var] * 100
        
        if init:
            all_scn_var_perError_df = comp_df[['HUC08', f'{var}_PER_ERROR']].groupby('HUC08').mean().reset_index()
            init = False
        else:
            all_scn_var_perError_df = all_scn_var_perError_df.merge(comp_df[['HUC08', f'{var}_PER_ERROR']].groupby('HUC08').mean().reset_index(), on='HUC08')
        all_scn_var_perError_df = all_scn_var_perError_df.rename(columns={f'{var}_PER_ERROR':f'{rcp}_{scn}_{var}_MPE'})

        # SEASONAL MPE
        # comp_df.boxplot(column=f'{var}_PER_ERROR', by='SEASON', figsize=(12,10))

        comp_shp = shp.merge(comp_df.groupby('HUC08').sum()[f'{var}_PER_ERROR']/52 , on='HUC08')

        plot_var_pError(outpath, comp_shp, var, min_max_var_dict, rcp, scn)

for var in var_lst:
    var_cols = [col for col in all_scn_var_perError_df.columns if var in col]

    var_perError_df = all_scn_var_perError_df[['HUC08'] + var_cols]

    boxplot_var_mpe(var_perError_df.iloc[:,1:], var)

    for scn, rcp in scn_rcp_lst:
        print(f'\n{rcp} {scn} -- {var}')
        print(f"max: {var_perError_df.iloc[var_perError_df[[f'{rcp}_{scn}_{var}_MPE']].idxmax()][['HUC08',f'{rcp}_{scn}_{var}_MPE']].to_string(index=False, header=False)}")
        print(f"min: {var_perError_df.iloc[var_perError_df[[f'{rcp}_{scn}_{var}_MPE']].idxmin()][['HUC08',f'{rcp}_{scn}_{var}_MPE']].to_string(index=False, header=False)}")
        print(f"mean: {var_perError_df[[f'{rcp}_{scn}_{var}_MPE']].mean().to_string(index=False, header=False)}")
        print(f"median: {var_perError_df[[f'{rcp}_{scn}_{var}_MPE']].median().to_string(index=False, header=False)}")


