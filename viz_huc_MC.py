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

def get_ci(full_df, z:float=1.96, study_area:bool=False, shp:pd.DataFrame=None):


    if study_area:
        if shp is not None:
            full_df = full_df.merge(shp[['HUC08', 'AREA']], on='HUC08')
        for i in range(2, len(full_df.iloc[:,2:].columns)):
            full_df.iloc[:,i] = full_df.iloc[:,i] * full_df['AREA']
        full_df = full_df.groupby('YR_SZN').sum()
        full_df = full_df.iloc[:,:-1]
        col_idx = 1
        CI_df = pd.DataFrame()

    else:
        col_idx = 2
        CI_df = pd.DataFrame()
        CI_df['YR_SZN'] = full_df['YR_SZN']
        CI_df['HUC08'] = full_df['HUC08']

    CI_df['MEAN'] = full_df.iloc[:,col_idx:].mean(axis=1)
    CI_df['STDEV'] = full_df.iloc[:,col_idx:].std(axis=1)
    n = len(full_df.iloc[:,col_idx:].columns)
    CI_df['LOWER_95_CI'] = CI_df['MEAN'] - ((z * CI_df['STDEV']) / math.sqrt(n))
    CI_df['UPPER_95_CI'] = CI_df['MEAN'] + ((z * CI_df['STDEV']) / math.sqrt(n))
    # CI_df

    return(CI_df)


def plot_gcm_ci(ci_dict:dict, huc:int, name:str):

    obs_df = pd.read_csv('../data/all_data_0118_p2.csv', index_col=0)
    obs_df = clean_df(obs_df, pred=False)
    obs_huc_df = obs_df[obs_df['HUC08'] == huc].sort_values('YR_SZN')


    fig, ax1 = plt.subplots(figsize=(20, 8))

    for key in ci_dict.keys():

        # scn = f'{key.split("_")[0]} - {key.split("_")[1]}'
        if key == 'RCP45_B1':
            col = '#005a32' # green
            lab = 'RCP 4.5 - B1'
        elif key == 'RCP85_B2':
            col = '#084594' # blue
            lab = 'RCP 8.5 - B2'
        elif key == 'RCP85_A2':
            col = '#91003f' # pink
            lab = 'RCP 8.5 - A2'
        else:
            continue

        CI_df_full = ci_dict[key]
        CI_df = CI_df_full[CI_df_full['HUC08'] == huc].sort_values('YR_SZN')
        # CI_df['YR_SZN'] = CI_df.index
        # CI_df['YEAR'] = CI_df['YR_SZN'].apply(lambda x: str(x)[0:4])
        # CI_df = CI_df.groupby('YEAR').mean()
        x = [i for i in range(375)]

        ax1.plot(np.asarray(CI_df['MEAN']), color = col, linewidth=1 , label=lab)
        ax1.fill_between(x, CI_df['LOWER_95_CI'], CI_df['UPPER_95_CI'], color = col, alpha=.3, label='95% CI')

    ax1.plot(np.asarray(obs_huc_df['OBS_WATER']), color = 'black', linewidth=1 , label='Observed - (DSWE)')

    # reordering the labels
    handles, labels = plt.gca().get_legend_handles_labels()
    
    # specify order
    order = [0, 1, 2, 3, 4, 5, 6]
    
    # pass handle & labels lists along with order as below
    ax1.legend([handles[i] for i in order], [labels[i] for i in order])
    # plt.show()

    ax1.set_xticks([0,16,36,56,76,96,116,136,156,176,196,216,236,256,276,296,316,336,356,375 ], minor=False)
    ax1.set_xticklabels(['','2010','','2020','','2030','','2040','','2050','','2060','','2070','','2080','','2090','','2099'], size = 16)

    ax1.xaxis.grid(True, which='major', linestyle = (0, (5, 10)))
    ax1.set_ylabel('Percent Surface Water Area',size=20)
    ax1.set_xlabel('Year', size=20)
    ax1.set_title(f'{name} - HUC {huc}', size=20)

    
    plt.savefig(f'../imgs/Paper2/Results/HUC_CI/HUC_{huc}_MC_MultCI.png', dpi=300,\
        facecolor='w', edgecolor='w', transparent=False, pad_inches=0)

    plt.close()

    print(f'HUC_{huc}_MC_MultCI.png saved. \n')

    return




def main():

    shp = gpd.read_file('../data/Shapefiles/HUC08/HUC08_paper2/HUC08_paper2.shp')
    shp = shp[['huc8','areasqkm']]
    shp['HUC08'] = shp['huc8'].apply(lambda x : int(x))
    shp['AREA'] = shp['areasqkm'].apply(lambda x : float(x))


    rcp_lst = ['RCP45', 'RCP85']
    foresce_lst = ['A1B', 'A2', 'B1', 'B2']
    gcm_lst = ['GFDL', 'HadGEM2', 'IPSL', 'MIROC5', 'NorESM1']


    for rcp in rcp_lst:
        for foresce in foresce_lst:
            full_MC_ci_info_dict = {}

            for gcm in gcm_lst:

                print(f'starting {gcm}')

                full_MC_path = f'../data/FutureData/GCM_FORESCE_CSVs/{gcm}/{gcm}_{rcp}_{foresce}_MC_output.csv'

                multi_output_ci = f'../data/FutureData/GCM_FORESCE_CSVs/HUC_CI/MULTIMODEL_{rcp}_{foresce}_MC_HUC_CI95.csv'
                multi_output_SA_ci = f'../data/FutureData/GCM_FORESCE_CSVs/StudyAreaCI/MULTIMODEL_{rcp}_{foresce}_MC_HUC_CI95.csv'

                if not os.path.exists(multi_output_ci) or not os.path.exists(multi_output_SA_ci):
                    full_MC_df = pd.read_csv(full_MC_path, index_col=0)

                    if gcm == 'GFDL':
                        multi_df = full_MC_df
                        del(full_MC_df)
                    else:
                        multi_df = multi_df.merge(full_MC_df, on=['YR_SZN', 'HUC08'], suffixes=[None, f'_{gcm}'])
                        del(full_MC_df)

                outpath_ci = f'../data/FutureData/GCM_FORESCE_CSVs/HUC_CI/{gcm}_{rcp}_{foresce}_MC_HUC_CI95.csv'

                if not os.path.exists(outpath_ci):
                    full_MC_df = pd.read_csv(full_MC_path, index_col=0)
                    mc_ci = get_ci(full_MC_df, study_area=False)
                    mc_ci.to_csv(outpath_ci)
                    print(f'{os.path.basename(outpath_ci)} saved')
                    del(full_MC_df)
                else:
                    print(f'{os.path.basename(outpath_ci)} already saved')
                    mc_ci = pd.read_csv(outpath_ci, index_col=0)

                
                # full_MC_ci_info_dict[f'{gcm}_{rcp}_{foresce}'] = mc_ci

                print(f'finished {gcm}')
            
            if not os.path.exists(multi_output_ci):
                mc_ci = get_ci(multi_df, study_area=False)
                mc_ci.to_csv(multi_output_ci)
    

            if not os.path.exists(multi_output_SA_ci):
                mc_ci = get_ci(multi_df, study_area=True, shp=shp)
                mc_ci.to_csv(multi_output_SA_ci)

            print(f'finished {foresce}')

            # plot_gcm_ci(full_MC_ci_info_dict)

            # del(full_MC_ci_info_dict)

    scn_ci_dict = {}
    scn_lst = ['RCP45_B1', 'RCP85_B2', 'RCP85_A2']
    for scn in scn_lst:
        multi_output_ci = f'../data/FutureData/GCM_FORESCE_CSVs/HUC_CI/MULTIMODEL_{scn}_MC_HUC_CI95.csv'
        ci_df = pd.read_csv(multi_output_ci, index_col=0)
        scn_ci_dict[scn] = ci_df
    

    huc_lst = [3130001, 3020201, 3100205, 6010201, 8090203]
    huc_names = ['Upper Chattahoochee', 'Upper Neuse', 'Hillsborough', 'Watts Bar Lake', 'Eastern Louisiana Coastal']
    for i in range(len(huc_lst)):
        plot_gcm_ci(scn_ci_dict, huc_lst[i], huc_names[i])
    

    return

if __name__ == '__main__':
    main()