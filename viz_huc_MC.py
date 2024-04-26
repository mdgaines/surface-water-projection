import os
import numpy as np 
import pandas as pd 
import geopandas as gpd

import math

import seaborn as sns
from matplotlib import cm
from matplotlib.colors import ListedColormap
import matplotlib.pyplot as plt
import matplotlib as mpl


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

    CI_df['MIN'] = full_df.iloc[:,col_idx:].min(axis=1)
    CI_df['MAX'] = full_df.iloc[:,col_idx:].max(axis=1)
    CI_df['90th'] = full_df.iloc[:,col_idx:].quantile(q=0.9 ,axis=1)
    CI_df['10th'] = full_df.iloc[:,col_idx:].quantile(q=0.1 ,axis=1)
    CI_df['25th'] = full_df.iloc[:,col_idx:].quantile(q=0.25 ,axis=1)
    CI_df['75th'] = full_df.iloc[:,col_idx:].quantile(q=0.75 ,axis=1)
    CI_df['50th'] = full_df.iloc[:,col_idx:].quantile(q=0.5 ,axis=1)

    n = len(full_df.iloc[:,col_idx:].columns)
    CI_df['LOWER_95_CI'] = CI_df['MEAN'] - ((z * CI_df['STDEV']) / math.sqrt(n))
    CI_df['UPPER_95_CI'] = CI_df['MEAN'] + ((z * CI_df['STDEV']) / math.sqrt(n))
    # CI_df

    return(CI_df)


def per_change_info(obs_huc_df:pd.DataFrame, CI_df:pd.DataFrame, stat:str):

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
    spring_CI_df = (CI_df.iloc[lst][[stat, '90th', '10th']] - mean_obs_huc_df.iloc[0].MEAN_OBS_WATER) / mean_obs_huc_df.iloc[0].MEAN_OBS_WATER * 100
    lst = [i + 1 for i in range(375) if i % 4 == 0]
    # summer_CI_df = (CI_df.iloc[lst][['MEAN', 'LOWER_95_CI', 'UPPER_95_CI']] - mean_obs_huc_df.iloc[1].MEAN_OBS_WATER) / mean_obs_huc_df.iloc[1].MEAN_OBS_WATER * 100
    summer_CI_df = (CI_df.iloc[lst][[stat, '90th', '10th']] - mean_obs_huc_df.iloc[1].MEAN_OBS_WATER) / mean_obs_huc_df.iloc[1].MEAN_OBS_WATER * 100
    lst = [i + 2 for i in range(375) if i % 4 == 0]
    # fall_CI_df = (CI_df.iloc[lst][['MEAN', 'LOWER_95_CI', 'UPPER_95_CI']] - mean_obs_huc_df.iloc[2].MEAN_OBS_WATER) / mean_obs_huc_df.iloc[2].MEAN_OBS_WATER * 100
    fall_CI_df = (CI_df.iloc[lst][[stat, '90th', '10th']] - mean_obs_huc_df.iloc[2].MEAN_OBS_WATER) / mean_obs_huc_df.iloc[2].MEAN_OBS_WATER * 100
    lst = [i + 3 for i in range(375) if i % 4 == 0][:-1]
    # winter_CI_df = (CI_df.iloc[lst][['MEAN', 'LOWER_95_CI', 'UPPER_95_CI']] - mean_obs_huc_df.iloc[3].MEAN_OBS_WATER) / mean_obs_huc_df.iloc[3].MEAN_OBS_WATER * 100
    winter_CI_df = (CI_df.iloc[lst][[stat, '90th', '10th']] - mean_obs_huc_df.iloc[3].MEAN_OBS_WATER) / mean_obs_huc_df.iloc[3].MEAN_OBS_WATER * 100

    CI_df = pd.concat([spring_CI_df, summer_CI_df, fall_CI_df, winter_CI_df])
    CI_df = CI_df.sort_index()

    return CI_df

def plot_huc_ci(ci_dict:dict, huc:int, name:str, per_change:bool, stat:str):

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

        if per_change is True:
            CI_df = per_change_info(obs_huc_df, CI_df, stat)
            outpath = f'../imgs/Paper2/Results/HUC_CI/HUC_{huc}_MC_Mult_10-90_{stat}_pChange.png'
        else:
            outpath = f'../imgs/Paper2/Results/HUC_CI/HUC_{huc}_MC_Mult_10-90_{stat}.png'

        # CI_df['YR_SZN'] = CI_df.index
        # CI_df['YEAR'] = CI_df['YR_SZN'].apply(lambda x: str(x)[0:4])
        # CI_df = CI_df.groupby('YEAR').mean()
        x = [i for i in range(375)]

        ax1.plot(np.asarray(CI_df[stat]), color = col, linewidth=1 , label=lab)
        # ax1.fill_between(x, CI_df['LOWER_95_CI'], CI_df['UPPER_95_CI'], color = col, alpha=.3, label='95% CI')
        ax1.fill_between(x, CI_df['90th'], CI_df['10th'], color = col, alpha=.2, label='10-90th Percentile')


    if per_change is True:
        ax1.set_ylabel('Percent Change in Surface Water Area',size=20)
        ax1.hlines(xmin=0, xmax=374, y=0, color = 'black', linewidth=1)

        # specify order
        order = [0, 1, 2, 3, 4, 5]

    else:
        ax1.plot(np.asarray(obs_huc_df['OBS_WATER']), color = 'black', linewidth=1 , label='Observed - (DSWE)')
        ax1.set_ylabel('Percent Surface Water Area',size=20)

        # specify order
        order = [0, 1, 2, 3, 4, 5, 6]

    # reordering the labels
    handles, labels = plt.gca().get_legend_handles_labels()

    # pass handle & labels lists along with order as below
    ax1.legend([handles[i] for i in order], [labels[i] for i in order])
    # plt.show()

    ax1.set_xticks([0,16,36,56,76,96,116,136,156,176,196,216,236,256,276,296,316,336,356,375 ], minor=False)
    ax1.set_xticklabels(['','2010','','2020','','2030','','2040','','2050','','2060','','2070','','2080','','2090','','2099'], size = 16)

    ax1.xaxis.grid(True, which='major', linestyle = (0, (5, 10)))
    ax1.set_xlabel('Year', size=20)
    ax1.set_title(f'{name} - HUC 0{huc}', size=20)
    
    plt.savefig(outpath, dpi=300,\
        facecolor='w', edgecolor='w', transparent=False, pad_inches=0)

    plt.close()

    print(f'HUC_{huc}_MC_Mult_10-90pChange.png saved. \n')

    return


def plot_multHUC_ci(ci_dict:dict, huc_lst:list, name_lst:list, per_change:bool, stat:str):

    outpath = f'../imgs/Paper2/Results/HUC_CI/HUC_group_MC_Mult_10-90_{stat}_pChange.png'

    obs_df = pd.read_csv('../data/all_data_0118_p2.csv', index_col=0)
    obs_df = clean_df(obs_df, pred=False)

    # read image file
    file = '../imgs/Paper2/Results/HUC_CI/pink_hucs_abb.jpeg'
    arr_image = plt.imread(file, format='jpeg')

    fig = plt.figure(figsize=(15, 17), layout='constrained')
    gs = fig.add_gridspec(3,1, hspace=.05, wspace=.03)
    ((ax1), (ax2), (ax3)) = gs.subplots()

    ax_lst = [ax1, ax2, ax3]

    fig.supxlabel('Year', size=30)
    fig.supylabel('Percent Change from 2001-2018 Mean Surface Water Area', size=30)

    for i in range(len(huc_lst)):
        huc = huc_lst[i]
        name = name_lst[i]
        obs_huc_df = obs_df[obs_df['HUC08'] == huc].sort_values('YR_SZN')
        ax = ax_lst[i]

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

            if per_change is True:
                CI_df = per_change_info(obs_huc_df, CI_df, stat)
                # outpath = f'../imgs/Paper2/Results/HUC_CI/HUC_{huc}_MC_Mult_10-90_pChange.png'
            # else:
                # outpath = f'../imgs/Paper2/Results/HUC_CI/HUC_{huc}_MC_Mult_10-90.png'

            # CI_df['YR_SZN'] = CI_df.index
            # CI_df['YEAR'] = CI_df['YR_SZN'].apply(lambda x: str(x)[0:4])
            # CI_df = CI_df.groupby('YEAR').mean()
            x = [i for i in range(375)]

            # ax.plot(np.asarray(CI_df['MEAN']), color = col, linewidth=1 , label=lab)
            ax.plot(np.asarray(CI_df[stat]), color = col, linewidth=1 , label=lab)
            # ax.fill_between(x, CI_df['LOWER_95_CI'], CI_df['UPPER_95_CI'], color = col, alpha=.3, label='95% CI')
            ax.fill_between(x, CI_df['90th'], CI_df['10th'], color = col, alpha=.2, label='10-90th Percentile')


        if per_change is True:
            # ax.set_ylabel('Percent Change in Surface Water Area',size=20)
            ax.hlines(xmin=0, xmax=374, y=0, color = 'black', linewidth=1)

            # specify order
            # order = [0, 1, 2, 3, 4, 5]

        else:
            ax.plot(np.asarray(obs_huc_df['OBS_WATER']), color = 'black', linewidth=1 , label='Observed - (DSWE)')
            # ax.set_ylabel('Percent Surface Water Area',size=20)

        # pass handle & labels lists along with order as below
        # ax.legend([handles[i] for i in order], [labels[i] for i in order])
        # plt.show()
        if huc == 3020201:
            # Draw image
            axin = ax.inset_axes([0.02,0.35,.62,.62], anchor='NW')    # create new inset axes in data coordinates
            axin.imshow(arr_image)
            # axin.axis('off')
            axin.set_xticks([])
            axin.set_yticks([])


        ax.set_xticks([0,16,36,56,76,96,116,136,156,176,196,216,236,256,276,296,316,336,356,375 ], minor=False)
        ax.set_xticklabels(['','2010','','2020','','2030','','2040','','2050','','2060','','2070','','2080','','2090','','2099'], size = 16)

        ax.xaxis.grid(True, which='major', linestyle = (0, (5, 10)))
        # ax.set_xlabel('Year', size=20)
        ax.set_title(f'{name} - HUC 0{huc}', size=26)
        
    
    # reordering the labels
    handles, labels = plt.gca().get_legend_handles_labels()
    
    # # specify order
    order = [0, 2, 4]

    # # pass handle & labels lists along with order as below
    ax1.legend([(handles[i], handles[i+1]) for i in order], [labels[i] for i in order],\
                ncol=1, fontsize=22,
                loc='upper right', bbox_to_anchor=(0.47, 1))
    ax1.text(377,1690,'(a)', size=30)
    ax2.text(377,-5,'(b)', size=30)
    ax3.text(377,1.5,'(c)', size=30)

    plt.savefig(outpath, dpi=300,\
        facecolor='w', edgecolor='w', transparent=False, pad_inches=0)

    plt.close()

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

            # plot_huc_ci(full_MC_ci_info_dict)

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
        plot_huc_ci(scn_ci_dict, huc_lst[i], huc_names[i], per_change=True, stat='MEAN')
        plot_huc_ci(scn_ci_dict, huc_lst[i], huc_names[i], per_change=False, stat='MEAN')

        plot_huc_ci(scn_ci_dict, huc_lst[i], huc_names[i], per_change=True, stat='50th')
        plot_huc_ci(scn_ci_dict, huc_lst[i], huc_names[i], per_change=False, stat='50th')

    name_lst = [ 'Upper Neuse (UN)', 'Watts Bar Lake (WBL)', 'Eastern Louisiana Coastal (ELC)']
    huc_lst = [3020201, 6010201, 8090203]
    plot_multHUC_ci(scn_ci_dict, huc_lst, name_lst, per_change=True, stat='MEAN')
    plot_multHUC_ci(scn_ci_dict, huc_lst, name_lst, per_change=True, stat='50th')

    return

if __name__ == '__main__':
    main()