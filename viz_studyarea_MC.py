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
        for i in range(2,1002):
            full_df.iloc[:,i] = full_df.iloc[:,i] * full_df['AREA']
        full_df = full_df.groupby('YR_SZN').sum()
        full_df = full_df.iloc[:,:-1]

    CI_df = pd.DataFrame()
    CI_df['MEAN'] = full_df.iloc[:,1:].mean(axis=1)
    CI_df['STDEV'] = full_df.iloc[:,1:].std(axis=1)
    n = len(full_df.iloc[:,2:].columns)
    CI_df['LOWER_95_CI'] = CI_df['MEAN'] - ((z * CI_df['STDEV']) / math.sqrt(n))
    CI_df['UPPER_95_CI'] = CI_df['MEAN'] + ((z * CI_df['STDEV']) / math.sqrt(n))
    # CI_df

    return(CI_df)


def plot_gcm_ci(ci_dict:dict):

    fig, ax1 = plt.subplots(figsize=(20, 8))

    for key in ci_dict.keys():

        CI_df = ci_dict[key]
        # CI_df['YR_SZN'] = CI_df.index
        # CI_df['YEAR'] = CI_df['YR_SZN'].apply(lambda x: str(x)[0:4])
        # CI_df = CI_df.groupby('YEAR').mean()
        x = [i for i in range(375)]

        ax1.plot(np.asarray(CI_df['MEAN']),linewidth=2, alpha=0.5, label=key.split("_")[0])
        ax1.fill_between(x, CI_df['LOWER_95_CI'], CI_df['UPPER_95_CI'],  alpha=.3, label='95% CI')


    # reordering the labels
    handles, labels = plt.gca().get_legend_handles_labels()
    
    # specify order
    order = [0, 2, 4, 6, 8, 1]
    
    # pass handle & labels lists along with order as below
    ax1.legend([handles[i] for i in order], [labels[i] for i in order])
    # plt.show()

    ax1.set_xticks([0,16,36,56,76,96,116,136,156,176,196,216,236,256,276,296,316,336,356,375 ], minor=False)
    ax1.set_xticklabels(['','2010','','2020','','2030','','2040','','2050','','2060','','2070','','2080','','2090','','2099'], size = 16)

    ax1.xaxis.grid(True, which='major', linestyle = (0, (5, 10)))
    ax1.yaxis.grid(True, which='major', linestyle = (0, (5, 10)))
    ax1.set_ylabel('Surface Water Area',size=20)
    ax1.set_ylim([87000, 142000])
    ax1.set_xlabel('Year', size=20)
    ax1.set_title(f'{key.split("_")[-2]} - {key.split("_")[-1]}', size=20)

    
    plt.savefig(f'../imgs/Paper2/Results/study_area_CI/{key.split("_")[-2]}_{key.split("_")[-1]}.png', dpi=300,\
        facecolor='w', edgecolor='w', transparent=False, pad_inches=0)

    plt.close()

    print(f'{key.split("_")[-2]}_{key.split("_")[-1]}.png saved. \n')

    return


def plot_gcm_multMean_box(ci_dict:dict, rcp:str, foresce:str):
        #plot boxplot

    multi_mean = (ci_dict[f'GFDL_{rcp}_{foresce}']['MEAN'] + \
        ci_dict[f'HadGEM2_{rcp}_{foresce}']['MEAN'] + \
        ci_dict[f'IPSL_{rcp}_{foresce}']['MEAN'] + \
        ci_dict[f'MIROC5_{rcp}_{foresce}']['MEAN'] + \
        ci_dict[f'NorESM1_{rcp}_{foresce}']['MEAN'] ) /5

    df = pd.DataFrame()
    for key in ci_dict.keys():
        df[f'{key.split("_")[0]}'] = ci_dict[key]['MEAN'] - multi_mean

    fig, ax = plt.subplots(figsize=(12,12))
    sns.boxplot(data=df, ax=ax, showmeans=True,\
        meanprops={"marker":"o",
                            "markerfacecolor":"white", 
                            "markeredgecolor":"black",
                            "markersize":"12"})
    sns.stripplot(data=df, color=".25", ax=ax, alpha=0.2)
    ax.set_title(f'{rcp}_{foresce}')
    ax.set_ylim([-500, 500])

    plt.savefig(f'../imgs/Paper2/Results/study_area_multiMean_BoxPlot/multiMean_{key.split("_")[-2]}_{key.split("_")[-1]}.png', dpi=300,\
        facecolor='w', edgecolor='w', transparent=False, pad_inches=0)

    plt.close()

    print(f'{key.split("_")[-2]}_{key.split("_")[-1]}.png saved. \n')

    return


def main():

    shp = gpd.read_file('../data/Shapefiles/HUC08/HUC08_paper2/HUC08_paper2.shp')
    shp = shp[['huc8','areasqkm']]
    shp['HUC08'] = shp['huc8'].apply(lambda x : int(x))
    shp['AREA'] = shp['areasqkm'].apply(lambda x : float(x))


    rcp_lst = ['RCP45', 'RCP85']
    foresce_lst = ['A1B', 'A2', 'B1', 'B2']
    gcm_lst = ['GFDL', 'HadGEM2', 'IPSL', 'MIROC5', 'NorESM1']

    full_MC_ci_info_dict = {}
    for rcp in rcp_lst:
        for foresce in foresce_lst:

            for gcm in gcm_lst:

                print(f'starting {gcm}')

                outpath = f'../data/FutureData/GCM_FORESCE_CSVs/{gcm}/{gcm}_{rcp}_{foresce}_MC_output.csv'
                outpath_ci = f'../data/FutureData/GCM_FORESCE_CSVs/StudyAreaCI/{gcm}_{rcp}_{foresce}_MC_CI95.csv'

                # if the MC output has not already been saved into one table, save it
                if not os.path.exists(outpath):

                    mc_sim_lst = glob(f'../data/FutureData/GCM_FORESCE_CSVs/{gcm}/{gcm}_{rcp}_{foresce}/{gcm}_{rcp}_{foresce}*.csv')

                    for i in range(len(mc_sim_lst)):

                        df_sim = pd.read_csv(mc_sim_lst[i], usecols=['YEAR','SEASON','PRED_LOG_WATER','HUC08'])
                        df_sim = clean_df(df_sim)

                        if i == 0:
                            full_MC_df = df_sim[['YR_SZN','HUC08', 'PR_WATER']]
                            # full_MC_df['AREA'] = full_MC_df.merge(shp, on='HUC08')
                        else:
                            full_MC_df = full_MC_df.merge(df_sim[['YR_SZN','HUC08','PR_WATER']], on=['YR_SZN', 'HUC08'], suffixes=[None,f'_{i}'])

                        del(df_sim)

                    full_MC_df.to_csv(outpath)
                    print(f'{os.path.basename(outpath)} saved')
                else:
                    print(f'{os.path.basename(outpath)} already saved')
                    
                
                if not os.path.exists(outpath_ci):
                    try:    
                        mc_ci = get_ci(full_MC_df, study_area=True, shp=shp)
                        mc_ci.to_csv(f'../data/FutureData/GCM_FORESCE_CSVs/StudyAreaCI/{gcm}_{rcp}_{foresce}_MC_CI95.csv')
                        print(f'{os.path.basename(outpath_ci)} saved')
                        del(full_MC_df)
                    except NameError:
                        full_MC_df = pd.read_csv(outpath, index_col=0)
                        mc_ci = get_ci(full_MC_df, study_area=True, shp=shp)
                        mc_ci.to_csv(f'../data/FutureData/GCM_FORESCE_CSVs/StudyAreaCI/{gcm}_{rcp}_{foresce}_MC_CI95.csv')
                        print(f'{os.path.basename(outpath_ci)} saved')
                        del(full_MC_df)
                else:
                    print(f'{os.path.basename(outpath_ci)} already saved')
                    mc_ci = pd.read_csv(outpath_ci, index_col=0)

                
                full_MC_ci_info_dict[f'{gcm}_{rcp}_{foresce}'] = mc_ci

                print(f'finished {gcm}')

            print(f'finished {foresce}')

            plot_gcm_ci(full_MC_ci_info_dict)
            # plot_gcm_multMean_box(full_MC_ci_info_dict, rcp, foresce)

            del(full_MC_ci_info_dict)

                

    

    return

if __name__ == '__main__':
    main()



obs_df = pd.read_csv('../data/all_data_0118_p2.csv', index_col=0)
obs_df = clean_df(obs_df, pred=False)
obs_df = obs_df[obs_df['YEAR']==2018]
obs_df = obs_df.merge(shp[['HUC08','AREA']], on='HUC08')
obs_df['WATER_AREA'] = obs_df['PR_WATER'] * obs_df['AREA']  
obs_df = obs_df.groupby('YR_SZN').sum()
obs_df['PR_WATER'] = obs_df['WATER_AREA'] / obs_df['AREA']

total_area = obs_df.iloc[[0]].AREA.values[0]

gcm_cols = {'GFDL': '#fde725', 
            'HadGEM2': '#81c067', 
            'IPSL': '#278b8c', 
            'MIROC5': '#434d7c', 
            'NorESM1': '#440154'}

# new fig of study area gcms
fig = plt.figure(figsize=(15, 18), layout='constrained')
gs = fig.add_gridspec(4,2, hspace=.1, wspace=.03)
((ax1, ax2), (ax3, ax4), (ax5, ax6), (ax7, ax8)) = gs.subplots()

fig.supxlabel('Year', size=25)
fig.supylabel('Difference from 2018 Surface Water Area (%)', size=25)

ax2_inset = ax2.inset_axes([0.6,0.6,0.4,0.4])
ax2_inset.set_xticklabels([])
# ax2_inset.tick_params(direction="in")

for rcp in rcp_lst:
    for foresce in foresce_lst:
        if rcp == 'RCP45' and foresce == 'B1':
            ax = ax1
            ax.set_ylabel('B1',size=22)
            ax.set_title('RCP 4.5', size=20)

        elif rcp == 'RCP85' and foresce == 'B1':
            ax = ax2
            ax.set_title('RCP 8.5', size=20)
        elif rcp == 'RCP45' and foresce == 'B2':
            ax = ax3
            ax.set_ylabel('B2',size=22)

        elif rcp == 'RCP85' and foresce == 'B2':
            ax = ax4
        elif rcp == 'RCP45' and foresce == 'A1B':
            ax = ax5
            ax.set_ylabel('A1B',size=22)

        elif rcp == 'RCP85' and foresce == 'A1B':
            ax = ax6
        elif rcp == 'RCP45' and foresce == 'A2':
            ax = ax7
            ax.set_ylabel('A2',size=22)

        elif rcp == 'RCP85' and foresce == 'A2':
            ax = ax8
        
        for gcm in gcm_lst:

            CI_df = full_MC_ci_info_dict[f'{gcm}_{rcp}_{foresce}']

            lst = [i for i in range(375) if i % 4 == 0]
            spring_CI_df = (CI_df.iloc[lst][['MEAN', 'LOWER_95_CI', 'UPPER_95_CI']] / obs_df.iloc[0].AREA - obs_df.iloc[0].PR_WATER) * 100
            lst = [i + 1 for i in range(375) if i % 4 == 0]
            summer_CI_df = (CI_df.iloc[lst][['MEAN', 'LOWER_95_CI', 'UPPER_95_CI']] / obs_df.iloc[1].AREA - obs_df.iloc[1].PR_WATER) * 100
            lst = [i + 2 for i in range(375) if i % 4 == 0]
            fall_CI_df = (CI_df.iloc[lst][['MEAN', 'LOWER_95_CI', 'UPPER_95_CI']] / obs_df.iloc[2].AREA - obs_df.iloc[2].PR_WATER) * 100
            lst = [i + 3 for i in range(375) if i % 4 == 0][:-1]
            winter_CI_df = (CI_df.iloc[lst][['MEAN', 'LOWER_95_CI', 'UPPER_95_CI']] / obs_df.iloc[3].AREA - obs_df.iloc[3].PR_WATER) * 100

            CI_df = pd.concat([spring_CI_df, summer_CI_df, fall_CI_df, winter_CI_df])
            CI_df = CI_df.sort_index()

            x = [i for i in range(375)]
            ax.plot(np.asarray(CI_df['MEAN']),linewidth=2, alpha=0.5, label=gcm, color=gcm_cols[gcm])
            ax.fill_between(x, CI_df['LOWER_95_CI'], CI_df['UPPER_95_CI'],  alpha=.3, label='95% CI', color=gcm_cols[gcm])
            ax.plot(x, [0] * 375, color='black')

            if rcp == 'RCP85' and foresce == 'B1':
                ax2_inset.plot(np.asarray(CI_df['MEAN']),linewidth=2, alpha=0.5, color=gcm_cols[gcm])
                ax2_inset.fill_between(x, CI_df['LOWER_95_CI'], CI_df['UPPER_95_CI'],  alpha=.3, label='95% CI', color=gcm_cols[gcm])
                ax2_inset.set_xlim(99.7,101.3) #(371,374)
                # ax2_inset.set_ylim(1.35,1.75)
                ax2_inset.set_ylim(0.7,0.88)
                # ax2_inset.set_yticklabels([1.5])
                ax2_inset.xaxis.grid(True, which='major', linestyle = (0, (5, 10)))
                # ax2_inset.yaxis.grid(True, which='major', linestyle = (0, (5, 10)))
                ax.indicate_inset_zoom(ax2_inset, edgecolor="black")

    
# # reordering the labels
handles, labels = plt.gca().get_legend_handles_labels()

# # specify order
order = [0, 2, 4, 6, 8]

# # pass handle & labels lists along with order as below
ax4.legend([(handles[i], handles[i+1]) for i in order], [labels[i] for i in order], ncol=2, fontsize=18)
# plt.show()
for ax in fig.get_axes():

    ax.set_xlim(-2,377)
    ax.set_xticks([0,16,36,56,76,96,116,136,156,176,196,216,236,256,276,296,316,336,356,375 ], minor=False)
    ax.set_xticklabels(['','2010','','2020','','2030','','2040','','2050','','2060','','2070','','2080','','2090','','2099'], size = 16)

    ax.xaxis.grid(True, which='major', linestyle = (0, (5, 10)))
    ax.yaxis.grid(True, which='major', linestyle = (0, (5, 10)))
    # ax.set_ylabel('Surface Water Area',size=20)
    ax.set_ylim([-1, 5.5])
    # ax.set_xlabel('Year', size=20)

    ax.label_outer()

    # ax.set_title(f'{key.split("_")[-2]} - {key.split("_")[-1]}', size=20)



# plt.setp(ax.collections, alpha=.3)
# new fig of study area gcms
fig = plt.figure(figsize=(15, 18), layout='constrained')
gs = fig.add_gridspec(4,2, hspace=.1, wspace=.03)
((ax1, ax2), (ax3, ax4), (ax5, ax6), (ax7, ax8)) = gs.subplots()

fig.supxlabel('Global Climate Models', size=25)
fig.supylabel('Difference from Multimodel Mean Surface Water Area (%)', size=25)

# ax2_inset = ax2.inset_axes([0.6,0.6,0.4,0.4])
# ax2_inset.set_xticklabels([])
# ax2_inset.tick_params(direction="in")

for rcp in rcp_lst:
    for foresce in foresce_lst:
        if rcp == 'RCP45' and foresce == 'B1':
            ax = ax1
            ax.set_ylabel('B1',size=22)
            ax.set_title('RCP 4.5', size=20)
            colors = ['#008607', '#509e42', '#7eb770', '#aacf9e', '#d4e7ce']

        elif rcp == 'RCP85' and foresce == 'B1':
            ax = ax2
            ax.set_title('RCP 8.5', size=20)
            colors = ['#005a01', '#437a37', '#729a66', '#a0bb97', '#cfddca']
        elif rcp == 'RCP45' and foresce == 'B2':
            ax = ax3
            ax.set_ylabel('B2',size=22)
            colors = ['#009ffa', '#68b1fc', '#96c4fd', '#bcd7fe', '#deebff']

        elif rcp == 'RCP85' and foresce == 'B2':
            ax = ax4
            colors = ['#00489e', '#5269b2', '#808cc5', '#abb0d8', '#d5d7ec']

        elif rcp == 'RCP45' and foresce == 'A1B':
            ax = ax5
            ax.set_ylabel('A1B',size=22)
            colors = ['#ff3cfe', '#ff76ff', '#ff9eff', '#ffc1ff', '#ffe1ff']

        elif rcp == 'RCP85' and foresce == 'A1B':
            ax = ax6
            colors = ['#65019f', '#8844b3', '#a872c6', '#c6a0d9', '#e3cfec']

        elif rcp == 'RCP45' and foresce == 'A2':
            ax = ax7
            ax.set_ylabel('A2',size=22)
            colors = ['#ef0096', '#f75caa', '#fd8abf', '#ffb3d4', '#ffdaea']

        elif rcp == 'RCP85' and foresce == 'A2':
            ax = ax8
            colors = ['#790149', '#97426a', '#b3718d', '#cd9fb2', '#e7ced8']

        multi_mean = (full_MC_ci_info_dict[f'GFDL_{rcp}_{foresce}']['MEAN'] + \
                    full_MC_ci_info_dict[f'HadGEM2_{rcp}_{foresce}']['MEAN'] + \
                    full_MC_ci_info_dict[f'IPSL_{rcp}_{foresce}']['MEAN'] + \
                    full_MC_ci_info_dict[f'MIROC5_{rcp}_{foresce}']['MEAN'] + \
                    full_MC_ci_info_dict[f'NorESM1_{rcp}_{foresce}']['MEAN'] ) /5  
        
        df = pd.DataFrame()
        for gcm in gcm_lst:

            # CI_df = full_MC_ci_info_dict[f'{gcm}_{rcp}_{foresce}']

            # for key in full_MC_ci_info_dict.keys():
            df[f'{gcm}'] = full_MC_ci_info_dict[f'{gcm}_{rcp}_{foresce}']['MEAN'] - multi_mean

        # fig, ax = plt.subplots(figsize=(12,12))
        sns.violinplot(data=(df / total_area) * 100, ax=ax, linewidth=0, inner=None,
                        color='lightgray', saturation=0.6)# palette=colors)#, boxprops={'zorder':1, 'alpha': 0.1})

        sns.boxplot(data=(df / total_area) * 100, ax=ax, showmeans=True,\
            meanprops={"marker":"o",
                                "markerfacecolor":"white", 
                                "markeredgecolor":"black",
                                "markersize":"12"},
                        palette=colors, boxprops={'zorder':2, 'alpha': 0.75})
        # sns.stripplot(data=(df / total_area) * 100, color=".25", ax=ax, alpha=0.2)
        # ax.set_title(f'{rcp}_{foresce}')
        ax.set_ylim([-.05, .05])

        ax.label_outer()

# for ax in fig.get_axes():
#     for violin in ax.collections[::2]:
#         violin.set_alpha(0.2)

plt.savefig(f'../imgs/Paper2/Results/study_area_multiMean_BoxPlot/boxViolin_studyArea_plot.png', dpi=300,\
    facecolor='w', edgecolor='w', transparent=False, pad_inches=0)