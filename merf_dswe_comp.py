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

# Set the style
sns.set_style("darkgrid")

MY_PALLET = {'RCP45_B1_MPE':'#008607', 'RCP85_B1_MPE':'#005a01',
            'RCP45_B2_MPE':'#009ffa', 'RCP85_B2_MPE':'#00489e',
            'RCP45_A1B_MPE':'#ff3cfe', 'RCP85_A1B_MPE':'#65019f',
            'RCP45_A2_MPE':'#ef0096', 'RCP85_A2_MPE':'#790149'}

def clean_df(df, huc=False, pred=True):

    # df['HUC_SEASON'] = df[huc].astype('str') + '_' + df['SEASON']
    # df.loc[df.SEASON == 'Spring', 'YR_SZN'] = df.YEAR * 100 + 0
    # df.loc[df.SEASON == 'Summer', 'YR_SZN'] = df.YEAR * 100 + 25
    # df.loc[df.SEASON == 'Fall', 'YR_SZN'] = df.YEAR * 100 + 50
    # df.loc[df.SEASON == 'Winter', 'YR_SZN'] = df.YEAR * 100 + 75
    # df = df.sort_values(by=['YR_SZN'])
    
    # df['HUC_SEASON'] = df[huc].astype('str') + '_' + df['SEASON']
    df.loc[df.SEASON == 'Spring', 'YR_SZN'] = df.loc[df.SEASON == 'Spring', 'YEAR'] * 100 + 0
    df.loc[df.SEASON == 'Summer', 'YR_SZN'] = df.loc[df.SEASON == 'Summer', 'YEAR'] * 100 + 25
    df.loc[df.SEASON == 'Fall', 'YR_SZN'] = df.loc[df.SEASON == 'Fall', 'YEAR'] * 100 + 50
    df.loc[df.SEASON == 'Winter', 'YR_SZN'] = df.loc[df.SEASON == 'Winter', 'YEAR'] * 100 + 75
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
        plt.show()
        # plt.savefig(outpath, facecolor='w', edgecolor='w', transparent=False, bbox_inches='tight')

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
        plt.show()
        # plt.savefig(outpath, facecolor='w', edgecolor='w', transparent=False, bbox_inches='tight')
    
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
    # sns.stripplot(y='mpe',x='model', data=mpe_huc_long, color=".25", ax=ax, alpha=0.2)

    # ax.set_yticklabels(['Climate', 'Anthro.', 'Combo.', 'Climate', 'Anthro.', 'Combo.', 'Climate', 'Anthro.', 'Combo.'])
    # ax.set_ylabel('MERF                     RF                      LMM', size=30)
    ax.set_xlabel('Mean Percent Error', size=30)
    # ax.legend(handles=legend_elements, loc='upper left')
    # ax.set_xlim([-55,55])
    ax.axhline(0, color='black', linewidth=3, linestyle='--')


def plot_HUC_MPE_4x1(mpe_huc_df, szn=False):
 
    mpe_huc_df['HUC02'] = mpe_huc_df['HUC08'].apply(lambda x: str(x)[0].zfill(2))

    plt.rcParams['font.family'] = 'Times New Roman'
    plt.rcParams['font.size'] = '28'

    fig = plt.figure(figsize=(23,6))
    gs = fig.add_gridspec(1, 4, wspace=.1, hspace=.1, bottom=0.18)
    ax1, ax2, ax3, ax4 = gs.subplots()
    
    if szn:
        mpe_huc_df = mpe_huc_df[mpe_huc_df['SEASON']==szn]
        fig.supylabel(szn, ha='right')
   
    # Study Area
    sns.boxplot(y='PER_ERROR', x='FORE-SCE', hue='RCP',\
                 data=mpe_huc_df, showfliers=False, #palette=MY_PALLET, \
                 ax=ax1, showmeans=True, order=['B1', 'B2', 'A1B' ,'A2'], \
                 meanprops={"marker":"o",
                            "markerfacecolor":"white", 
                            "markeredgecolor":"black",
                            "markersize":"12"})
    ax1.set_title('Study Area', size=28,fontweight="bold")
    ax1.set_ylabel('HUC08 Mean Percent Deviation\n(2006-2018)', size=28)
    ax1.set_xlabel('')

    # South Atlantic Gulf
    sns.boxplot(y='PER_ERROR', x='FORE-SCE', hue='RCP',\
                 data=mpe_huc_df.loc[mpe_huc_df['HUC02']=='03'], showfliers=False, #palette=MY_PALLET, \
                 ax=ax2, showmeans=True, order=['B1', 'B2', 'A1B' ,'A2'], \
                 meanprops={"marker":"o",
                            "markerfacecolor":"white", 
                            "markeredgecolor":"black",
                            "markersize":"12"})
    ax2.set_title('South Atlantic Gulf', size=28,fontweight="bold")
    ax2.set_ylabel('')
    ax2.set_xlabel('')

    # Tennessee
    sns.boxplot(y='PER_ERROR', x='FORE-SCE', hue='RCP',\
                 data=mpe_huc_df.loc[mpe_huc_df['HUC02']=='06'], showfliers=False, #palette=MY_PALLET, \
                 ax=ax3, showmeans=True, order=['B1', 'B2', 'A1B' ,'A2'], \
                 meanprops={"marker":"o",
                            "markerfacecolor":"white", 
                            "markeredgecolor":"black",
                            "markersize":"12"})
    ax3.set_title('Tennessee', size=28,fontweight="bold")
    ax3.set_ylabel('')
    ax3.set_xlabel('')

    # Lower Mississippi
    sns.boxplot(y='PER_ERROR', x='FORE-SCE', hue='RCP',\
                 data=mpe_huc_df.loc[mpe_huc_df['HUC02']=='08'], showfliers=False, #palette=MY_PALLET, \
                 ax=ax4, showmeans=True, order=['B1', 'B2', 'A1B' ,'A2'], \
                 meanprops={"marker":"o",
                            "markerfacecolor":"white", 
                            "markeredgecolor":"black",
                            "markersize":"12"})
    ax4.set_title('Lower Mississippi', size=28,fontweight="bold")
    ax4.set_ylabel('')
    ax4.set_xlabel('')
    # sns.move_legend(ax4, "upper left", bbox_to_anchor=(1, 0.75))

    sns.move_legend(ax3, "upper left", title='Climate Scenarios')#, bbox_to_anchor=(1, 0.75))

    ax1.get_legend().remove()
    ax2.get_legend().remove()
    ax4.get_legend().remove()

    ax2.set_yticklabels([])
    ax3.set_yticklabels([])
    ax4.set_yticklabels([])

    ax1.set_ylim(-120,1100)
    ax2.set_ylim(-120,1100)
    ax3.set_ylim(-120,1100)
    ax4.set_ylim(-120,1100)

    ax1.axhline(0, color='black', linewidth=2, linestyle='--', alpha=0.7)
    ax2.axhline(0, color='black', linewidth=2, linestyle='--', alpha=0.7)
    ax3.axhline(0, color='black', linewidth=2, linestyle='--', alpha=0.7)
    ax4.axhline(0, color='black', linewidth=2, linestyle='--', alpha=0.7)

    fig.supxlabel('SRES Land Cover Scenarios', size=28)

    plt.savefig(f'../imgs/Paper2/Results/proj_dswe_mpe_boxplots/MULTIMODEL_DSWE_MEAN_all_4x1.png', facecolor='w', edgecolor='w', transparent=False, bbox_inches='tight')



def plot_HUC_MPE_4x4(mpe_huc_df, var='PER_ERROR'):
 
    mpe_huc_df['HUC02'] = mpe_huc_df['HUC08'].apply(lambda x: str(x)[0].zfill(2))

    plt.rcParams['font.family'] = 'Times New Roman'
    plt.rcParams['font.size'] = '26'

    fig = plt.figure(figsize=(20,20))
    gs = fig.add_gridspec(4, 4, wspace=.02, hspace=.04, bottom=0.07, left=0.1)
    ax_lst = gs.subplots()
    
    szn_lst = ['Spring', 'Summer', 'Fall', 'Winter']

    fig.supylabel('HUC08 Mean Percent Deviation (2006-2018)', size=38.)

    for i in range(4):
        szn = szn_lst[i]
        szn_mpe_huc_df = mpe_huc_df[mpe_huc_df['SEASON']==szn]
   
        # Study Area
        ax1 = ax_lst[i,0]
        sns.boxplot(y=var, x='FORE-SCE', hue='RCP',\
                    data=szn_mpe_huc_df, showfliers=False, #palette=MY_PALLET, \
                    ax=ax1, showmeans=True, order=['B1', 'B2', 'A1B' ,'A2'], \
                    meanprops={"marker":"o",
                                "markerfacecolor":"white", 
                                "markeredgecolor":"black",
                                "markersize":"12"})
        # ax1.set_ylabel(szn, size=28)
        ax1.set_ylabel('')
        ax1.set_xlabel('')

        # South Atlantic Gulf
        ax2 = ax_lst[i,1]
        sns.boxplot(y=var, x='FORE-SCE', hue='RCP',\
                    data=szn_mpe_huc_df.loc[szn_mpe_huc_df['HUC02']=='03'], showfliers=False, #palette=MY_PALLET, \
                    ax=ax2, showmeans=True, order=['B1', 'B2', 'A1B' ,'A2'], \
                    meanprops={"marker":"o",
                                "markerfacecolor":"white", 
                                "markeredgecolor":"black",
                                "markersize":"12"})
        ax2.set_ylabel('')
        ax2.set_xlabel('')

        # Tennessee
        ax3 = ax_lst[i,2]
        sns.boxplot(y=var, x='FORE-SCE', hue='RCP',\
                    data=szn_mpe_huc_df.loc[szn_mpe_huc_df['HUC02']=='06'], showfliers=False, #palette=MY_PALLET, \
                    ax=ax3, showmeans=True, order=['B1', 'B2', 'A1B' ,'A2'], \
                    meanprops={"marker":"o",
                                "markerfacecolor":"white", 
                                "markeredgecolor":"black",
                                "markersize":"12"})
        ax3.set_ylabel('')
        ax3.set_xlabel('')

        # Lower Mississippi
        ax4 = ax_lst[i,3]
        sns.boxplot(y=var, x='FORE-SCE', hue='RCP',\
                    data=szn_mpe_huc_df.loc[szn_mpe_huc_df['HUC02']=='08'], showfliers=False, #palette=MY_PALLET, \
                    ax=ax4, showmeans=True, order=['B1', 'B2', 'A1B' ,'A2'], \
                    meanprops={"marker":"o",
                                "markerfacecolor":"white", 
                                "markeredgecolor":"black",
                                "markersize":"12"})
        # ax4.set_ylabel('')
        ax4.yaxis.set_label_position("right")
        ax4.set_ylabel(szn, size=32, rotation=-90, labelpad=35)
        ax4.set_xlabel('')

        ax2.set_yticklabels([])
        ax3.set_yticklabels([])
        ax4.set_yticklabels([])

        ax1.axhline(0, color='black', linewidth=2, linestyle='--', alpha=0.7)
        ax2.axhline(0, color='black', linewidth=2, linestyle='--', alpha=0.7)
        ax3.axhline(0, color='black', linewidth=2, linestyle='--', alpha=0.7)
        ax4.axhline(0, color='black', linewidth=2, linestyle='--', alpha=0.7)

    fig.supxlabel('SRES Land Cover Scenarios', size=32)

    ax_lst[0,0].set_title('Study Area', size=32,fontweight="bold")
    ax_lst[0,1].set_title('South Atlantic Gulf', size=32,fontweight="bold")
    ax_lst[0,2].set_title('Tennessee', size=32,fontweight="bold")
    ax_lst[0,3].set_title('Lower Mississippi', size=32,fontweight="bold")

    for i in range(4):
        for j in range(4):

            if i != 3:
                ax_lst[i,j].set_xticklabels([])
            if i == 3:
                ax_lst[i][j].tick_params(bottom=True)

            if var == 'PER_ERROR': # dswe mpe's
                if i == 2:
                    ax_lst[i,j].set_ylim(-120,2500)
                else:
                    ax_lst[i,j].set_ylim(-120,1100)
                ax_lst[i,j].axhline(1000, color='gray', linewidth=2, linestyle='--', alpha=0.5)
                # if i == 3 and j == 0:
                #     sns.move_legend(ax_lst[3,0], "upper right", title='Climate Scenarios')
                #     continue

            if var == 'PRECIP_PER_ERROR': # precip mpe's
                ax_lst[i,j].set_ylim(-25,80)
            
            if var == 'MAX_TMP_PER_ERROR': # precip mpe's
                ax_lst[i,j].set_ylim(-0.25,2.25)

            if i == 2 and j == 3:
                sns.move_legend(ax_lst[2,3], "upper left", title='Climate Scenarios', bbox_to_anchor=(1.05, 1.25), title_fontsize=28, fontsize=28)
                continue            

            ax_lst[i,j].get_legend().remove()
                
    plt.savefig(f'../imgs/Paper2/Results/proj_dswe_mpe_boxplots/MULTIMODEL_{var}_SZN_MEAN_all_4x4.png', facecolor='w', edgecolor='w', transparent=False, bbox_inches='tight')

    return


def plot_HUC_LCLU_var_MPE_4x4(mpe_huc_df):
 
    mpe_huc_df['HUC02'] = mpe_huc_df['HUC08'].apply(lambda x: str(x)[0].zfill(2))

    plt.rcParams['font.family'] = 'Times New Roman'
    plt.rcParams['font.size'] = '26'

    fig = plt.figure(figsize=(20,15))
    gs = fig.add_gridspec(4, 5, wspace=.02, hspace=.06, bottom=0.07, left=0.1, height_ratios=[1, .2, .8, 1], width_ratios=[1, .2, 1, 1, 1])
    ax_lst = [ [fig.add_subplot(gs[0,0]), fig.add_subplot(gs[0,1]), fig.add_subplot(gs[0,2]), fig.add_subplot(gs[0,3]), fig.add_subplot(gs[0,4])],
               [fig.add_subplot(gs[1,0]), fig.add_subplot(gs[1:3,1]), fig.add_subplot(gs[1:3,2]), fig.add_subplot(gs[1:3,3]), fig.add_subplot(gs[1, 4])],
               [fig.add_subplot(gs[2,0]), fig.add_subplot(gs[2,4])],
               [fig.add_subplot(gs[3,0]), fig.add_subplot(gs[3,1]), fig.add_subplot(gs[3,2]), fig.add_subplot(gs[3,3]), fig.add_subplot(gs[3,4])]]

    ax_lst[0][1].set_visible(False)               
    ax_lst[1][1].set_visible(False)               
    # ax_lst[2][1].set_visible(False)               
    ax_lst[3][1].set_visible(False)               
    # ax_lst[0][1].set_visible(False)               
                   
    var_lst = ['PR_NAT_PER_ERROR', 'PR_AG_PER_ERROR', 'PR_INT_PER_ERROR']
    var_label_dict = dict(zip(var_lst, ['Forest-Dominated', 'Agricultural', 'Intensive']))
    
    d = .025  # how big to make the diagonal lines in axes coordinates
    # arguments to pass to plot, just so we don't keep repeating them

    fig.supylabel('HUC08 Mean Percent Deviation (2006-2018)', size=38.)

    for i in range(4):
        var_mpe_huc_df = mpe_huc_df[mpe_huc_df['SEASON']=='Spring']

        if i == 1 or i == 2:
            var = 'PR_AG_PER_ERROR'

            if i == 2:
                # Study Area
                ax1 = ax_lst[i][0]
                sns.boxplot(y=var, x='FORE-SCE', hue='RCP',\
                            data=var_mpe_huc_df, showfliers=False, #palette=MY_PALLET, \
                            ax=ax1, showmeans=True, order=['B1', 'B2', 'A1B' ,'A2'], \
                            meanprops={"marker":"o",
                                        "markerfacecolor":"white", 
                                        "markeredgecolor":"black",
                                        "markersize":"12"})
                # ax1.set_ylabel(var, size=28)
                ax1.set_ylabel('')
                ax1.set_xlabel('')
                ax1.set_ylim(-200,2100)
                
                # Lower Mississippi
                ax4 = ax_lst[i][1]
                sns.boxplot(y=var, x='FORE-SCE', hue='RCP',\
                            data=var_mpe_huc_df.loc[var_mpe_huc_df['HUC02']=='08'], showfliers=False, #palette=MY_PALLET, \
                            ax=ax4, showmeans=True, order=['B1', 'B2', 'A1B' ,'A2'], \
                            meanprops={"marker":"o",
                                        "markerfacecolor":"white", 
                                        "markeredgecolor":"black",
                                        "markersize":"12"})

                ax4.yaxis.set_label_position("right")
                ax4.set_ylabel(var_label_dict[var], size=32, rotation=-90, labelpad=42, y=.68)
                ax4.set_xlabel('')
                ax4.set_ylim(-200,1250)
                ax4.yaxis.tick_right()
                ax4.tick_params(right=False)


                ax1.axhline(0, color='black', linewidth=2, linestyle='--', alpha=0.7)
                ax4.axhline(0, color='black', linewidth=2, linestyle='--', alpha=0.7)

                ax1.axhline(1000, color='gray', linewidth=2, linestyle='--', alpha=0.5)
                ax4.axhline(1000, color='gray', linewidth=2, linestyle='--', alpha=0.5)


                # kwargs = dict(transform=ax1.transAxes, color='k', clip_on=False)
                # ax1.plot((-d, +d), (-d, +d), **kwargs)        # top-left diagonal
                # ax1.plot((1 - d, 1 + d), (-d, +d), **kwargs)  # top-right diagonal

                kwargs.update(transform=ax1.transAxes, color='dimgray', clip_on=False)  # switch to the bottom axes
                ax1.plot((-d, +d), (1 - d, 1 + d), **kwargs)  # bottom-left diagonal
                # ax1.plot((1 - d, 1 + d), (1 - d, 1 + d), **kwargs)  # bottom-right diagonal

                kwargs.update(transform=ax4.transAxes, color='dimgray', clip_on=False)  # switch to the bottom axes
                # ax4.plot((-d, +d), (1 - d, 1 + d), **kwargs)  # bottom-left diagonal
                ax4.plot((1 - d, 1 + d), (1 - d, 1 + d), **kwargs)  # bottom-right diagonal

                continue

            # Study Area
            ax1 = ax_lst[i][0]
            sns.boxplot(y=var, x='FORE-SCE', hue='RCP',\
                        data=var_mpe_huc_df, showfliers=False, #palette=MY_PALLET, \
                        ax=ax1, showmeans=True, order=['B1', 'B2', 'A1B' ,'A2'], \
                        meanprops={"marker":"o",
                                    "markerfacecolor":"white", 
                                    "markeredgecolor":"black",
                                    "markersize":"12"})
            # ax1.set_ylabel(var, size=28)
            ax1.set_ylabel('')
            ax1.set_xlabel('')
            ax1.set_ylim(3100,5700)

            # Lower Mississippi
            ax4 = ax_lst[i][4]
            sns.boxplot(y=var, x='FORE-SCE', hue='RCP',\
                        data=var_mpe_huc_df.loc[var_mpe_huc_df['HUC02']=='08'], showfliers=False, #palette=MY_PALLET, \
                        ax=ax4, showmeans=True, order=['B1', 'B2', 'A1B' ,'A2'], \
                        meanprops={"marker":"o",
                                    "markerfacecolor":"white", 
                                    "markeredgecolor":"black",
                                    "markersize":"12"})

            ax4.yaxis.set_label_position("right")
            # ax4.set_ylabel(var, size=32, rotation=-90, labelpad=35)
            ax4.set_xlabel('')
            ax4.set_ylabel('')
            ax4.set_ylim(9000,18000)
            ax4.yaxis.tick_right()
            ax4.tick_params(right=False)

            kwargs = dict(transform=ax1.transAxes, color='dimgray', clip_on=False)
            ax1.plot((-d, +d), (-d*3.5, +d*3.5), **kwargs)        # top-left diagonal
            # ax1.plot((1 - d, 1 + d), (-d, +d), **kwargs)  # top-right diagonal

            kwargs = dict(transform=ax4.transAxes, color='dimgray', clip_on=False)
            # ax4.plot((-d, +d), (-d, +d), **kwargs)        # top-left diagonal
            ax4.plot((1 - d, 1 + d), (-d*3.5, +d*3.5), **kwargs)  # top-right diagonal

            # kwargs.update(transform=ax4.transAxes)  # switch to the bottom axes
            # ax4.plot((-d, +d), (1 - d, 1 + d), **kwargs)  # bottom-left diagonal
            # ax4.plot((1 - d, 1 + d), (1 - d, 1 + d), **kwargs)  # bottom-right diagonal


            # South Atlantic Gulf
            ax2 = ax_lst[i][2]
            sns.boxplot(y=var, x='FORE-SCE', hue='RCP',\
                        data=var_mpe_huc_df.loc[var_mpe_huc_df['HUC02']=='03'], showfliers=False, #palette=MY_PALLET, \
                        ax=ax2, showmeans=True, order=['B1', 'B2', 'A1B' ,'A2'], \
                        meanprops={"marker":"o",
                                    "markerfacecolor":"white", 
                                    "markeredgecolor":"black",
                                    "markersize":"12"})
            ax2.set_ylabel('')
            ax2.set_xlabel('')
            ax2.set_ylim(-200,2200)

            # Tennessee
            ax3 = ax_lst[i][3]
            sns.boxplot(y=var, x='FORE-SCE', hue='RCP',\
                        data=var_mpe_huc_df.loc[var_mpe_huc_df['HUC02']=='06'], showfliers=False, #palette=MY_PALLET, \
                        ax=ax3, showmeans=True, order=['B1', 'B2', 'A1B' ,'A2'], \
                        meanprops={"marker":"o",
                                    "markerfacecolor":"white", 
                                    "markeredgecolor":"black",
                                    "markersize":"12"})
            ax3.set_ylabel('')
            ax3.set_xlabel('')
            ax3.set_ylim(-200,2200)


            # ax2.set_yticklabels([])
            ax3.set_yticklabels([])
            # ax4.set_yticklabels([])

            # ax1.axhline(0, color='black', linewidth=2, linestyle='--', alpha=0.7)
            ax2.axhline(0, color='black', linewidth=2, linestyle='--', alpha=0.7)
            ax3.axhline(0, color='black', linewidth=2, linestyle='--', alpha=0.7)
            
            ax2.axhline(1000, color='gray', linewidth=2, linestyle='--', alpha=0.5)
            ax3.axhline(1000, color='gray', linewidth=2, linestyle='--', alpha=0.5)

            # ax4.axhline(0, color='black', linewidth=2, linestyle='--', alpha=0.7)
            

            continue


        elif i == 0:
            var = var_lst[i]
        else:
            var = var_lst[2]

   
        # Study Area
        ax1 = ax_lst[i][0]
        sns.boxplot(y=var, x='FORE-SCE', hue='RCP',\
                    data=var_mpe_huc_df, showfliers=False, #palette=MY_PALLET, \
                    ax=ax1, showmeans=True, order=['B1', 'B2', 'A1B' ,'A2'], \
                    meanprops={"marker":"o",
                                "markerfacecolor":"white", 
                                "markeredgecolor":"black",
                                "markersize":"12"})
        # ax1.set_ylabel(var, size=28)
        ax1.set_ylabel('')
        ax1.set_xlabel('')

        # South Atlantic Gulf
        ax2 = ax_lst[i][2]
        sns.boxplot(y=var, x='FORE-SCE', hue='RCP',\
                    data=var_mpe_huc_df.loc[var_mpe_huc_df['HUC02']=='03'], showfliers=False, #palette=MY_PALLET, \
                    ax=ax2, showmeans=True, order=['B1', 'B2', 'A1B' ,'A2'], \
                    meanprops={"marker":"o",
                                "markerfacecolor":"white", 
                                "markeredgecolor":"black",
                                "markersize":"12"})
        ax2.set_ylabel('')
        ax2.set_xlabel('')

        # Tennessee
        ax3 = ax_lst[i][3]
        sns.boxplot(y=var, x='FORE-SCE', hue='RCP',\
                    data=var_mpe_huc_df.loc[var_mpe_huc_df['HUC02']=='06'], showfliers=False, #palette=MY_PALLET, \
                    ax=ax3, showmeans=True, order=['B1', 'B2', 'A1B' ,'A2'], \
                    meanprops={"marker":"o",
                                "markerfacecolor":"white", 
                                "markeredgecolor":"black",
                                "markersize":"12"})
        ax3.set_ylabel('')
        ax3.set_xlabel('')

        # Lower Mississippi
        ax4 = ax_lst[i][4]
        sns.boxplot(y=var, x='FORE-SCE', hue='RCP',\
                    data=var_mpe_huc_df.loc[var_mpe_huc_df['HUC02']=='08'], showfliers=False, #palette=MY_PALLET, \
                    ax=ax4, showmeans=True, order=['B1', 'B2', 'A1B' ,'A2'], \
                    meanprops={"marker":"o",
                                "markerfacecolor":"white", 
                                "markeredgecolor":"black",
                                "markersize":"12"})

        ax4.yaxis.set_label_position("right")
        ax4.set_ylabel(var_label_dict[var], size=32, rotation=-90, labelpad=35)
        ax4.set_xlabel('')

        ax2.set_yticklabels([])
        ax3.set_yticklabels([])
        ax4.set_yticklabels([])

        ax1.axhline(0, color='black', linewidth=2, linestyle='--', alpha=0.7)
        ax2.axhline(0, color='black', linewidth=2, linestyle='--', alpha=0.7)
        ax3.axhline(0, color='black', linewidth=2, linestyle='--', alpha=0.7)
        ax4.axhline(0, color='black', linewidth=2, linestyle='--', alpha=0.7)

        for j in range(5):
            if var == 'PR_NAT_PER_ERROR': # forest-dominated mpe's
                ax_lst[i][j].set_ylim(-55,55)

            if var == 'PR_AG_PER_ERROR' and i != 1 and i != 2: # agriculture mpe's
                ax_lst[i][j].set_ylim(-200,2200)
            
            if var == 'PR_INT_PER_ERROR': # intensive mpe's
                ax_lst[i][j].set_ylim(-275,275)

    fig.supxlabel('SRES Land Cover Scenarios', size=32)

    ax_lst[0][0].set_title('Study Area', size=32,fontweight="bold")
    ax_lst[0][2].set_title('South Atlantic Gulf', size=32,fontweight="bold")
    ax_lst[0][3].set_title('Tennessee', size=32,fontweight="bold")
    ax_lst[0][4].set_title('Lower Mississippi', size=32,fontweight="bold")

    for i in range(len(ax_lst)):
        for j in range(len(ax_lst[i])):

            if i != 3:
                ax_lst[i][j].set_xticklabels([])
            if i == 3:
                ax_lst[i][j].tick_params(bottom=True)

                # if i == 3 and j == 0:
                #     sns.move_legend(ax_lst[3,0], "upper right", title='Climate Scenarios')
                #     continue


            if i == 0 and j == 4:
                sns.move_legend(ax_lst[0][4], "upper left", title='Climate Scenarios', bbox_to_anchor=(1.14, 0.8), title_fontsize=28, fontsize=28)
                continue            

            try:
                ax_lst[i][j].get_legend().remove()
            except AttributeError:
                continue

    plt.savefig(f'../imgs/Paper2/Results/proj_dswe_mpe_boxplots/MULTIMODEL_LCLU_MEAN_all_4x3.png', facecolor='w', edgecolor='w', transparent=False, bbox_inches='tight')

    return
    


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
dswe_06_18_fl_lst = [fl for fl in dswe_obs_lst if int(os.path.basename(fl)[0:4]) in range(2006, 2019)]
dswe_06_18_fl_lst.sort()

df_full = pd.DataFrame()

for fl in dswe_06_18_fl_lst:
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
    comp_df['PER_ERROR'] = (comp_df['MEAN'] - comp_df['OBS_PR_WATER']) / comp_df['OBS_PR_WATER'] * 100

    if init:
        all_scn_perError_df = comp_df[['HUC08', 'PER_ERROR']].groupby('HUC08').mean().reset_index()
        all_scn_perError_df_long = comp_df[['HUC08', 'PER_ERROR']].groupby('HUC08').mean().reset_index()
        all_scn_perError_df_long['FORE-SCE'] = scn
        all_scn_perError_df_long['RCP'] = f'RCP {rcp[3]}.{rcp[4]}'
        
        szn_scn_perError_df_long = comp_df[['HUC08', 'SEASON', 'PER_ERROR']].groupby(['HUC08', 'SEASON']).mean().reset_index()
        szn_scn_perError_df_long['FORE-SCE'] = scn
        szn_scn_perError_df_long['RCP'] = f'RCP {rcp[3]}.{rcp[4]}'
        init = False
    else:
        all_scn_perError_df = all_scn_perError_df.merge(comp_df[['HUC08', 'PER_ERROR']].groupby('HUC08').mean().reset_index(), on='HUC08')
        all_scn_perError_df_long1 = comp_df[['HUC08', 'PER_ERROR']].groupby('HUC08').mean().reset_index()
        all_scn_perError_df_long1['FORE-SCE'] = scn
        all_scn_perError_df_long1['RCP'] = f'RCP {rcp[3]}.{rcp[4]}'
        all_scn_perError_df_long = all_scn_perError_df_long.append(all_scn_perError_df_long1)
        
        szn_scn_perError_df_long1 = comp_df[['HUC08', 'SEASON', 'PER_ERROR']].groupby(['HUC08', 'SEASON']).mean().reset_index()
        szn_scn_perError_df_long1['FORE-SCE'] = scn
        szn_scn_perError_df_long1['RCP'] = f'RCP {rcp[3]}.{rcp[4]}'
        szn_scn_perError_df_long = szn_scn_perError_df_long.append(szn_scn_perError_df_long1)
    all_scn_perError_df = all_scn_perError_df.rename(columns={'PER_ERROR':f'{rcp}_{scn}_MPE'})

    # plot_HUC_MPE(all_scn_perError_df)

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

            shp.merge(comp_df.loc[comp_df['SEASON']==szn].groupby('HUC08').sum()['PER_ERROR']/13 , on='HUC08').plot('PER_ERROR', legend=True)
            print(f"{szn}\n\tmin: {(comp_df.loc[comp_df['SEASON']==szn].groupby('HUC08').sum()['PER_ERROR']/13).min()}\
                  \n\tmed: {(comp_df.loc[comp_df['SEASON']==szn].groupby('HUC08').sum()['PER_ERROR']/13).median()}\
                  \n\tmax: {(comp_df.loc[comp_df['SEASON']==szn].groupby('HUC08').sum()['PER_ERROR']/13).max()}")

    plt.suptitle(f'{scn}-{rcp} Seasonal-HUC MPE (proj-obs)')
    plt.tight_layout()
    # plt.show()
    plt.savefig(outpath, facecolor='w', edgecolor='w', transparent=False, bbox_inches='tight')


plot_HUC_MPE_4x1(all_scn_perError_df_long)
plot_HUC_MPE_4x4(szn_scn_perError_df_long)

focal_scn_info = [['RCP 4.5', 'B1'], ['RCP 8.5', 'B2'], ['RCP 8.5', 'A2'],
                  ['RCP 8.5', 'B1'], ['RCP 4.5', 'B2'], ['RCP 4.5', 'A2'],
                  ['RCP 4.5', 'A1B'], ['RCP 8.5', 'A1B']]
season_lst = ['Spring', 'Summer', 'Fall', 'Winter']

for rcp, scn in focal_scn_info:
    sub_df = all_scn_perError_df_long[(all_scn_perError_df_long['RCP'] == rcp) & 
                                      (all_scn_perError_df_long['FORE-SCE'] == scn)]
    sub_gdf = shp.merge(sub_df, on='HUC08')

    print(rcp, scn)
    print(f"min: {sub_gdf['PER_ERROR'].min()}")
    # print(f"10: {sub_gdf['PER_ERROR'].quantile(.10)}")
    # print(f"20: {sub_gdf['PER_ERROR'].quantile(.20)}")
    # print(f"30: {sub_gdf['PER_ERROR'].quantile(.30)}")
    # print(f"40: {sub_gdf['PER_ERROR'].quantile(.40)}")
    print(f"med: {sub_gdf['PER_ERROR'].median()}")
    print(f"avg: {sub_gdf['PER_ERROR'].mean()}")
    # print(f"60: {sub_gdf['PER_ERROR'].quantile(.60)}")
    # print(f"70: {sub_gdf['PER_ERROR'].quantile(.70)}")
    # print(f"80: {sub_gdf['PER_ERROR'].quantile(.80)}")
    # print(f"90: {sub_gdf['PER_ERROR'].quantile(.90)}")
    print(f"max: {sub_gdf['PER_ERROR'].max()}\n")

# plot DSWE MPE over the HUCs for each season and focal scenario

import matplotlib.colors
import matplotlib as mpl
from matplotlib_scalebar.scalebar import ScaleBar
from shapely.geometry.point import Point

POINTS = gpd.GeoSeries(
    [Point(-73.5, 40.5), Point(-74.5, 40.5)], crs=4326
)  # Geographic WGS 84 - degrees

POINTS = POINTS.to_crs(32619)  # Projected WGS 84 - meters

DIST_METERS = POINTS[0].distance(POINTS[1])


HUC02 = gpd.read_file('../data/Shapefiles/HUC02/HUC02_clean_paper2/HUC02_clean_paper2.shp')


plt.rcParams['font.family'] = 'Times New Roman'
plt.rcParams['font.size'] = '16'



def per_error_maps(szn_shp, scn, szn, var):

    bounds = [-50, -10, 10, 50, 250, 500, 1000]

    colors = ['#f77495', '#ffbac0', '#fffce5', '#a7d1ef', '#6e9fe8', '#456dd4', '#283cb1', '#1a0a7b']      
        
    cmap = matplotlib.colors.ListedColormap(colors)
    norm = matplotlib.colors.BoundaryNorm(bounds, len(colors), extend='both')

    fig, ax = plt.subplots(figsize=(20, 12))
    
    ax.set_aspect('equal')
    ax.set_axis_off()

    szn_shp.plot(column=var, ax=ax, edgecolor='black', linewidth=0.5, cmap=cmap, norm=norm, alpha=0.9)

    HUC02.plot(ax=ax, edgecolor='black', facecolor='none', linewidth=4)

    if scn == 'RCP45_B1':
        ax.set_title(szn, size=74)
    
    if scn == 'RCP85_A2' and szn == 'Winter':
        ax.add_artist(ScaleBar(DIST_METERS, location='lower left', length_fraction=0.5, width_fraction=0.02, border_pad = 0.8, pad = 0.8))


    outpath = f"../imgs/Paper2/Results/proj_dswe_mpe_maps/MULTIMODEL_{scn}_{szn}.png"

    plt.savefig(outpath, dpi=300, facecolor='w', edgecolor='w', transparent=False, bbox_inches='tight')

    plt.close()

    return

def plot_legend():

    bounds = [-50, -10, 10, 50, 250, 500, 1000]

    colors = ['#f77495', '#ffbac0', '#fffce5', '#a7d1ef', '#6e9fe8', '#456dd4', '#283cb1', '#1a0a7b']      
   
    cmap = matplotlib.colors.ListedColormap(colors)
    norm = matplotlib.colors.BoundaryNorm(bounds, len(colors), extend='both')

    plt.rcParams['font.size'] = '72'

    fig = plt.figure(figsize=(20, 0.75))
    gs = fig.add_gridspec(1,1, wspace=2,hspace=2)

    gs.update(wspace=0, hspace=0)

    ax1 = gs.subplots()

    fig.colorbar(mpl.cm.ScalarMappable(norm=norm, cmap=cmap),
                cax=ax1, orientation='horizontal',
                label='Mean Percent Error (2006-2018)')
    
    return

focal_scn_info = [['RCP 4.5', 'B1'], ['RCP 8.5', 'B2'], ['RCP 8.5', 'A2']]
season_lst = ['Spring', 'Summer', 'Fall', 'Winter']

for rcp, scn in focal_scn_info:
    for szn in season_lst:
        sub_df = szn_scn_perError_df_long[(szn_scn_perError_df_long['SEASON'] == szn) & 
                                          (szn_scn_perError_df_long['RCP'] == rcp) & 
                                          (szn_scn_perError_df_long['FORE-SCE'] == scn)]
        sub_gdf = shp.merge(sub_df, on='HUC08')
        per_error_maps(sub_gdf, f'{rcp[0:3]}{rcp[4]}{rcp[6]}_{scn}', szn, 'PER_ERROR')

        # print(f"min: {sub_gdf['PER_ERROR'].min()}")
        # print(f"10: {sub_gdf['PER_ERROR'].quantile(.10)}")
        # print(f"20: {sub_gdf['PER_ERROR'].quantile(.20)}")
        # print(f"30: {sub_gdf['PER_ERROR'].quantile(.30)}")
        # print(f"40: {sub_gdf['PER_ERROR'].quantile(.40)}")
        # print(f"med: {sub_gdf['PER_ERROR'].median()}")
        # print(f"60: {sub_gdf['PER_ERROR'].quantile(.60)}")
        # print(f"70: {sub_gdf['PER_ERROR'].quantile(.70)}")
        # print(f"80: {sub_gdf['PER_ERROR'].quantile(.80)}")
        # print(f"90: {sub_gdf['PER_ERROR'].quantile(.90)}")
        # print(f"max: {sub_gdf['PER_ERROR'].max()}\n")

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

    dswe = pd.read_csv('../data/all_data_0118_p2.csv', index_col=0)
    dswe = clean_df(dswe, pred=False)

    rcp_lst = ['RCP85'] #['RCP45', 'RCP85']
    foresce_lst = ['A2', 'B2'] # ['A1B', 'A2', 'B1', 'B2']
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

                        szn_MC_df = full_MC_df[['HUC08', 'YEAR', 'SEASON', var]]#.loc[full_MC_df['YEAR'] <= 2018]

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
all_scn_var_perError_df_long_dict = {}
init1 = True
init2 = True

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
        outpath = f'../imgs/Paper2/var_projections/MC_mean_maps/MULTIMODEL_{rcp}_{scn}_{var}_MEAN.png'
        # if os.path.exists(outpath):
        #     print(f'{os.path.basename(outpath)} exists.')
        #     continue

        comp_df[f'{var}_OBS_DIFF'] = comp_df[f'MULTIMODEL_MEAN_{var}'] - comp_df[var]
        comp_df[f'{var}_PER_ERROR'] = (comp_df[f'MULTIMODEL_MEAN_{var}'] - comp_df[var]) / comp_df[var] * 100
        
        if init1:
            all_scn_var_perError_df = comp_df[['HUC08', f'{var}_PER_ERROR']].groupby('HUC08').mean().reset_index()
            init1 = False
        else:
            all_scn_var_perError_df = all_scn_var_perError_df.merge(comp_df[['HUC08', f'{var}_PER_ERROR']].groupby('HUC08').mean().reset_index(), on='HUC08')
        all_scn_var_perError_df = all_scn_var_perError_df.rename(columns={f'{var}_PER_ERROR':f'{rcp}_{scn}_{var}_MPE'})

        if var == var_lst[0]:
            szn_scn_var_perError_df_long = comp_df[['HUC08', 'SEASON', f'{var}_PER_ERROR']].groupby(['HUC08', 'SEASON']).mean().reset_index()
            szn_scn_var_perError_df_long['FORE-SCE'] = scn
            szn_scn_var_perError_df_long['RCP'] = f'RCP {rcp[3]}.{rcp[4]}'
            # init2 = False
        else:
            szn_scn_var_perError_df_long = szn_scn_var_perError_df_long.merge(comp_df[['HUC08', 'SEASON', f'{var}_PER_ERROR']].groupby(
                ['HUC08', 'SEASON']).mean().reset_index(), on=['HUC08', 'SEASON'])
    
        # SEASONAL MPE
        # comp_df.boxplot(column=f'{var}_PER_ERROR', by='SEASON', figsize=(12,10))

        comp_shp = shp.merge(comp_df.groupby('HUC08').sum()[f'{var}_PER_ERROR']/52 , on='HUC08')

        # plot_var_pError(outpath, comp_shp, var, min_max_var_dict, rcp, scn)
    
        # szn_scn_var_perError_df_long_dict[f'{rcp}_{scn}_{var}'] = szn_scn_var_perError_df_long

    if init2:
        szn_scn_var_perError_df_long_full = szn_scn_var_perError_df_long
        init2 = False
    else:
        # szn_scn_var_perError_df_long1['FORE-SCE'] = scn
        # szn_scn_var_perError_df_long1['RCP'] = f'RCP {rcp[3]}.{rcp[4]}'
        szn_scn_var_perError_df_long_full = szn_scn_var_perError_df_long_full.append(szn_scn_var_perError_df_long)


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


plot_HUC_MPE_4x4(szn_scn_var_perError_df_long_full, 'PRECIP_PER_ERROR')
plot_HUC_MPE_4x4(szn_scn_var_perError_df_long_full, 'MAX_TMP_PER_ERROR')
plot_HUC_LCLU_var_MPE_4x4(szn_scn_var_perError_df_long_full)



# updating for comp with 2001-2018 avg

def lulc_per_error_maps(df_shp, scn, yr, var):

    # if 'PER_ERROR' in var:
    #     bounds = [-60, -40, -25, -10, -5, 5, 10, 25, 40, 60] # 
    #     # bounds = [-60, -40, -25, -10, 0, 10, 25, 40, 60] # 
    # elif 'DIFF' in var:
    bounds = [-40, -30, -20, -10, 0, 10, 20, 30, 40]

    # colors = ['#543005', '#8c510a', '#bf812d', '#dfc27d', '#f6e8c3', 
    #           '#f5f5f5', 
    #           '#c7eae5', '#80cdc1', '#35978f', '#01665e', '#003c30']  

    colors = ['#543005', '#8c510a', '#bf812d', '#dfc27d', '#f6e8c3', 
              '#c7eae5', '#80cdc1', '#35978f', '#01665e', '#003c30']

    cmap = matplotlib.colors.ListedColormap(colors)
    norm = matplotlib.colors.BoundaryNorm(bounds, len(colors), extend='both')

    fig, ax = plt.subplots(figsize=(20, 12))
    
    ax.set_aspect('equal')
    ax.set_axis_off()

    df_shp.plot(column=var, ax=ax, edgecolor='black', linewidth=0.5, cmap=cmap, norm=norm, alpha=0.9)

    HUC02.plot(ax=ax, edgecolor='black', facecolor='none', linewidth=4)

    plt.rcParams['font.size'] = '40'

    if scn == 'RCP45_B1':
        ax.set_title(yr, size=74)
    
    if scn == 'RCP85_A2' and yr == 209900:
        ax.add_artist(ScaleBar(DIST_METERS, location='lower left', length_fraction=0.5, width_fraction=0.02, border_pad = 0.8, pad = 0.8))

    print(var.split('_')[0])

    outpath = f"../imgs/Paper2/Results/proj_dswe_mpe_maps/MULTIMODEL_{scn}_{yr}_{var}.png"

    plt.savefig(outpath, dpi=300, facecolor='w', edgecolor='w', transparent=False, bbox_inches='tight')

    plt.close()

    return


def LULC_plot_legend():

    # bounds = [-60, -40, -25, -10, -5, 5, 10, 25, 40, 60] # 
    # bounds = [-50, -40, -30, -20, -10, 10, 20, 30, 40, 50]


    # colors = ['#543005', '#8c510a', '#bf812d', '#dfc27d', '#f6e8c3', 
    #           '#f5f5f5', 
    #           '#c7eae5', '#80cdc1', '#35978f', '#01665e', '#003c30']  

    bounds = [-40, -30, -20, -10, 0, 10, 20, 30, 40]

    colors = ['#543005', '#8c510a', '#bf812d', '#dfc27d', '#f6e8c3', 
              '#c7eae5', '#80cdc1', '#35978f', '#01665e', '#003c30']


    cmap = matplotlib.colors.ListedColormap(colors)
    norm = matplotlib.colors.BoundaryNorm(bounds, len(colors), extend='both')

    plt.rcParams['font.size'] = '64'

    fig = plt.figure(figsize=(30, 1.5))
    gs = fig.add_gridspec(1,1, wspace=2,hspace=2)

    gs.update(wspace=0, hspace=0)

    ax1 = gs.subplots()

    # fig.colorbar(mpl.cm.ScalarMappable(norm=norm, cmap=cmap),
    #             cax=ax1, orientation='horizontal',
    #             label='Difference in Percent of Projected and 2001-2018 Mean\nForest-Dominated Land-Use/Land-Cover')
    
    # fig.colorbar(mpl.cm.ScalarMappable(norm=norm, cmap=cmap),
    #             cax=ax1, orientation='horizontal',
    #             label='Difference in Percent of Projected and 2001-2018 Mean\nAgricultural Land-Use/Land-Cover')

    fig.colorbar(mpl.cm.ScalarMappable(norm=norm, cmap=cmap),
                cax=ax1, orientation='horizontal',
                label='Difference in Percent of Projected and 2001-2018 Mean\nIntensive Land-Use/Land-Cover')

    return



obs_df = pd.read_csv('../data/all_data_0118_p2.csv', index_col=0)
obs_df = obs_df.reset_index()
obs_df = clean_df(obs_df, pred=False)

obs_df = obs_df.merge(shp[['HUC08','AREA']], on='HUC08')
obs_df['WATER_AREA'] = obs_df['PR_WATER'] * obs_df['AREA']
obs_df['FRST_AREA'] = obs_df['PR_NAT'] * obs_df['AREA']
obs_df['AGRI_AREA'] = obs_df['PR_AG'] * obs_df['AREA']
obs_df['INTS_AREA'] = obs_df['PR_INT'] * obs_df['AREA']


spring_obs_df = obs_df[obs_df['SEASON']=='Spring']
# avg_spring_obs = spring_obs_df.groupby('YR_SZN').sum()['FRST_AREA'].mean()

# summer_obs_df = obs_df[obs_df['SEASON']=='Summer']
# avg_summer_obs = summer_obs_df.groupby('YR_SZN').sum()['FRST_AREA'].mean()

# fall_obs_df = obs_df[obs_df['SEASON']=='Fall']
# avg_fall_obs = fall_obs_df.groupby('YR_SZN').sum()['FRST_AREA'].mean()

# winter_obs_df = obs_df[obs_df['SEASON']=='Winter']
# avg_winter_obs = winter_obs_df.groupby('YR_SZN').sum()['FRST_AREA'].mean()

obs_huc_avg = spring_obs_df.groupby('HUC08').mean()
obs_huc_avg = obs_huc_avg.reset_index()
obs_huc_avg = shp[['HUC08','geometry']].merge(obs_huc_avg, on='HUC08')

scn_rcp_lst = ['RCP45_B1', 'RCP85_B2', 'RCP85_A2']
standard_cols = ['HUC08', 'YR_SZN']

var_dct = {'FRST' : 'PR_NAT',
           'AGRI' : 'PR_AG',
           'INTS' : 'PR_INT'}

for rcpscn in scn_rcp_lst:
    print(f'{rcpscn}')
    inpath_fixed_vars = f'../data/FutureData/GCM_FORESCE_CSVs/VAR_MEANS/MULTIMODEL_{rcpscn}_VAR_MEAN.csv'
    fixed_var_df = pd.read_csv(inpath_fixed_vars, index_col=0)
    fixed_var_df = clean_df(fixed_var_df)
    fixed_var_df = fixed_var_df.loc[(fixed_var_df['YR_SZN'] == 204000) | 
                                    (fixed_var_df['YR_SZN'] == 207000) | 
                                    (fixed_var_df['YR_SZN'] == 209900)]
    
    fixed_multi_cols = standard_cols + [col for col in fixed_var_df.columns if 'MULTIMODEL' in col]

    dswe_obs_cols = [col for col in obs_df.columns if 'WATER' not in col]    

    for var in ['FRST', 'AGRI', 'INTS']:

        comp_df = obs_huc_avg[['HUC08', 'AREA', f'{var}_AREA', var_dct[var], 'geometry']].merge(
                            fixed_var_df[fixed_multi_cols], on=['HUC08'])

        comp_df[f'PROJ_{var}_AREA'] = comp_df[f'MULTIMODEL_MEAN_{var_dct[var]}'] * comp_df['AREA']

        comp_df[f'{var}_PER_ERROR'] = (comp_df[f'PROJ_{var}_AREA'] - comp_df[f'{var}_AREA']) / comp_df[f'{var}_AREA'] * 100
        comp_df[f'{var}_DIFF'] = (comp_df[f'MULTIMODEL_MEAN_{var_dct[var]}'] - comp_df[var_dct[var]]) * 100


        for yr in [204000, 207000, 209900]:

            yr_comp_df = comp_df[comp_df['YR_SZN']==yr]

            # print(f"min: {yr_comp_df['FRST_PER_ERROR'].min()}")
            # print(f"10: {yr_comp_df['FRST_PER_ERROR'].quantile(.10)}")
            # print(f"20: {yr_comp_df['FRST_PER_ERROR'].quantile(.20)}")
            # print(f"30: {yr_comp_df['FRST_PER_ERROR'].quantile(.30)}")
            # print(f"40: {yr_comp_df['FRST_PER_ERROR'].quantile(.40)}")
            # print(f"med: {yr_comp_df['FRST_PER_ERROR'].median()}")
            # print(f"60: {yr_comp_df['FRST_PER_ERROR'].quantile(.60)}")
            # print(f"70: {yr_comp_df['FRST_PER_ERROR'].quantile(.70)}")
            # print(f"80: {yr_comp_df['FRST_PER_ERROR'].quantile(.80)}")
            # print(f"90: {yr_comp_df['FRST_PER_ERROR'].quantile(.90)}")
            # print(f"max: {yr_comp_df['FRST_PER_ERROR'].max()}\n")

            # lulc_per_error_maps(yr_comp_df, rcpscn, yr, f'{var}_PER_ERROR')
            lulc_per_error_maps(yr_comp_df, rcpscn, yr, f'{var}_DIFF')

