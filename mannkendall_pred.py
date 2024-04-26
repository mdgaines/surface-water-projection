import os
import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt
import matplotlib.colors
import matplotlib as mpl
import geopandas as gpd
import pymannkendall as mk

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


def szn_per_change_info(obs_df:pd.DataFrame, CI_df:pd.DataFrame, yr:int, szn:str='Spring', huc=False):

    szn_obs = obs_df.loc[obs_df['SEASON']==szn]
    szn_obs_mean = szn_obs.groupby('HUC08').mean()
    df_szn = CI_df.loc[CI_df['YR_SZN']==yr]
    df_szn = df_szn.set_index('HUC08')
    df_szn['PER_CHANGE'] = df_szn['MEAN'].sub(szn_obs_mean['OBS_WATER'], level=0).div(szn_obs_mean['OBS_WATER'], level=0) * 100
    df_szn['DIFF'] = df_szn['MEAN'].sub(szn_obs_mean['OBS_WATER'], level=0)

    df_szn['10th_PER_CHANGE'] = df_szn['10th'].sub(szn_obs_mean['OBS_WATER'], level=0).div(szn_obs_mean['OBS_WATER'], level=0) * 100
    df_szn['90th_PER_CHANGE'] = df_szn['90th'].sub(szn_obs_mean['OBS_WATER'], level=0).div(szn_obs_mean['OBS_WATER'], level=0) * 100


    df_szn = df_szn.reset_index()

    if huc:
        df_szn = df_szn[df_szn['HUC08']==huc]

    return df_szn

def get_plus_minus(df):

    plus = (df['90th_PER_CHANGE'] - df['PER_CHANGE'])
    minus = (df['10th_PER_CHANGE'] - df['PER_CHANGE'])

    print(f'mean: {round(df.PER_CHANGE.values[0], 2)}, 90th diff: {round(plus.values[0], 2)}, 10th diff: {round(minus.values[0], 2)}')

def print_10_90(df):

    print(f"mean : {round(df.PER_CHANGE.values[0], 2)}")
    print(f"90th : {round(df['90th_PER_CHANGE'].values[0], 2)}")
    print(f"10th : {round(df['10th_PER_CHANGE'].values[0], 2)}")



def get_ci_dict(scn_lst:list, study_area:bool=False):
    
    scn_ci_dict = {}
    for scn in scn_lst:
        if study_area:
            multi_output_ci = f'../data/FutureData/GCM_FORESCE_CSVs/StudyAreaCI/MULTIMODEL_{scn}_MC_CI95.csv'
        else:
            multi_output_ci = f'../data/FutureData/GCM_FORESCE_CSVs/HUC_CI/MULTIMODEL_{scn}_MC_HUC_CI95.csv'
        ci_df = pd.read_csv(multi_output_ci, index_col=0)
        scn_ci_dict[scn] = ci_df

    return scn_ci_dict


def get_full_mk_dict(scn_lst:list, scn_ci_dict:dict):

    scn_mk_dict = {}
    for scn in scn_lst:

        outpath = f'../data/FutureData/GCM_FORESCE_CSVs/HUC_MK/MULTIMODEL_{scn}_MC_HUC_MK.csv'
        if os.path.exists(outpath):
            df = pd.read_csv(outpath, index_col=0)
            scn_mk_dict[scn] = df
            continue

        df = scn_ci_dict[scn]

        mk_res_lst = [0] * len(np.unique(df.HUC08))
        huc_lst = df['HUC08'].unique()

        for i in range(len(mk_res_lst)):
            
            temp_df = df[df['HUC08']==huc_lst[i]].sort_values('YR_SZN')
            temp_df = temp_df.set_index('YR_SZN')

            res = mk.seasonal_test(temp_df['MEAN'], period=4)

            mk_res_lst[i] = [huc_lst[i], res.trend, res.h, res.p, res.z, res.Tau, res.s, res.var_s, res.slope, res.intercept]

            # mk_savepng(np.asarray(temp_df['PR_WATER']), res, huc_lst[i])

        mk_res_df = pd.DataFrame(mk_res_lst, columns=['HUC','TREND_DIR','TREND_PRES','P_VALUE','Z','TAU','S','VAR_S','SLOPE','INTERCEPT'])
        mk_res_df.to_csv(outpath)

        scn_mk_dict[scn] = mk_res_df

    return scn_mk_dict


def get_30yr_mk_dict(scn_lst:list, scn_ci_dict:dict, study_area:bool=False):

    scn_30yr_mk_dict = {}
    for scn in scn_lst:

        df = scn_ci_dict[scn]

        if study_area:
            mk_res_lst = [0] * 3
            i=0
        else:
            mk_res_lst = [0] * len(np.unique(df.HUC08))
            huc_lst = df['HUC08'].unique()

        yrs = [2040, 2070, 2099]
        for yr in yrs:
            if yr == 2040:
                start_yr_str = '2006'
                start_yr_int = 200600
                end_yr_int = 204100
            if yr == 2070:
                start_yr_str = '2041'
                start_yr_int = 204100
                end_yr_int = 207100
            if yr == 2099:
                start_yr_str = '2071'
                start_yr_int = 207100
                end_yr_int = 210000

            if study_area:
                outpath = f'../data/FutureData/GCM_FORESCE_CSVs/StudyAreaMK/MULTIMODEL_{scn}_MC_MK.csv'
                if os.path.exists(outpath):
                    df = pd.read_csv(outpath, index_col=0)
                    scn_30yr_mk_dict[scn + str(yr)] = df
                    continue

                temp_df = df.sort_values('YR_SZN')
                temp_df = temp_df.loc[start_yr_int:end_yr_int]

                res = mk.seasonal_test(temp_df['MEAN'], period=4)

                mk_res_lst[i] = [yr, res.trend, res.h, res.p, res.z, res.Tau, res.s, res.var_s, res.slope, res.intercept]

                scn_30yr_mk_dict[scn + str(yr)] = mk_res_lst[i]
                i += 1

            else:
                outpath = f'../data/FutureData/GCM_FORESCE_CSVs/HUC_MK/MULTIMODEL_{scn}_{start_yr_str}_{yr}_MC_HUC_MK.csv'
                if os.path.exists(outpath):
                    df = pd.read_csv(outpath, index_col=0)
                    scn_30yr_mk_dict[scn + str(yr)] = df
                    continue


                for i in range(len(mk_res_lst)):
                    
                    temp_df = df[df['HUC08']==huc_lst[i]].sort_values('YR_SZN')
                    temp_df = temp_df[(temp_df['YR_SZN'] >= start_yr_int) & (temp_df['YR_SZN'] < end_yr_int)]
                    temp_df = temp_df.set_index('YR_SZN')

                    res = mk.seasonal_test(temp_df['MEAN'], period=4)

                    mk_res_lst[i] = [huc_lst[i], res.trend, res.h, res.p, res.z, res.Tau, res.s, res.var_s, res.slope, res.intercept]

                    # mk_savepng(np.asarray(temp_df['PR_WATER']), res, huc_lst[i])

                mk_res_df = pd.DataFrame(mk_res_lst, columns=['HUC','TREND_DIR','TREND_PRES','P_VALUE','Z','TAU','S','VAR_S','SLOPE','INTERCEPT'])
                mk_res_df.to_csv(outpath)

                scn_30yr_mk_dict[scn + str(yr)] = mk_res_df

        if study_area:        
            mk_res_df = pd.DataFrame(mk_res_lst, columns=['YEAR','TREND_DIR','TREND_PRES','P_VALUE','Z','TAU','S','VAR_S','SLOPE','INTERCEPT'])
            mk_res_df.to_csv(outpath)

    return scn_30yr_mk_dict






def diff_maps(yr_shp, scn, yr):

    bounds = [-0.35,-0.25,-0.15,-0.05,0.05,0.15,0.25,0.35]

    if scn == 'RCP45_B1':
        colors = ['#8d5b29', '#a8908d', '#d1c6c6', '#ffffef', '#aec6ab', '#608f6c', '#005a32']
    elif scn == 'RCP85_B2':
        colors = ['#8d5b29', '#a8908d', '#d1c6c6', '#ffffef', '#b9bcd1',  '#717db3', '#084594']
    elif scn == 'RCP85_A2':
        colors = ['#8d5b29', '#a8908d', '#d1c6c6', '#ffffef', '#e0b2b0', '#bc6675', '#91003f']
                  
    cmap = matplotlib.colors.ListedColormap(colors)
    norm = matplotlib.colors.BoundaryNorm(bounds, len(colors))

    fig, ax = plt.subplots(figsize=(20, 12))#, facecolor = "#C0C0C0")
    # ax.set_facecolor("#C0C0C0")
    plt.subplots_adjust(top = 1, bottom = 0, right = 1, left = 0, hspace = 0, wspace = 0)
    ax.set_aspect('equal')
    ax.set_axis_off()

    yr_shp.plot(column=scn+"_DIFF", ax=ax, edgecolor='black', linewidth=0.5, legend=True, cmap=cmap, norm=norm)

    yr_shp[yr_shp[f"{scn}_TREND"] == "increasing"].plot(ax=ax, hatch="//", edgecolor='black', facecolor="none")
    yr_shp[yr_shp[f"{scn}_TREND"] == "decreasing"].plot(ax=ax, hatch="\\\\", edgecolor='black', facecolor="none")

    HUC02.plot(ax=ax, edgecolor='black', facecolor='none', linewidth=3)
        
        # | (yr_shp["HUC08"] == 3020201) | \
        #    (yr_shp["HUC08"] == 3100205) | (yr_shp["HUC08"] == 6010201) | \
        #     (yr_shp["HUC08"] == 8090203)
    yr_shp[(yr_shp["HUC08"] == 3130001)].plot(ax=ax, edgecolor="#ff00ff", \
                                               facecolor="none", linewidth=2)
    
    # outpath = f"../imgs/Paper2/Results/diff_maps/MULTIMODEL_{scn}_{yr}_DIFF_MK-UpperChat.png"

    # plt.savefig(outpath, dpi=300, facecolor='w', edgecolor='w', transparent=False, pad_inches=0)

    # plt.close()

    return


def per_diff_maps(yr_shp, scn, yr, var):

    # bounds = [-50, -10, 10, 50, 100, 250, 500, 1000]
    bounds = [-50, -10, 10, 50, 100, 150, 250, 500, 1000]


    if scn == 'RCP45_B1': #green
        # colors = ['#bda997', '#ded3bd', '#ffffe3', '#c9dfbe', '#97bf89', '#679e5b', '#3c7c36', '#165b1c', '#003912']
        colors = ['#a28676', '#d0c1ac', '#ffffe3', '#cde2c3', '#9dc591', '#6ea765', '#418942', '#146a2f', '#064a26', '#0a2b10']
    elif scn == 'RCP85_B2': #blue
        # colors = ['#bda997', '#ded3bd', '#ffffe3', '#cdd6e4', '#a1b0d6', '#768ac5', '#4d66b1', '#254398', '#002078']
        colors = ['#a28676',  '#d0c1ac', '#ffffe3', '#d6dce9', '#aebad8', '#8699c7', '#5d79b5', '#2f5aa3', '#093a8c', '#001772']
    elif scn == 'RCP85_A2': #pink
        # colors = ['#bda997', '#ded3bd', '#ffffe3', '#f0d6d6', '#deaeb3', '#cb8792', '#b66072', '#9f3853', '#820634']
        colors = ['#a28676',  '#d0c1ac', '#ffffe3', '#f0d7d7', '#dfafb4', '#cd8893', '#b96174', '#a33656', '#870037', '#570015']
#  -%23a28676-%23d0c1ac-%23ffffe3-%23f0d7d7-%23dfafb4-%23cd8893-%23b96174-%23a33656-%23870037-%23570015
#  -%23a28676-%23d0c1ac-%23ffffe3-%23cde2c3-%239dc591-%236ea765-%23418942-%23146a2f-%23064a26-%230a2b10        
        
    cmap = matplotlib.colors.ListedColormap(colors)
    norm = matplotlib.colors.BoundaryNorm(bounds, len(colors), extend='both')

    fig, ax = plt.subplots(figsize=(20, 12))#, facecolor = "#C0C0C0")
    # ax.set_facecolor("#C0C0C0")
    # plt.subplots_adjust(top = 1, bottom = 0, right = 1, left = 0, hspace = 0, wspace = 0)
    ax.set_aspect('equal')
    ax.set_axis_off()

    yr_shp.plot(column=var, ax=ax, edgecolor='black', linewidth=0.5, cmap=cmap, norm=norm, alpha=0.9) #, legend=True

    # yr_shp[yr_shp[f"{scn}_TREND"] == "increasing"].plot(ax=ax, hatch="//", edgecolor='black', facecolor="none")
    # yr_shp[yr_shp[f"{scn}_TREND"] == "decreasing"].plot(ax=ax, hatch="\\\\", edgecolor='black', facecolor="none")

    yr_shp[yr_shp[f"10-90"] == "agree"].plot(ax=ax, hatch="xx", edgecolor='black', facecolor="none")


    HUC02.plot(ax=ax, edgecolor='black', facecolor='none', linewidth=4)

        #     | (yr_shp["HUC08"] == 3020201) | \
        #    (yr_shp["HUC08"] == 3100205) | (yr_shp["HUC08"] == 6010201) | \
        #     (yr_shp["HUC08"] == 8090203)
    # yr_shp[(yr_shp["HUC08"] == 3130001)].plot(ax=ax, edgecolor="#ff00ff", \
    #                                            facecolor="none", linewidth=2)
    yr_shp[(yr_shp["HUC08"] == 3020201) | \
            (yr_shp["HUC08"] == 6010201) | \
            (yr_shp["HUC08"] == 8090203)].plot(ax=ax, edgecolor="#ff00ff", \
                                                facecolor="none", linewidth=2)  
    
    if scn == 'RCP45_B1':
        ax.set_title(yr, size=74)
    
    if scn == 'RCP85_A2' and yr == '2099':
        ax.add_artist(ScaleBar(DIST_METERS, location='lower left', length_fraction=0.5, width_fraction=0.02, border_pad = 0.8, pad = 0.8))


    outpath = f"../imgs/Paper2/Results/diff_maps/MULTIMODEL_{scn}_{yr}_{var}_10-90_agree.png"

    plt.savefig(outpath, dpi=300, facecolor='w', edgecolor='w', transparent=False, pad_inches=0)

    plt.close()

    return


def plot_10th_90th_agreement(yr_shp, scn, yr):


    if scn == 'RCP45_B1': #green
        agreement_palette = {'Positive':'#005a32', 
                             'Negative':'#a28676', 
                             'Disagree':'#ffffef'}
    elif scn == 'RCP85_B2': #blue
        agreement_palette = {'Positive':'#084594', 
                             'Negative':'#a28676', 
                             'Disagree':'#ffffef'}
    elif scn == 'RCP85_A2': #pink
        agreement_palette = {'Positive':'#91003f', 
                             'Negative':'#a28676', 
                             'Disagree':'#ffffef'}
    
    yr_shp['color'] = agreement_palette['Disagree']
    yr_shp.loc[(yr_shp['10th_PER_CHANGE'] >=0) & (yr_shp['90th_PER_CHANGE'] >=0), 'color'] = agreement_palette['Positive']
    yr_shp.loc[(yr_shp['10th_PER_CHANGE'] <0) & (yr_shp['90th_PER_CHANGE'] < 0), 'color'] = agreement_palette['Negative']

    fig, ax = plt.subplots(figsize=(20, 12))

    yr_shp.plot(ax=ax, color = yr_shp['color'],  edgecolor='black', linewidth=0.5)
    ax.set_axis_off()
    # ax.add_artist(ScaleBar(DIST_METERS, location='lower left', length_fraction=0.5, width_fraction=0.02, border_pad = 0.8, pad = 0.8))

    HUC02.plot(ax=ax, edgecolor='black', facecolor='none', linewidth=3)

    
    # if scn == 'RCP45_B1':
    ax.set_title(yr, size=74)
    
    # if scn == 'RCP85_A2' and yr == '2099':
    # ax.add_artist(ScaleBar(DIST_METERS, location='lower left', length_fraction=0.5, width_fraction=0.02, border_pad = 0.8, pad = 0.8))

    yr_shp[(yr_shp["HUC08"] == 3020201) | \
        (yr_shp["HUC08"] == 6010201) | \
        (yr_shp["HUC08"] == 8090203)].plot(ax=ax, edgecolor="#ff00ff", \
                                            facecolor="none", linewidth=2) 

    agreement_dict = {}
    for ctype, data in yr_shp.groupby('color'):
        print(ctype)
        # print(data)
        # label=ctype
        agreement_dict[ctype + '_03'] = yr_shp.loc[(yr_shp['HUC02']=='03') & (yr_shp['color'] == ctype)].count()['HUC02']
        agreement_dict[ctype + '_06'] = yr_shp.loc[(yr_shp['HUC02']=='06') & (yr_shp['color'] == ctype)].count()['HUC02']
        agreement_dict[ctype + '_08'] = yr_shp.loc[(yr_shp['HUC02']=='08') & (yr_shp['color'] == ctype)].count()['HUC02']

        color = ctype
        data.plot(color=color, ax=ax, edgecolor='black', linewidth=0.5)
    
    row_labels = ['Positive', 'Negative', 'Disagree']
    col_labels = ['South Atlantic Gulf', ' Tennessee', 'Lower Mississippi']
    table_vals = [[agreement_dict[f'{agreement_palette["Positive"]}_03'], agreement_dict[f'{agreement_palette["Positive"]}_06'], agreement_dict[f'{agreement_palette["Positive"]}_08']],
                    [agreement_dict[f'{agreement_palette["Negative"]}_03'], agreement_dict[f'{agreement_palette["Negative"]}_06'], agreement_dict[f'{agreement_palette["Negative"]}_08']],
                    [agreement_dict[f'{agreement_palette["Disagree"]}_03'], agreement_dict[f'{agreement_palette["Disagree"]}_06'], agreement_dict[f'{agreement_palette["Disagree"]}_08']]]
    info_table = plt.table(bbox=[.1, .02, .49, .25],
                            cellText=table_vals, 
                            cellLoc='center',
                            rowLabels=row_labels, 
                            colLabels=col_labels, 
                            rowColours=[agreement_palette["Positive"], '#a28676','#ffffef'])
    plt.text(-92.5, 28, 
                '10th and 90th Percentile Projection HUC08 Agreement Counts\nper Water Resource Region',
                multialignment='center')       

    outpath = f"../imgs/Paper2/Results/diff_maps/MULTIMODEL_{scn}_{yr}_10-90_agree_MK.png"

    plt.savefig(outpath, dpi=300, facecolor='w', edgecolor='w', transparent=False, pad_inches=0)

    plt.close()


def plot_legend():

    bounds = [-50, -10, 10, 50, 100, 150, 250, 500, 1000]

    green_colors = ['#a28676', '#d0c1ac', '#ffffe3', '#cde2c3', '#9dc591', '#6ea765', '#418942', '#146a2f', '#064a26', '#0a2b10']
    blue_colors = ['#a28676',  '#d0c1ac', '#ffffe3', '#d6dce9', '#aebad8', '#8699c7', '#5d79b5', '#2f5aa3', '#093a8c', '#001772']
    pink_colors = ['#a28676',  '#d0c1ac', '#ffffe3', '#f0d7d7', '#dfafb4', '#cd8893', '#b96174', '#a33656', '#870037', '#570015']
        
    green_cmap = matplotlib.colors.ListedColormap(green_colors)
    blue_cmap = matplotlib.colors.ListedColormap(blue_colors)
    pink_cmap = matplotlib.colors.ListedColormap(pink_colors)

    norm = matplotlib.colors.BoundaryNorm(bounds, len(green_colors), extend='both')
    
    plt.rcParams['font.size'] = '28'

    fig = plt.figure(figsize=(20, 2))#,ncols=1,nrows=3)#, layout='constrained')
    gs = fig.add_gridspec(3,1, wspace=2,hspace=2)

    # gs = gridspec.Gridspec(3,1)
    gs.update(wspace=0, hspace=0)

    ax1, ax2, ax3 = gs.subplots()
    ax_lst = [ax1, ax2, ax3]

    fig.colorbar(mpl.cm.ScalarMappable(norm=norm, cmap=green_cmap),
                cax=ax1, orientation='horizontal')

    fig.colorbar(mpl.cm.ScalarMappable(norm=norm, cmap=blue_cmap),
                cax=ax2, orientation='horizontal')

    for a in ax_lst:
        a.set_xticklabels([])
        a.set_yticklabels([])

    fig.colorbar(mpl.cm.ScalarMappable(norm=norm, cmap=pink_cmap),
                cax=ax3, orientation='horizontal',
                label='Percent Change from 2001-2018 Mean Surface Water Area')
    
    return


def mk_maps(yr_shp, scn, yr, var):


    fig, ax = plt.subplots(figsize=(20, 12))#, facecolor = "#C0C0C0")
    # ax.set_facecolor("#C0C0C0")
    # plt.subplots_adjust(top = 1, bottom = 0, right = 1, left = 0, hspace = 0, wspace = 0)
    ax.set_aspect('equal')
    ax.set_axis_off()

    # yr_shp.plot(column=var, ax=ax, edgecolor='black', linewidth=0.5, cmap=cmap, norm=norm, alpha=0.9) #, legend=True

    # #61aea3, #efefef

    color_dict = {'RCP45_B1':'#005a32', #green
                  'RCP85_B2':'#084594', #blue 
                  'RCP85_A2': '#91003f'} #pink

    yr_shp[yr_shp[f"{scn}_TREND"] == "increasing"].plot(ax=ax, edgecolor='black', facecolor=color_dict[scn]) # hatch="//", 
    yr_shp[yr_shp[f"{scn}_TREND"] == "decreasing"].plot(ax=ax, edgecolor='black', facecolor="#a28676") # hatch="\\\\",  
    yr_shp[yr_shp[f"{scn}_TREND"] == "no trend"].plot(ax=ax, edgecolor='black', facecolor="#ffffe3") # hatch="\\\\",  

    # yr_shp[yr_shp[f"10-90"] == "agree"].plot(ax=ax, hatch="xx", edgecolor='black', facecolor="none")


    HUC02.plot(ax=ax, edgecolor='black', facecolor='none', linewidth=4)

        #     | (yr_shp["HUC08"] == 3020201) | \
        #    (yr_shp["HUC08"] == 3100205) | (yr_shp["HUC08"] == 6010201) | \
        #     (yr_shp["HUC08"] == 8090203)
    # yr_shp[(yr_shp["HUC08"] == 3130001)].plot(ax=ax, edgecolor="#ff00ff", \
    #                                            facecolor="none", linewidth=2)
    yr_shp[(yr_shp["HUC08"] == 3020201) | \
            (yr_shp["HUC08"] == 6010201) | \
            (yr_shp["HUC08"] == 8090203)].plot(ax=ax, edgecolor="#ff00ff", \
                                                facecolor="none", linewidth=2)  
    
    if scn == 'RCP45_B1':
        ax.set_title(yr, size=74)
    
    if scn == 'RCP85_A2' and yr == '2099':
        ax.add_artist(ScaleBar(DIST_METERS, location='lower left', length_fraction=0.5, width_fraction=0.02, border_pad = 0.8, pad = 0.8))


    outpath = f"../imgs/Paper2/Results/diff_maps/MULTIMODEL_{scn}_{yr}_{var}_color_MK.png"

    plt.savefig(outpath, dpi=300, facecolor='w', edgecolor='w', transparent=False, pad_inches=0)

    plt.close()

    return


shp = gpd.read_file('../data/Shapefiles/HUC08/HUC08_paper2/HUC08_paper2.shp')
shp = shp[['huc8','areasqkm', 'geometry']]
shp['HUC08'] = shp['huc8'].apply(lambda x : int(x))
shp['AREA'] = shp['areasqkm'].apply(lambda x : float(x))

obs_df = pd.read_csv('../data/all_data_0118_p2.csv', index_col=0)
obs_df = clean_df(obs_df, pred=False)
# obs_2018_spring = obs_df[(obs_df["YR_SZN"]==201800)]

# shp = shp.merge(obs_2018_spring[["HUC08", "OBS_WATER"]], on="HUC08")


scn_lst = ['RCP45_B1', 'RCP85_B2', 'RCP85_A2']#, \
        #    'RCP85_B1', 'RCP45_B2', 'RCP45_A2', 'RCP45_A1B', 'RCP85_A1B']

scn_ci_dict = get_ci_dict(scn_lst=scn_lst)
scn_30yr_mk_dict = get_30yr_mk_dict(scn_lst=scn_lst, scn_ci_dict=scn_ci_dict)

yr_shp_dict = {}
min_val = 0
max_val = 0
for scn in scn_lst:

    df = scn_ci_dict[scn]
    yrs = [2040, 2070, 2099]

    for yr in yrs:
        mk_df = scn_30yr_mk_dict[scn + str(yr)]
        # print(f"{scn} {yr}\n{mk_df.TREND_DIR.value_counts()}\n")
        mk_df = mk_df.rename(columns={"HUC":"HUC08"})

        yr_szn_per_change_df = szn_per_change_info(obs_df, df, yr=yr*100, szn='Spring')

        # yr_df = df[df['YR_SZN'] == yr*100]
        yr_shp = shp.merge(yr_szn_per_change_df[["HUC08", "MEAN", "DIFF", "PER_CHANGE", "10th_PER_CHANGE", "90th_PER_CHANGE"]], on='HUC08')
        yr_shp = yr_shp.rename(columns={"MEAN":scn})
        yr_shp = yr_shp.rename(columns={"DIFF":scn+"_DIFF"})
        yr_shp = yr_shp.rename(columns={"PER_CHANGE":scn+"_PER_CHANGE"})

        yr_shp = yr_shp.merge(mk_df[["HUC08", "TREND_DIR"]])
        yr_shp = yr_shp.rename(columns={"TREND_DIR":scn+"_TREND"})
        yr_shp['HUC02'] = yr_shp['HUC08'].apply(lambda x: str(x)[0].zfill(2))

        yr_shp['10-90'] = 'disagree'
        yr_shp.loc[((yr_shp['10th_PER_CHANGE'] >=0) & (yr_shp['90th_PER_CHANGE'] >=0)) | 
                   ((yr_shp['10th_PER_CHANGE'] <0) & (yr_shp['90th_PER_CHANGE'] < 0)),
                   '10-90'] = 'agree'

        # per_diff_maps(yr_shp, scn, yr)
        # yr_shp[scn + "_PER_CHANGE"].hist(legend=False, alpha=0.4, bins=50)
        # yr_shp[scn + "_PER_CHANGE"].hist(legend=False, alpha=0.4, bins=50, range=(-80,1000))

        # print(f"{scn} {yr}\nmin :{yr_shp[scn + '_PER_CHANGE'].min()}\
        #     \nmed :{yr_shp[scn + '_PER_CHANGE'].median()}\
        #     \nmax :{yr_shp[scn + '_PER_CHANGE'].max()}\n")

        # print(f'min:\t{yr_shp[scn+"_PER_CHANGE"].min()}')
        # print(f'10th:\t{yr_shp[scn+"_PER_CHANGE"].quantile(.1)}')
        # print(f'20th:\t{yr_shp[scn+"_PER_CHANGE"].quantile(.2)}')
        # print(f'30th:\t{yr_shp[scn+"_PER_CHANGE"].quantile(.3)}')
        # print(f'40th:\t{yr_shp[scn+"_PER_CHANGE"].quantile(.4)}')
        # print(f'50th:\t{yr_shp[scn+"_PER_CHANGE"].quantile(.5)}')
        # print(f'60th:\t{yr_shp[scn+"_PER_CHANGE"].quantile(.6)}')
        # print(f'70th:\t{yr_shp[scn+"_PER_CHANGE"].quantile(.7)}')
        # print(f'80th:\t{yr_shp[scn+"_PER_CHANGE"].quantile(.8)}')
        # print(f'90th:\t{yr_shp[scn+"_PER_CHANGE"].quantile(.9)}')
        # print(f'max:\t{yr_shp[scn+"_PER_CHANGE"].max()}')

        # if yr_shp[scn + "_DIFF"].max() > max_val:
        #     max_val = yr_shp[scn + "_DIFF"].max()
        #     print(f"max: {scn} {yr}\n\t{max_val}\n")
        # if yr_shp[scn + "_DIFF"].min() < min_val:
        #     min_val = yr_shp[scn + "_DIFF"].min()
        #     print(f"min: {scn} {yr}\n\t{min_val}\n")
        
        yr_shp_dict[f"{scn}_{yr}"] = yr_shp

        diff_maps(yr_shp, scn, yr)

all_srs = pd.Series()
for key in yr_shp_dict.keys():
    scn = '_'.join(key.split('_')[0:2])
    all_srs = all_srs.append(yr_shp_dict[key][scn+"_PER_CHANGE"])

print(f'min:\t{all_srs.min()}')
print(f'10th:\t{all_srs.quantile(.1)}')
print(f'20th:\t{all_srs.quantile(.2)}')
print(f'30th:\t{all_srs.quantile(.3)}')
print(f'40th:\t{all_srs.quantile(.4)}')
print(f'50th:\t{all_srs.quantile(.5)}')
print(f'60th:\t{all_srs.quantile(.6)}')
print(f'70th:\t{all_srs.quantile(.7)}')
print(f'80th:\t{all_srs.quantile(.8)}')
print(f'90th:\t{all_srs.quantile(.9)}')
print(f'max:\t{all_srs.max()}')

print(f'mean + std:\t{all_srs.mean()+all_srs.std()}')
print(f'      mean:\t{all_srs.mean()}')
print(f'mean - std:\t{all_srs.mean()-all_srs.std()}')



for key in yr_shp_dict.keys():
    key_info = key.split('_')
    scn = '_'.join(key_info[0:2])
    yr = key_info[2]
    yr_shp = yr_shp_dict[key]

    # if yr == '2099':
    # per_diff_maps(yr_shp, scn, yr, var=f'{scn}_PER_CHANGE')
    mk_maps(yr_shp, scn, yr, var=f'{scn}_PER_CHANGE')
    # per_diff_maps(yr_shp, scn, yr, var=f'10th_PER_CHANGE')
    # per_diff_maps(yr_shp, scn, yr, var=f'90th_PER_CHANGE')
    # plot_10th_90th_agreement(yr_shp, scn, yr)

for key in yr_shp_dict.keys():
    yr_shp = yr_shp_dict[key]
    print(f'\n{key}:')
    print(f"10-90 agree: {len(yr_shp[yr_shp['10-90']=='agree'])}")
    print(f"10-90 disagree: {len(yr_shp[yr_shp['10-90']=='disagree'])}")


# huc_lst = [3130001, 3020201, 3100205, 6010201, 8090203]
# huc_names = ['Upper Chattahoochee', 'Upper Neuse', 'Hillsborough', 'Watts Bar Lake', 'Eastern Louisiana Coastal']
# for i in range(len(huc_lst)):
    


################ STUDY AREA MK ####################

scn_lst = ['RCP45_B1', 'RCP85_B2', 'RCP85_A2', \
           'RCP85_B1', 'RCP45_B2', 'RCP45_A2', 'RCP45_A1B', 'RCP85_A1B']

scn_ci_dict = get_ci_dict(scn_lst=scn_lst, study_area=True)

scn_30yr_mk_dict = get_30yr_mk_dict(scn_lst=scn_lst, scn_ci_dict=scn_ci_dict, study_area=True)




############# HUC-wise MK ################

scn_lst = ['RCP45_B1', 'RCP85_B2', 'RCP85_A2', \
           'RCP85_B1', 'RCP45_B2', 'RCP45_A2', 'RCP45_A1B', 'RCP85_A1B']

scn_ci_dict = get_ci_dict(scn_lst=scn_lst, study_area=False)

scn_mk_dict = get_full_mk_dict(scn_lst, scn_ci_dict)

scn_mk_inc_dict = {}
scn_mk_dec_dict = {}

for scn in scn_lst:
    df = scn_mk_dict[scn]
    scn_mk_inc_dict[scn] = df[df.TREND_DIR == 'increasing'].HUC.count()
    scn_mk_dec_dict[scn] = df[df.TREND_DIR == 'decreasing'].HUC.count()

dict(sorted(scn_mk_inc_dict.items(), key=lambda item: item[1]))
dict(sorted(scn_mk_dec_dict.items(), key=lambda item: item[1]))


scn_30yr_huc_mk_dict = get_30yr_mk_dict(scn_lst, scn_ci_dict, study_area=False)

scn_30yr_mk_inc_dict = {}
scn_30yr_mk_dec_dict = {}

init = True
count = 0
scn_30yr_dir_df_dict = {}
for key in scn_30yr_huc_mk_dict.keys():
    df = scn_30yr_huc_mk_dict[key]
    scn_30yr_mk_inc_dict[key] = df[df.TREND_DIR == 'increasing'].HUC.count()
    scn_30yr_mk_dec_dict[key] = df[df.TREND_DIR == 'decreasing'].HUC.count()

    if init:
        time_df = df[['HUC', 'TREND_DIR']]
        init=False
    else:
        time_df = time_df.merge(df[['HUC', 'TREND_DIR']], on='HUC')
    time_df = time_df.rename(columns={'TREND_DIR':f'TREND_{key}'})
    count += 1

    if count == 3:
        scn_30yr_dir_df_dict[key[:-4]] = time_df
        del(time_df)
        init = True
        count = 0


dict(sorted(scn_30yr_mk_inc_dict.items(), key=lambda item: item[1]))
dict(sorted(scn_30yr_mk_dec_dict.items(), key=lambda item: item[1]))


for key in scn_30yr_dir_df_dict.keys():
    print(key)
    print((scn_30yr_dir_df_dict[key].iloc[:,1:4].eq(scn_30yr_dir_df_dict[key].iloc[:,1], axis=0)).all(axis=1).value_counts())
    print()


########################################
######  Scenario Analysis (Fig 6) ######
########################################

def get_avg_szn_range(df, col_name):
    running_sum = 0
    for i in range(0, len(df.index), 4):
        min_val = df.iloc[i:i+4][col_name].min()
        max_val = df.iloc[i:i+4][col_name].max()

        range_val = max_val - min_val

        running_sum += range_val
    return running_sum/len(df.index)



obs_df = pd.read_csv('../data/all_data_0118_p2.csv', index_col=0)
obs_df = clean_df(obs_df, pred=False)

huc_dict ={3020201 : 'Upper Neuse', 
           6010201 : 'Watts Bar Lake',
           8090203 : 'Eastern Lousisiana Coastal'}

for huc8 in [3020201, 6010201, 8090203]:
    print(f'\n\n{huc_dict[huc8]}')
    # huc8 = 3020201

    for scn in ['RCP85_A2', 'RCP85_B2', 'RCP45_B1']:
        print(f'\n{scn}')
        scn_df = scn_ci_dict[scn]

        df_2099 = szn_per_change_info(obs_df, scn_df, 2099*100, 'Spring', huc8)
        # get_plus_minus(df_2099)
        print_10_90(df_2099)

        near_df = scn_df[(scn_df['HUC08']==huc8) & (scn_df['YR_SZN'] < 204100)]
        print(f"'06-'40 avg seasonal range : {get_avg_szn_range(near_df, 'MEAN') * 100} %")
        mid_df = scn_df[(scn_df['HUC08']==huc8) & (scn_df['YR_SZN'] >= 204100) & (scn_df['YR_SZN'] < 207100)]
        print(f"'41-'70 avg seasonal range : {get_avg_szn_range(mid_df, 'MEAN') * 100} %")
        far_df = scn_df[(scn_df['HUC08']==huc8) & (scn_df['YR_SZN'] >= 207100)]
        print(f"'71-'99 avg seasonal range : {get_avg_szn_range(far_df, 'MEAN') * 100} %")
        obs_huc_df = obs_df[obs_df['HUC08']==huc8]
        print(f"'01-'18 OBD avg seasonal range : {get_avg_szn_range(obs_huc_df, 'OBS_WATER') * 100} %")


df_2050 = szn_per_change_info(obs_df, scn_ci_dict['RCP85_B2'], 2050*100, 'Spring', huc8)
get_plus_minus(df_2050)

df_2050 = szn_per_change_info(obs_df, scn_ci_dict['RCP45_B1'], 2050*100, 'Spring', huc8)
get_plus_minus(df_2050)

df_2099 = szn_per_change_info(obs_df, scn_ci_dict['RCP45_B1'], 2099*100, 'Spring', huc8)





########################################
######  Scenario Analysis (Fig 7) ######
########################################

scn_30yr_lst = ['RCP45_B12040', 'RCP45_B12070', 'RCP45_B12099', \
                'RCP85_B22040', 'RCP85_B22070', 'RCP85_B22099', \
                'RCP85_A22040', 'RCP85_A22070', 'RCP85_A22099']

scn_huc2_30yr_mk_inc_dict = {}
scn_huc2_30yr_mk_dec_dict = {}

for key in scn_30yr_lst:
    scn_30yr_huc_mk_df = scn_30yr_huc_mk_dict[key]
    scn_30yr_huc_mk_df['HUC02'] = scn_30yr_huc_mk_df['HUC'].apply(lambda x: str(x)[0].zfill(2))

    for huc2 in ['03', '06', '08']:
        df = scn_30yr_huc_mk_df[scn_30yr_huc_mk_df['HUC02'] == huc2]

        scn_huc2_30yr_mk_inc_dict[key + f'_{huc2}'] = df[df.TREND_DIR == 'increasing'].HUC.count()
        scn_huc2_30yr_mk_dec_dict[key + f'_{huc2}'] = df[df.TREND_DIR == 'decreasing'].HUC.count()


dict(sorted(scn_huc2_30yr_mk_inc_dict.items(), key=lambda item: item[1]))
dict(sorted(scn_huc2_30yr_mk_dec_dict.items(), key=lambda item: item[1]))


scn_huc2_30yr_min_dict = {}
scn_huc2_30yr_max_dict = {}

all_vars_dict = {}

for key in yr_shp_dict.keys():
    yr_shp_df = yr_shp_dict[key]

    for huc2 in ['03', '06', '08']:
        df = yr_shp_df[yr_shp_df['HUC02'] == huc2]
        # scn_huc2_30yr_min_dict[key+ f'_{huc2}'] = yr_shp_df[f'{key[:-5]}_PER_CHANGE'].min()
        # scn_huc2_30yr_max_dict[key+ f'_{huc2}'] = yr_shp_df[f'{key[:-5]}_PER_CHANGE'].max()

        all_vars_dict[key+ f'_{huc2}'] = [df[f'{key[:-5]}_PER_CHANGE'].min(), 
                                          df[f'{key[:-5]}_PER_CHANGE'].max(), 
                                          df[df[f'{key[:-5]}_TREND'] == 'increasing'].HUC08.count(),
                                          df[df[f'{key[:-5]}_TREND'] == 'decreasing'].HUC08.count(),
                                          df[df[f'{key[:-5]}_PER_CHANGE'] >= 0].HUC08.count(),
                                          df[df[f'{key[:-5]}_PER_CHANGE'] < 0].HUC08.count()]
        

all_vars_df = pd.DataFrame.from_dict(all_vars_dict, 'index', columns=['MIN_PER_CHANGE', 'MAX_PER_CHANGE', 
                                                                      'MK_INC', 'MK_DEC',
                                                                      'POS_CHANGE_TOTAL', 'NEG_CHANGE_TOTAL'])
all_vars_df = all_vars_df.reset_index()
all_vars_df['SCENARIO'] = all_vars_df['index'].apply(lambda x: str(x)[0:8])
all_vars_df['YEAR'] = all_vars_df['index'].apply(lambda x: str(x)[9:13])
all_vars_df['HUC02'] = all_vars_df['index'].apply(lambda x: str(x)[14:])

all_vars_df['PER_MK_INC'] = 0
all_vars_df.loc[all_vars_df['HUC02']=='03','PER_MK_INC'] = all_vars_df.loc[all_vars_df['HUC02']=='03','MK_INC'] / 203 * 100
all_vars_df.loc[all_vars_df['HUC02']=='06','PER_MK_INC'] = all_vars_df.loc[all_vars_df['HUC02']=='06','MK_INC'] / 32 * 100
all_vars_df.loc[all_vars_df['HUC02']=='08','PER_MK_INC'] = all_vars_df.loc[all_vars_df['HUC02']=='08','MK_INC'] / 82 * 100


all_vars_df['PER_MK_DEC'] = 0
all_vars_df.loc[all_vars_df['HUC02']=='03','PER_MK_DEC'] = all_vars_df.loc[all_vars_df['HUC02']=='03','MK_DEC'] / 203 * 100
all_vars_df.loc[all_vars_df['HUC02']=='06','PER_MK_DEC'] = all_vars_df.loc[all_vars_df['HUC02']=='06','MK_DEC'] / 32 * 100
all_vars_df.loc[all_vars_df['HUC02']=='08','PER_MK_DEC'] = all_vars_df.loc[all_vars_df['HUC02']=='08','MK_DEC'] / 82 * 100

all_vars_df['RANGE'] = all_vars_df['MAX_PER_CHANGE'] - all_vars_df['MIN_PER_CHANGE']


#### Overall info (across all 9 maps in Fig 7) ####

def get_scn_info(df, scn='all', col='SCENARIO'):
    
    if scn != 'all':
        df = df[df[col]==scn]
        print(f'For all {scn}:\n')
    else:
        print('For all focal scnarios and time steps:')

    min_min_per_change = df[df['MIN_PER_CHANGE'] == df['MIN_PER_CHANGE'].min()][['index', 'MIN_PER_CHANGE']]
    max_max_per_change = df[df['MAX_PER_CHANGE'] == df['MAX_PER_CHANGE'].max()][['index', 'MAX_PER_CHANGE']]

    max_range = df[df['RANGE'] == df['RANGE'].max()][['index', 'RANGE']]
    min_range = df[df['RANGE'] == df['RANGE'].min()][['index', 'RANGE']]

    max_per_mk_inc = df[df['PER_MK_INC'] == df['PER_MK_INC'].max()][['index', 'PER_MK_INC']]
    max_per_mk_dec = df[df['PER_MK_DEC'] == df['PER_MK_DEC'].max()][['index', 'PER_MK_DEC']]


    max_min_per_change = df[df['MIN_PER_CHANGE'] == df['MIN_PER_CHANGE'].max()][['index', 'MIN_PER_CHANGE']]
    min_max_per_change = df[df['MAX_PER_CHANGE'] == df['MAX_PER_CHANGE'].min()][['index', 'MAX_PER_CHANGE']]

    min_per_mk_inc = df[df['PER_MK_INC'] == df['PER_MK_INC'].min()][['index', 'PER_MK_INC']]
    min_per_mk_dec = df[df['PER_MK_DEC'] == df['PER_MK_DEC'].min()][['index', 'PER_MK_DEC']]

    print(f'\nMIN minimum mean percent change:\n{min_min_per_change}')
    print(f'\nMAX maximum mean percent change:\n{max_max_per_change}')

    print(f'\nMAX range in mean percent change:\n{max_range}')
    print(f'\nMIN range in mean percent change:\n{min_range}')

    print(f'\nMAX percent Mann_kendall increase :\n{max_per_mk_inc}')
    print(f'\nMAX percent Mann_kendall decrease :\n{max_per_mk_dec}')

    print(f'\nMAX minimum mean percent change:\n{max_min_per_change}')
    print(f'\nMIN maximum mean percent change:\n{min_max_per_change}')
    print(f'\nMIN percent Mann_kendall increase :\n{min_per_mk_inc}')
    print(f'\nMIN percent Mann_kendall decrease :\n{min_per_mk_dec}')

    return



all_vars_df[all_vars_df['MIN_PER_CHANGE'] == all_vars_df['MIN_PER_CHANGE'].min()]
all_vars_df[all_vars_df['MAX_PER_CHANGE'] == all_vars_df['MAX_PER_CHANGE'].max()]

all_vars_df[all_vars_df['PER_MK_INC'] == all_vars_df['PER_MK_INC'].max()]
all_vars_df[all_vars_df['PER_MK_DEC'] == all_vars_df['PER_MK_DEC'].max()]

# all_vars_df[all_vars_df['MK_INC'] == all_vars_df['MK_INC'].max()]
# all_vars_df[all_vars_df['MK_DEC'] == all_vars_df['MK_DEC'].max()]


all_vars_df[all_vars_df['MIN_PER_CHANGE'] == all_vars_df['MIN_PER_CHANGE'].max()]
all_vars_df[all_vars_df['MAX_PER_CHANGE'] == all_vars_df['MAX_PER_CHANGE'].min()]

all_vars_df[all_vars_df['PER_MK_INC'] == all_vars_df['PER_MK_INC'].min()]
all_vars_df[all_vars_df['PER_MK_DEC'] == all_vars_df['PER_MK_DEC'].min()]


