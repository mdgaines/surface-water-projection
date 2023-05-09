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



scn_ci_dict = {}
scn_lst = ['RCP45_B1', 'RCP85_B2', 'RCP85_A2', \
           'RCP85_B1', 'RCP45_B2', 'RCP45_A2', 'RCP45_A1B', 'RCP85_A1B']
for scn in scn_lst:
    multi_output_ci = f'../data/FutureData/GCM_FORESCE_CSVs/HUC_CI/MULTIMODEL_{scn}_MC_HUC_CI95.csv'
    ci_df = pd.read_csv(multi_output_ci, index_col=0)
    scn_ci_dict[scn] = ci_df

scn_mk_dict = {}
for scn in scn_lst:

    outpath = f'../data/FutureData/GCM_FORESCE_CSVs/HUC_MK/MULTIMODEL_{scn}_MC_HUC_MK.csv'
    if os.path.exists(outpath):
        df = pd.read_csv(outpath, index_col=0)
        scn_mk_dict[scn] = df

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



scn_30yr_mk_dict = {}
for scn in scn_lst:

    df = scn_ci_dict[scn]

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



import matplotlib.colors



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

    yr_shp.plot(column=scn+"_DIFF", ax=ax, edgecolor='black', linewidth=0.5, legend=False, cmap=cmap, norm=norm)

    yr_shp[yr_shp[f"{scn}_TREND"] == "increasing"].plot(ax=ax, hatch="///", edgecolor='black', facecolor="none")
    yr_shp[yr_shp[f"{scn}_TREND"] == "decreasing"].plot(ax=ax, hatch="\\\\\\", edgecolor='black', facecolor="none")

    huc02.plot(ax=ax, edgecolor='black', facecolor='none', linewidth=3)

    yr_shp[(yr_shp["HUC08"] == 3130001) | (yr_shp["HUC08"] == 3020201) | \
           (yr_shp["HUC08"] == 3100205) | (yr_shp["HUC08"] == 6010201) | \
            (yr_shp["HUC08"] == 8090203)].plot(ax=ax, edgecolor="#ff00ff", \
                                               facecolor="none", linewidth=2)
    
    outpath = f"../imgs/Paper2/Results/diff_maps/MULTIMODEL_{scn}_{yr}_DIFF_MK.png"

    plt.savefig(outpath, dpi=300, facecolor='w', edgecolor='w', transparent=False, pad_inches=0)

    plt.close()

    return




huc02 = gpd.read_file('../data/Shapefiles/HUC02/HUC02_clean_paper2/HUC02_clean_paper2.shp')

shp = gpd.read_file('../data/Shapefiles/HUC08/HUC08_paper2/HUC08_paper2.shp')
shp = shp[['huc8','areasqkm', 'geometry']]
shp['HUC08'] = shp['huc8'].apply(lambda x : int(x))
shp['AREA'] = shp['areasqkm'].apply(lambda x : float(x))

obs_df = pd.read_csv('../data/all_data_0118_p2.csv', index_col=0)
obs_df = clean_df(obs_df, pred=False)
obs_2018_spring = obs_df[(obs_df["YR_SZN"]==201800)]

shp = shp.merge(obs_2018_spring[["HUC08", "OBS_WATER"]], on="HUC08")

yr_shp_dict = {}
min_val = 0
max_val = 0
for scn in scn_lst:

    df = scn_ci_dict[scn]
    yrs = [2040, 2070, 2099]

    for yr in yrs:
        mk_df = scn_30yr_mk_dict[scn + str(yr)]
        print(f"{scn} {yr}\n{mk_df.TREND_DIR.value_counts()}\n")
        mk_df = mk_df.rename(columns={"HUC":"HUC08"})
        yr_df = df[df['YR_SZN'] == yr*100]
        yr_shp = shp.merge(yr_df[["HUC08", "MEAN"]], on='HUC08')
        yr_shp = yr_shp.rename(columns={'MEAN':scn})
        yr_shp = yr_shp.merge(mk_df[["HUC08", "TREND_DIR"]])
        yr_shp = yr_shp.rename(columns={'TREND_DIR':scn+"_TREND"})
        # calc diff
        yr_shp[scn + "_DIFF"] = yr_shp[scn] - yr_shp["OBS_WATER"]
        # yr_shp[scn + "_DIFF"].hist(legend=True, alpha=0.4, bins=50)

        # if yr_shp[scn + "_DIFF"].max() > max_val:
        #     max_val = yr_shp[scn + "_DIFF"].max()
        #     print(f"max: {scn} {yr}\n\t{max_val}\n")
        # if yr_shp[scn + "_DIFF"].min() < min_val:
        #     min_val = yr_shp[scn + "_DIFF"].min()
        #     print(f"min: {scn} {yr}\n\t{min_val}\n")
        
        yr_shp_dict[f"{scn}_{yr}"] = yr_shp

        diff_maps(yr_shp, scn, yr)


# huc_lst = [3130001, 3020201, 3100205, 6010201, 8090203]
# huc_names = ['Upper Chattahoochee', 'Upper Neuse', 'Hillsborough', 'Watts Bar Lake', 'Eastern Louisiana Coastal']
# for i in range(len(huc_lst)):
    