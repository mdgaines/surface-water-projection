# find seasonal average of daily total surface water supply 
# available for use at the outlet of the HUC8 in millions of gallons per day

import os
import pandas as pd
import numpy as np
from glob import glob

import matplotlib.pyplot as plt
import matplotlib.colors
from matplotlib.patches import Patch
import geopandas as gpd
from matplotlib_scalebar.scalebar import ScaleBar
from shapely.geometry.point import Point

POINTS = gpd.GeoSeries(
    [Point(-73.5, 40.5), Point(-74.5, 40.5)], crs=4326
)  # Geographic WGS 84 - degrees

POINTS = POINTS.to_crs(32619)  # Projected WGS 84 - meters

DIST_METERS = POINTS[0].distance(POINTS[1])


plt.rcParams['font.family'] = 'Times New Roman'
plt.rcParams['font.size'] = '20'


HUC02 = gpd.read_file('../data/Shapefiles/HUC02/HUC02_clean_paper2/HUC02_clean_paper2.shp')

def _get_yr_szn(df:pd.DataFrame):
    
    # df['HUC_SEASON'] = df[huc].astype('str') + '_' + df['SEASON']
    df.loc[df.SEASON == 'SPRING', 'YR_SZN'] = df.loc[df.SEASON == 'SPRING', 'YEAR'] * 100 + 0
    df.loc[df.SEASON == 'SUMMER', 'YR_SZN'] = df.loc[df.SEASON == 'SUMMER', 'YEAR'] * 100 + 25
    df.loc[df.SEASON == 'FALL', 'YR_SZN'] = df.loc[df.SEASON == 'FALL', 'YEAR'] * 100 + 50
    df.loc[df.SEASON == 'WINTER', 'YR_SZN'] = df.loc[df.SEASON == 'WINTER', 'YEAR'] * 100 + 75
    df = df.sort_values(by=['YR_SZN'])

    df = df.loc[ (df['YEAR'] == 2040) | (df['YEAR'] == 2070) | (df['YEAR'] == 2099) ]

    return(df)


def _huc_seasonal_wassi(wassi_huc_df:pd.DataFrame, i:int ):
        
    if wassi_huc_df['CELL'].iloc[i] != wassi_huc_df['CELL'].iloc[i+1] and \
        wassi_huc_df['CELL'].iloc[i] != wassi_huc_df['CELL'].iloc[i+2]:
        print("huc error")
        return

    if wassi_huc_df['MONTH'].iloc[i] == 3:
        season = 'SPRING'
    elif wassi_huc_df['MONTH'].iloc[i] == 6:
        season = 'SUMMER'        
    elif wassi_huc_df['MONTH'].iloc[i] == 9:
        season = 'FALL'
    elif wassi_huc_df['MONTH'].iloc[i] == 12:
        season = 'WINTER'
    else:
        print('month error')
        return
    
    sws_avg = (wassi_huc_df['SWS_MGD'].iloc[i] + wassi_huc_df['SWS_MGD'].iloc[i+1] + 
               wassi_huc_df['SWS_MGD'].iloc[i+2]) / 3
    
    cell = wassi_huc_df['CELL'].iloc[i]
    year = wassi_huc_df['YEAR'].iloc[i]

    return [cell, year, season, sws_avg]


def calc_seasonal_wassi(wassi_df:pd.DataFrame):

    huc08 = wassi_df.CELL.unique()

    wassi_seasonal_df = pd.DataFrame()
    for huc in huc08:
        wassi_huc_df = wassi_df[wassi_df['CELL'] == huc]
        wassi_huc_df = wassi_huc_df[2:-1]

        huc_szn_avg_df = pd.DataFrame(0, index=np.arange(len(wassi_huc_df)/3), 
                                      columns=['CELL', 'YEAR', 'SEASON', 'SWS_MGD'])
        counter = 0
        for i in range(0, len(wassi_huc_df), 3):

            huc_szn_avg_df.iloc[counter] = _huc_seasonal_wassi(wassi_huc_df, i)

            counter += 1

        wassi_seasonal_df = wassi_seasonal_df.append(huc_szn_avg_df, ignore_index=True)

    wassi_seasonal_df = _get_yr_szn(wassi_seasonal_df)
    wassi_seasonal_df = wassi_seasonal_df.rename({'CELL':'HUC08'}, axis='columns')

    return wassi_seasonal_df


def import_files(scn:str, rcp:str): #, rcp:str):

    merf_fl = glob(f'../data/FutureData/GCM_FORESCE_CSVs/HUC_CI/MULTIMODEL*{rcp}*{scn}*.csv')
    merf_fl.sort()
    merf_df = pd.read_csv(merf_fl[0], index_col=0)

    huc08 = merf_df.HUC08.unique()
    # 8080100 is really 8080101
    huc08 = np.append(huc08, 8080100)


    wassi_fl_lst = glob(f'../data/WASSI/MONTHWaSSI_LCLU/{scn}*/*{rcp}*.txt')
    wassi_fl_lst.sort()

    wassi_dict = {}
    for wassi_fl in wassi_fl_lst:
        key = os.path.basename(wassi_fl)[:-4]
        df = pd.read_csv(wassi_fl)
        df = df[df['CELL'].isin(huc08)]
        seasonal_df = calc_seasonal_wassi(df)
        wassi_dict[key] = seasonal_df

    return( wassi_dict, merf_df)


def calc_multimodel_wassi(wassi_dict:dict, yr:str):

    yr_keys = [key for key in wassi_dict.keys() if yr in key]

    rcp = yr_keys[0].split('_')[5]
    scn = yr_keys[0].split('_')[3]

    for szn in ['SPRING', 'SUMMER', 'FALL', 'WINTER']:
        outpath = f'../data/WASSI/SEASONAL_MULTIMEAN_LCLU/MULTIMODEL_{rcp}_{scn}_{yr}_{szn}_SWS-MGD.csv'
        if os.path.exists(outpath):
            print(f'{os.path.basename(outpath)} exists')
            continue

        for i in range(len(yr_keys)):
            wassi_df = wassi_dict[yr_keys[i]]
            wassi_df = wassi_df.loc[ wassi_df['SEASON'] == szn]
            if i == 0:
                multi_wassi_df = wassi_df
            else:
                multi_wassi_df = multi_wassi_df.merge(wassi_df[['HUC08', 'SWS_MGD', 'YR_SZN']], 
                                                      on=['HUC08', 'YR_SZN'], 
                                                      suffixes=('', f'_{yr_keys[i].split("_")[4][0:4]}'))
        
        multi_wassi_df['SWS_MGD_MULTIMEAN'] = multi_wassi_df.filter(regex='SWS_MGD').mean(axis=1)

        multi_wassi_df.to_csv(outpath)

    return


def print_quantiles(yr_shp, yr, scn, rcp, model, col_val=''):
        
        if col_val == '':
            if model == 'WASSI':
                # col_val = 'DIFF_SWS_km2'
                col_val = 'DIFF'

            elif model == 'MERF':
                col_val = 'DIFF'

        print(f"\n{model} {yr} {scn}-{rcp}\n")
        print(f"\tmin: {yr_shp[col_val].min()}\n\
        10: {yr_shp[col_val].quantile(0.1)}\n\
        25: {yr_shp[col_val].quantile(0.25)}\n\
        45: {yr_shp[col_val].quantile(0.45)}\n\
        50: {yr_shp[col_val].quantile(0.5)}\n\
        55: {yr_shp[col_val].quantile(0.55)}\n\
        75: {yr_shp[col_val].quantile(0.75)}\n\
        90: {yr_shp[col_val].quantile(0.9)}\n\
        max: {yr_shp[col_val].max()}\n\
        median: {yr_shp[col_val].median()}")


def plot_sws_maps(yr_shp, scn, yr, model):

    if model == 'WASSI':
        # outpath = f"../imgs/Paper2/Results/sws_diff_maps/MULTIMODEL_{scn}_{yr}_DIFF_SWS_KM2.png"
        # outpath = f"../imgs/Paper2/Results/sws_diff_maps/MULTIMODEL_{scn}_{yr}_PER-DIFF_SWS.png"
        outpath = f"../imgs/Paper2/Results/sws_diff_maps/MULTIMODEL_{scn}_{yr}_PER-DIFF_WASSI.png"
        # bounds = [-2000, -700, -400, -200, 0, 200, 400, 700, 2000] # for raw diffs
        # col_name = 'DIFF'
        col_name = 'PER_DIFF_wassi'
        bounds = [-300, -100, -50, -25, -1, 0, 1, 25, 50, 100, 300]
        # bounds = [-0.6, -0.2, -0.1, -0.05, 0, 0.05, 0.1, 0.2, 0.6] # for diffs per km2
        # col_name = 'DIFF_SWS_km2'
        label = "Percent Change"# in Surface Water Supply (millions gallons per day) from 2011"
    elif model == 'MERF':
        # outpath = f"../imgs/Paper2/Results/projswa_diff_maps/MULTIMODEL_{scn}_{yr}_DIFF_projSWA_KM2.png"
        # outpath = f"../imgs/Paper2/Results/projswa_diff_maps/MULTIMODEL_{scn}_{yr}_PER-DIFF_projSWA_KM2.png"
        outpath = f"../imgs/Paper2/Results/projswa_diff_maps/MULTIMODEL_{scn}_{yr}_PER-DIFF_MERF.png"
        # bounds = [-0.1, -0.02, -0.01, -0.005, 0, 0.005, 0.01, 0.02, 0.1]
        # col_name = 'DIFF'
        col_name = 'PER_DIFF_merf'
        bounds = [-300, -100, -50, -25, -10, 0, 10, 25, 50, 100, 300]
        # bounds = [-1, -.50, -.25, -.1, 0, .25, .75, 1, 3]
        label = "Percent Change"# in Surface Water Area (per km2) from 2011"

    elif model == 'merged':
        outpath = f"../imgs/Paper2/Results/agree_diff_maps/MULTIMODEL_{scn}_{yr}_WaSSI-MERF-AGREE-3.png"
        col_name = 'AGREEMENT'

        label = 'Agreement of WaSSI and MERF projection models'

    # if os.path.exists(outpath):
    #     print(f"{os.path.basename(outpath)} exists.")
    #     return
    


    fig, ax = plt.subplots(figsize=(20, 12))
    
    plt.subplots_adjust(top = 1, bottom = 0, right = 1, left = 0, hspace = 0, wspace = 0)
    ax.set_aspect('equal')
    ax.set_axis_off()

    if model == 'merged':
        agreement_palette = {'Positive':'#54adee', 
                             'Negative':'#ffa5b6', 
                             'Disagree':'#ffffef'} 
        
        legend_elements = [Patch(facecolor=agreement_palette['Positive'], 
                                 edgecolor='black', linewidth=0.5,
                                 label='Increase'),
                           Patch(facecolor=agreement_palette['Negative'], 
                                 edgecolor='black', linewidth=0.5,
                                 label='Decrease'),
                           Patch(facecolor=agreement_palette['Disagree'], 
                                 edgecolor='black', linewidth=0.5,
                                 label='Disagree')]

        agreement_dict = {}
        for ctype, data in yr_shp.groupby('AGREEMENT'):
            print(ctype)
            # print(data)
            # label=ctype
            agreement_dict[ctype + '_03'] = merged_df.loc[(merged_df['HUC02']=='03') & (merged_df['AGREEMENT'] == ctype)].count()['HUC02']
            agreement_dict[ctype + '_06'] = merged_df.loc[(merged_df['HUC02']=='06') & (merged_df['AGREEMENT'] == ctype)].count()['HUC02']
            agreement_dict[ctype + '_08'] = merged_df.loc[(merged_df['HUC02']=='08') & (merged_df['AGREEMENT'] == ctype)].count()['HUC02']

            color = agreement_palette[ctype]
            data.plot(color=color, ax=ax, edgecolor='black', linewidth=0.5)
        
        row_labels = ['Increasing', 'Decreasing', 'Disagree']
        col_labels = ['South Atlantic Gulf', ' Tennessee', 'Lower Mississippi']
        table_vals = [[agreement_dict['Positive_03'], agreement_dict['Positive_06'], agreement_dict['Positive_08']],
                      [agreement_dict['Negative_03'], agreement_dict['Negative_06'], agreement_dict['Negative_08']],
                      [agreement_dict['Disagree_03'], agreement_dict['Disagree_06'], agreement_dict['Disagree_08']]]
        info_table = plt.table(bbox=[.1, .02, .49, .25],
                               cellText=table_vals, 
                               cellLoc='center',
                               rowLabels=row_labels, 
                               colLabels=col_labels, 
                               rowColours=['#54adee', '#ffa5b6','#ffffef'])
        plt.text(-92.5, 28, 
                 'WaSSI and MERF Projection HUC08 Agreement Counts\nper Water Resource Region',
                 multialignment='center')
        # ax.legend(handles=legend_elements, loc=[.8,.1], title = 'WaSSI and MERF Projection Agreement', prop={'size':24})

    else:
        # colors = ['#3a1800', '#6a3a0a', '#9b6119', '#c88d3c', '#e9c080', '#8bd2c9', '#49a59d', '#21776f', '#084b43', '#002319']
        colors = ['#4a2400', '#704010', '#966028', '#ba8347', '#dca76c', '#eed3a3', '#ade1d9', '#70beb6', '#3d9a92', '#23756c', '#0d5247', '#003125']

        cmap=matplotlib.colors.ListedColormap(colors)
        norm = matplotlib.colors.BoundaryNorm(bounds, len(colors), extend='both')

        yr_shp.plot(column=yr_shp[col_name], ax=ax, edgecolor='black', linewidth=0.5, legend=True,
                    cmap=cmap, norm=norm, 
                    legend_kwds={"label": label,
                                "orientation": "horizontal", 
                                "shrink": 0.5, "pad":0.0, 
                                "extendfrac": 'auto'})


    HUC02.plot(ax=ax, edgecolor='black', facecolor='none', linewidth=3)

#  | (yr_shp["HUC08"] == ) 3130001| \
#            (yr_shp["HUC08"] == 3100205) | (yr_shp["HUC08"] == 6010201) | \
#             (yr_shp["HUC08"] == 8090203)

    yr_shp[(yr_shp["HUC08"] == 3020201) | \
           (yr_shp["HUC08"] == 6010201) | \
           (yr_shp["HUC08"] == 8090203)].plot(ax=ax, edgecolor="#ff00ff", \
                                            facecolor="none", linewidth=2)
    
    ax.set_title(f"{yr} {scn}", size=28)

    plt.savefig(outpath, dpi=300, facecolor='w', edgecolor='w', transparent=False, pad_inches=0)

    plt.close()

    return


def plot_agree_maps(scn_rcp_yr_dict:dict, keys_lst:list):

    # new fig of study area gcms
    # fig = plt.figure(figsize=(15, 15))#, layout='constrained')

    # gs = fig.add_gridspec(3, 3, hspace=0, wspace=0)
    # ((ax1, ax2, ax3), (ax4, ax5, ax6), (ax7, ax8, ax9)) = gs.subplots()
    # ax_lst = [ax1, ax2, ax3, ax4, ax5, ax6, ax7, ax8, ax9]

    # fig.supxlabel('Years', size=28)
    # fig.supylabel('Focal Scenarios', size=28)

    count = 0
    for key in keys_lst:

        fig, ax = plt.subplots(figsize=(20, 12))

        yr_shp = scn_rcp_yr_dict[key]
        # ax = ax_lst[count]
        print(key)

        agreement_palette = {'Positive':'#54adee', 
                             'Negative':'#ffa5b6', 
                             'Disagree':'#ffffef'} 
        
        legend_elements = [Patch(facecolor=agreement_palette['Positive'], 
                                 edgecolor='black', linewidth=0.5,
                                 label='Increase'),
                           Patch(facecolor=agreement_palette['Negative'], 
                                 edgecolor='black', linewidth=0.5,
                                 label='Decrease'),
                           Patch(facecolor=agreement_palette['Disagree'], 
                                 edgecolor='black', linewidth=0.5,
                                 label='Disagree')]

        yr_shp['color'] = ''
        yr_shp.loc[yr_shp['AGREEMENT']=='Positive', 'color'] = agreement_palette['Positive']
        yr_shp.loc[yr_shp['AGREEMENT']=='Negative', 'color'] = agreement_palette['Negative']
        yr_shp.loc[yr_shp['AGREEMENT']=='Disagree', 'color'] = agreement_palette['Disagree']

        yr_shp.plot(ax=ax, color = yr_shp['color'],  edgecolor='black', linewidth=0.5)
        ax.set_axis_off()
        ax.add_artist(ScaleBar(DIST_METERS, location='lower left', length_fraction=0.5, width_fraction=0.02, border_pad = 0.8, pad = 0.8))

        HUC02.plot(ax=ax, edgecolor='black', facecolor='none', linewidth=3)

        key_info = key.split('_')
        yr = key_info[2]
        rcp = key_info[1]
        scn = key_info[0]
        if count in [0, 1, 2]:
            ax.set_title(yr, size=74)
        if count in [0,3,6]:
            ax.set_ylabel(f'RCP {rcp[:-2]} - {scn}')

    #  | (yr_shp["HUC08"] == ) 3130001| \
    #            (yr_shp["HUC08"] == 3100205) | (yr_shp["HUC08"] == 6010201) | \
    #             (yr_shp["HUC08"] == 8090203)

        yr_shp[(yr_shp["HUC08"] == 3020201) | \
            (yr_shp["HUC08"] == 6010201) | \
            (yr_shp["HUC08"] == 8090203)].plot(ax=ax, edgecolor="#ff00ff", \
                                                facecolor="none", linewidth=2)        
    
    # ax.set_title(f"{yr} {scn}", size=28)
        # outpath = f"../imgs/Paper2/Results/agree_diff_maps/MULTIMODEL_{scn}-{rcp}_{yr}_WaSSI-MERF-AGREE-3.png"
        outpath = f"../imgs/Paper2/Results/agree_diff_maps/scalebar.png"

        plt.savefig(outpath, dpi=300, facecolor='w', edgecolor='w', transparent=False, pad_inches=0)

        plt.close()

        count += 1

    return




#### MAIN ####

# import shape files for WASSI and MERF HUC08's
shp = gpd.read_file('../data/Shapefiles/HUC08/HUC08_wassiweb_paper2/HUC08_wassiweb_paper2.shp')
shp = shp[['HUC8','SQ_MILES', 'geometry']]
shp = shp.rename({'HUC8':'HUC08'}, axis='columns')
shp['SQ_KM'] = shp['SQ_MILES'] / 0.3861 # sq mi to sq km conversion
shp['HUC02'] = shp['HUC08'].apply(lambda x: str(x)[0].zfill(2))

shp2 = gpd.read_file('../data/Shapefiles/HUC08/HUC08_paper2/HUC08_paper2.shp')
shp2 = shp2[['huc8','areasqkm', 'geometry']]
shp2['HUC08'] = shp2['huc8'].apply(lambda x : int(x))
shp2['AREA'] = shp2['areasqkm'].apply(lambda x : float(x))
shp2['HUC02'] = shp2['HUC08'].apply(lambda x: str(x)[0].zfill(2))

# import WASSI and MERF ESTIMATED 2011 VALUES
wassi_2011_obs = pd.read_csv('../data/WASSI/SEASONAL_2011_OBS.csv', index_col=0)
wassi_2011_SPRING_obs = wassi_2011_obs.loc[wassi_2011_obs['SEASON']=='SPRING']
wassi_2011_SPRING_obs = wassi_2011_SPRING_obs.rename(columns={'SWS_MGD':'SWS_MGD_2011'})

merf_2011_obs = pd.read_csv('../data/FutureData/dswe_2011_obs_proj.csv', index_col=0)
merf_2011_SPRING_obs = merf_2011_obs.loc[merf_2011_obs['SEASON']=='Spring']

# Set up for loop variables
scn_lst = ['A1B', 'A2', 'B1', 'B2']
rcp_lst = ['RCP45', 'RCP85']
yrs_lst = ['2040', '2070', '2099']
scn_rcp_lst = [(scn, rcp) for scn in scn_lst for rcp in rcp_lst]


scn_rcp_yr_dict = {}

for scn, rcp in scn_rcp_lst:
    inpath_merf = f'../data/FutureData/GCM_FORESCE_CSVs/HUC_CI/MULTIMODEL_{rcp}_{scn}_MC_HUC_CI95.csv'
    merf_df = pd.read_csv(inpath_merf, index_col=0)

    for yr in yrs_lst:
        inpath_wassi = f'../data/WASSI/SEASONAL_MULTIMEAN_LCLU/MULTIMODEL_{rcp}_{scn}_{yr}_SPRING_SWS-MGD.csv'

        if not os.path.exists(inpath_wassi):

            wassi_dict, merf_df = import_files(scn, rcp)

            calc_multimodel_wassi(wassi_dict, yr)
        else:
            print(f'{os.path.basename(inpath_wassi)} exists.')

        spring_wassi_df = pd.read_csv(inpath_wassi, index_col=0)
        spring_wassi_df = spring_wassi_df.merge(wassi_2011_SPRING_obs[['HUC08','SWS_MGD_2011']], on='HUC08')
        spring_wassi_df['DIFF'] = spring_wassi_df['SWS_MGD_MULTIMEAN'] - spring_wassi_df['SWS_MGD_2011']
        spring_wassi_df['PER_DIFF'] = (spring_wassi_df['SWS_MGD_MULTIMEAN'] - spring_wassi_df['SWS_MGD_2011']) / spring_wassi_df['SWS_MGD_2011'] * 100
        
        # yr_shp['SWS_km2'] = yr_shp['SWS_MGD_MULTIMEAN'] / yr_shp['SQ_KM']
        yr_shp_wassi = shp.merge(spring_wassi_df[["HUC08", "PER_DIFF", "DIFF"]], on="HUC08")
        # yr_shp_wassi['DIFF_SWS_km2'] = yr_shp_wassi['DIFF'] / yr_shp_wassi['SQ_KM']

        yr_shp_wassi.loc[yr_shp_wassi['HUC08'] == 8080100, 'HUC08'] = 8080101

        spring_merf_df = merf_df.loc[merf_df['YR_SZN'] == int(yr)*100]
        spring_merf_df = spring_merf_df.merge(merf_2011_SPRING_obs[['HUC08','PRED_PR_WATER']], on='HUC08')
        spring_merf_df['DIFF'] = spring_merf_df['MEAN'] - spring_merf_df['PRED_PR_WATER']
        spring_merf_df['PER_DIFF'] = (spring_merf_df['MEAN'] - spring_merf_df['PRED_PR_WATER']) / spring_merf_df['PRED_PR_WATER'] * 100
        yr_shp_merf = shp2.merge(spring_merf_df[["HUC08", "PER_DIFF", "DIFF"]], on="HUC08")

        merged_df = yr_shp_merf[['HUC08', 'PER_DIFF', 'DIFF', 'geometry']].merge(yr_shp_wassi[['HUC08', 'DIFF', 'PER_DIFF', 'HUC02']], on='HUC08', suffixes=('_merf', '_wassi'))
        merged_df['AGREEMENT'] = ''
        merged_df.loc[(merged_df['DIFF_merf'] > 0) & (merged_df['DIFF_wassi'] > 0), 'AGREEMENT'] = 'Positive'
        merged_df.loc[(merged_df['DIFF_merf'] < 0) & (merged_df['DIFF_wassi'] < 0), 'AGREEMENT'] = 'Negative'
        merged_df.loc[(merged_df['AGREEMENT']==''), 'AGREEMENT'] = 'Disagree'


        for huc02 in ['03', '06', '08']:
            print(f'\nHUC02: {huc02}')
            print(f'WASSI')
            print(f"\tnum inc: {yr_shp_wassi.loc[(yr_shp_wassi['HUC02']==huc02) & (yr_shp_wassi['DIFF'] > 0)].count()['DIFF']}\
                num total: {yr_shp_wassi.loc[(yr_shp_wassi['HUC02']==huc02)].count()['DIFF']}")
            print(f'MERF')
            print(f"\tnum inc: {yr_shp_merf.loc[(yr_shp_merf['HUC02']==huc02) & (yr_shp_merf['DIFF'] > 0)].count()['DIFF']}\
                num total: {yr_shp_merf.loc[(yr_shp_merf['HUC02']==huc02)].count()['DIFF']}")
            
            print(f"\nBoth inc count: {merged_df.loc[(merged_df['HUC02']==huc02) & (merged_df['DIFF_merf'] > 0) & (merged_df['DIFF_wassi'] > 0)].count()['HUC02']}")
            print(f"Both dec count: {merged_df.loc[(merged_df['HUC02']==huc02) & (merged_df['DIFF_merf'] < 0) & (merged_df['DIFF_wassi'] < 0)].count()['HUC02']}")

        scn_rcp_yr_dict[f'{scn}_{rcp}_{yr}'] = merged_df


        print_quantiles(yr_shp_wassi, yr, scn, rcp, 'WASSI', col_val = 'PER_DIFF')
        print_quantiles(yr_shp_merf, yr, scn, rcp, 'MERF', col_val = 'PER_DIFF')

        plot_sws_maps(merged_df, f"{scn}-{rcp}", yr, 'merged')

        plot_sws_maps(yr_shp_wassi, f"{scn}-{rcp}", yr, 'WASSI')
        plot_sws_maps(yr_shp_merf, f"{scn}-{rcp}", yr, 'MERF')



keys_lst = ['B1_RCP45_2040', 'B1_RCP45_2070', 'B1_RCP45_2099',\
           'B2_RCP85_2040', 'B2_RCP85_2070', 'B2_RCP85_2099',\
           'A2_RCP85_2040', 'A2_RCP85_2070', 'A2_RCP85_2099']

plot_agree_maps(scn_rcp_yr_dict, keys_lst)


for key in keys_lst:
    key_info = key.split('_')
    scn = key_info[0]
    rcp = key_info[1]
    yr = key_info[2]

    plot_sws_maps(scn_rcp_yr_dict[key], f"{rcp} - {scn}", yr, 'WASSI')
    plot_sws_maps(scn_rcp_yr_dict[key], f"{rcp} - {scn}", yr, 'MERF')



# run MERF with no LCLU change to compare with WASSI ? - not yet

# does WASSI use any LCLU change ? READ SOME PAPERS
# --> WASSI LCLU changes are uniform from user inputs. 
# --  We re-ran WASSI based on differences between the NLCD 2011 LCLU values and the
# --  FORESCE scenario LCLU values for 2040, 2070, and 2080. 
# --  We will now get the multi-model WASSI SWS_MGD values for each of these years 
# --  for all HUCs and compare them to the MERF multimodel pSWA visually on a map.

# run MERF on historical data ? - not yet
# calc historical DSWE ? - not yet
# wassi_seasonal_df[wassi_seasonal_df['YR_SZN'] >= 200600]
