

import os
import pandas as pd
import numpy as np
import geopandas as gpd

SHP = gpd.read_file('../data/Shapefiles/HUC08/HUC08_paper2/HUC08_paper2.shp')
SHP_DF = SHP[['HUC08','areasqkm']]
SHP_DF['HUC08'] = pd.to_numeric(SHP_DF['HUC08'])
SHP_DF['areasqkm'] = pd.to_numeric(SHP_DF['areasqkm'])


def add_climate_vars(szn=str, gcm=str, scn=str):

    szn_var_dict = {'SPRING':'Sp', 'SUMMER':'Su', "FALL":'Fa', "WINTER":"Wi"}
    if gcm=='GRIDMET':
        szn = szn_var_dict[szn]
        pr_var = "Pr"
        mxTemp_var = "maxTemp"
        tag = "OBS"
    else:
        pr_var = "PRECIP"
        mxTemp_var = "MAX_TEMP"
        tag = f"{scn}_PROJ"

    precip_df = pd.read_csv(f'../data/ClimateData/macav2livneh_GRIDMET_CSVs/{gcm}/{gcm}_{szn}_{pr_var}_AVG_{tag}.csv', index_col=0)
    precip_df = precip_df.sort_values('huc8')
    mxTemp_df = pd.read_csv(f'../data/ClimateData/macav2livneh_GRIDMET_CSVs/{gcm}/{gcm}_{szn}_{mxTemp_var}_AVG_{tag}.csv', index_col=0)
    mxTemp_df = mxTemp_df.sort_values('huc8')

    precip_long = pd.melt(precip_df, id_vars=['huc8'], value_vars=precip_df.columns[1:], ignore_index=True)
    precip_long = precip_long.rename({'huc8':'HUC08', 'variable':'YEAR', 'value': 'PRECIP'}, axis='columns')

    mxTemp_long = pd.melt(mxTemp_df, id_vars=['huc8'], value_vars=mxTemp_df.columns[1:], ignore_index=True)
    mxTemp_long = mxTemp_long.rename({'huc8':'HUC08', 'variable':'YEAR', 'value': 'MAX_TMP'}, axis='columns')

    df = precip_long.merge(mxTemp_long, on=['HUC08','YEAR'])

    return(df)


def add_lclu_vars(scn=str):

    var_dict = {'AGRI':'PR_AG', 'INTS':'PR_INT', 'FRST':'PR_NAT'}
    var_lst = ['AGRI', 'INTS', 'FRST']
    if scn == 'NLCDCDL':
        src = "NLCDCDL"
        tag = "OBS"
    else:
        src = "FORESCE"
        tag = f"{scn}_PROJ"

    for var in var_lst:
        proj_df = pd.read_csv(f'../data/LandCover/FORESCE_NLCDCDL_CSVs/{scn}/{src}_{var}_FRAC_{tag}.csv', index_col=0)
        proj_df = proj_df.sort_values('huc8')

        if scn == 'NLCDCDL':
        
            proj_df['2002'] = proj_df['2001'] * (2/3) + proj_df['2004'] * (1/3)
            proj_df['2003'] = proj_df['2001'] * (1/3) + proj_df['2004'] * (2/3)
            proj_df['2005'] = proj_df['2001'] * 0.5 + proj_df['2004'] * 0.5
            proj_df['2007'] = proj_df['2006'] * 0.5 + proj_df['2008'] * 0.5

        proj_df_long = pd.melt(proj_df, id_vars=['huc8'], value_vars=proj_df, ignore_index=True)
        proj_df_long = proj_df_long.rename({'huc8':'HUC08', 'variable':'YEAR', 'value': var_dict[var]}, axis='columns')

        if var == "AGRI":
            df = proj_df_long
        else:
            df = df.merge(proj_df_long, on=['HUC08','YEAR'])

    return(df)


def add_dswe_var(yrs=list, szn=str):

    for year in range(yrs[0],yrs[1]+1):
        dswe_df = pd.read_csv(f'../data/DSWE_SE/huc_stats_p2/{year}_{szn.capitalize()}.csv')
        dswe_df['YEAR'] = str(year)
        dswe_df = dswe_df.rename({'huc8':'HUC08'}, axis='columns')
        dswe_df = dswe_df.merge(SHP_DF, on='HUC08')
        dswe_df['PR_WATER'] = (dswe_df['total_water'] * 0.0009) / dswe_df['areasqkm']

        if year == yrs[0]:
            df = dswe_df[['YEAR', 'HUC08', 'PR_WATER']]
        else:
            df = pd.concat([df,dswe_df[['YEAR', 'HUC08', 'PR_WATER']]])
    
    return(df)


def mk_proj_csv(gcm=str, foresce=str, scn=str, outpath=str):
    spring_df = add_climate_vars('SPRING', gcm, scn)
    summer_df = add_climate_vars('SUMMER', gcm, scn)
    fall_df = add_climate_vars('FALL', gcm, scn)
    winter_df = add_climate_vars('WINTER', gcm, scn)

    spring_df = spring_df.merge(add_lclu_vars(foresce), on=['HUC08','YEAR'])
    summer_df = summer_df.merge(add_lclu_vars(foresce), on=['HUC08','YEAR'])
    fall_df = fall_df.merge(add_lclu_vars(foresce), on=['HUC08','YEAR'])
    winter_df = winter_df.merge(add_lclu_vars(foresce), on=['HUC08','YEAR'])

    if gcm == 'GRIDMET':
        spring_df = spring_df.merge(add_dswe_var([2001,2018], 'SPRING'), on=['HUC08','YEAR'])
        summer_df = summer_df.merge(add_dswe_var([2001,2018], 'SUMMER'), on=['HUC08','YEAR'])
        fall_df = fall_df.merge(add_dswe_var([2001,2018], 'FALL'), on=['HUC08','YEAR'])
        winter_df = winter_df.merge(add_dswe_var([2001,2018], 'WINTER'), on=['HUC08','YEAR'])        

    spring_df['SEASON'] = 'Spring'
    summer_df['SEASON'] = 'Summer'
    fall_df['SEASON'] = 'Fall'
    winter_df['SEASON'] = 'Winter'

    full_df = pd.concat([spring_df, summer_df, fall_df, winter_df])

    full_df.to_csv(outpath) 

    return()
    

def main():

    GCM_LST = ['GFDL', 'HadGEM2', 'IPSL', 'MIROC5', 'NorESM1']
    SCENARIO_LST = ['RCP45', 'RCP85']
    FORESCE_LST = ['A1B', 'A2', 'B1', 'B2']
    # SEASON_LST = ['SPRING', 'SUMMER', 'FALL', 'WINTER']

    for gcm in GCM_LST:
        for foresce in FORESCE_LST:
            for scn in SCENARIO_LST:
                outpath = f'../data/FutureData/GCM_FORESCE_CSVs/{gcm}_{scn}_{foresce}_ALL.csv'

                if not os.path.exists(outpath):
                    if not os.path.exists(os.path.dirname(outpath)):
                        os.makedirs(os.path.dirname(outpath))

                    mk_proj_csv(gcm, foresce, scn, outpath)
    
    outpath = f'../data/all_data_0118_p2.csv'
    if not os.path.exists(outpath):
        mk_proj_csv(gcm="GRIDMET", foresce="NLCDCDL", outpath=outpath)



if __name__ == '__main__':
    main()