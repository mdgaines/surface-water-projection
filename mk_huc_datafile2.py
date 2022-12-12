

import os
import pandas as pd
import numpy as np


def add_climate_vars(szn=str, gcm=str, scn=str):

    precip_df = pd.read_csv(f'../data/ClimateData/macav2livneh_GRIDMET_CSVs/{gcm}/{gcm}_{szn}_PRECIP_AVG_{scn}_PROJ.csv', index_col=0)
    precip_df = precip_df.sort_values('huc8')
    mxTemp_df = pd.read_csv(f'../data/ClimateData/macav2livneh_GRIDMET_CSVs/{gcm}/{gcm}_{szn}_MAX_TEMP_AVG_{scn}_PROJ.csv', index_col=0)
    mxTemp_df = mxTemp_df.sort_values('huc8')

    precip_long = pd.melt(precip_df, id_vars=['huc8'], value_vars=precip_df.columns[1:], ignore_index=True)
    precip_long = precip_long.rename({'huc8':'HUC08', 'variable':'YEAR', 'value': 'PRECIP'}, axis='columns')

    mxTemp_long = pd.melt(mxTemp_df, id_vars=['huc8'], value_vars=mxTemp_df.columns[1:], ignore_index=True)
    mxTemp_long = mxTemp_long.rename({'huc8':'HUC08', 'variable':'YEAR', 'value': 'MAX_TMP'}, axis='columns')

    df = precip_long.merge(mxTemp_long, on=['HUC08','YEAR'])

    return(df)


def add_lclu_vars(df=pd.DataFrame, scn=str):

    var_dict = {'AGRI':'PR_AG', 'INTS':'PR_INT', 'FRST':'PR_NAT'}
    var_lst = ['AGRI', 'INTS', 'FRST']
    for var in var_lst:
        proj_df = pd.read_csv(f'../data/LandCover/FORESCE_NLCDCDL_CSVs/{scn}/FORESCE_{var}_FRAC_{scn}_PROJ.csv', index_col=0)
        proj_df = proj_df.sort_values('huc8')

        proj_df_long = pd.melt(proj_df, id_vars=['huc8'], value_vars=proj_df.columns[:-1], ignore_index=True)
        proj_df_long = proj_df_long.rename({'huc8':'HUC08', 'variable':'YEAR', 'value': var_dict[var]}, axis='columns')

        df = df.merge(proj_df_long, on=['HUC08','YEAR'])

    return(df)


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

                    spring_df = add_climate_vars('SPRING', gcm, scn)
                    summer_df = add_climate_vars('SUMMER', gcm, scn)
                    fall_df = add_climate_vars('FALL', gcm, scn)
                    winter_df = add_climate_vars('WINTER', gcm, scn)

                    spring_df = add_lclu_vars(spring_df, foresce)
                    summer_df = add_lclu_vars(summer_df, foresce)
                    fall_df = add_lclu_vars(fall_df, foresce)
                    winter_df = add_lclu_vars(winter_df, foresce)

                    spring_df['SEASON'] = 'Spring'
                    summer_df['SEASON'] = 'Summer'
                    fall_df['SEASON'] = 'Fall'
                    winter_df['SEASON'] = 'Winter'

                    full_df = pd.concat([spring_df, summer_df, fall_df, winter_df])

                    full_df.to_csv(outpath) 


if __name__ == '__main__':
    main()