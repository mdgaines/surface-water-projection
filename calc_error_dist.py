import os
import pandas as pd
import numpy as np
from glob import glob

import matplotlib.pyplot as plt



def mk_year_season_csv(src=str, szn=str, var=str, scn=str, tag=str):
    '''
        Makes CSVs with data (projected or observed) from 1985-2022 for each season
        and climate/land cover variable.
        
        parameters:
            src: data source (GRIDMET, GFDL, etc.)
            szn: season (Sp, Su, Fa, Wi for GRIDEMT; SPRING, SUMMER, FALL, WINTER for GCMs)
            var: variable (Pr, maxTemp, minTemp for GRIDMET; PRECIP, MAX_TEMP, MIN_TEMP for GCMs)
            scn: scenario (HISTORICAL, RCP45, RCP85 for GCMs)
            tag: OBS for GRIDMET and PROJ for GCMs
    '''
    outpath = f'../data/ClimateData/macav2liveneh_GRIDMET_CSVs/{src}_{szn}_{var}_AVG_{tag}.csv'

    if src=='GRIDMET':
        fl_lst = glob(f'../data/ClimateData/GRIDMET_AVG_STDEV/GRIDMET_YR_AVG/*{szn}_{var}*.csv')
    else:
        fl_lst = glob(f'../data/ClimateData/{src}*/{scn}/zonal_avg/*{szn}_{var}*.csv')

    # loop through files and get climate var for each year
    for i in range(len(fl_lst)):
        fl = pd.read_csv(fl_lst[i])
        if i == 0:
            full_df = fl[['huc8', 'mean']]
            full_df = full_df.rename(columns={'mean':fl['Yr_Szn'][0][:4]})
        else:
            full_df = full_df.join(fl[['huc8', 'mean']].set_index('huc8'), on='huc8')
            full_df = full_df.rename(columns={'mean':fl['Yr_Szn'][0][:4]})

    # save CSV
    full_df.to_csv(outpath)

    return()




### MAIN ###
# 1) make all CSVs OBS and PROJ
# 2) calculate diff between obs and proj and save to csv
#    - raw diff, mean percent error, median percent error
#    - try to figure out which is best to use for MC bootstrapping
# 3) plot error dist for each szn, huc, variable
# 4) check for temporal trend in error
# 5) fit distributions...

if not os.path.exists('../data/ClimateData/macav2liveneh_GRIDMET_CSVs/GRIDMET*.csv'):
    XXX = pd.read_csv