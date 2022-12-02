import os
import pandas as pd
import numpy as np
from glob import glob

import matplotlib.pyplot as plt



def mk_year_season_csv(outpath = str):
    '''
        Makes CSVs with data (projected or observed) from 1985-2022 for each season
        and climate/land cover variable.
        
        parameter:
            outpath: path to where the csv will be saved
                '../data/ClimateData/macav2livneh_GRIDMET_CSVs/{src}/{src}_{szn}_{var}_AVG_{scn (opitonal)}_{tag}.csv'
    '''
    info_lst = os.path.basename(outpath)[:-4].split('_')
    src = info_lst[0] # data source (GRIDMET, GFDL, etc.)
    szn = info_lst[1] # season (Sp, Su, Fa, Wi for GRIDEMT; SPRING, SUMMER, FALL, WINTER for GCMs)
    if 'TEMP' in outpath:
        var = '_'.join([info_lst[2], info_lst[3]])
    else:
        var = info_lst[2] # variable (Pr, maxTemp, minTemp for GRIDMET; PRECIP, MAX_TEMP, MIN_TEMP for GCMs)
    scn = info_lst[-2] # scenario (HISTORICAL, RCP45, RCP85 for GCMs)

    if src=='GRIDMET':
        fl_lst = glob(f'../data/ClimateData/GRIDMET_AVG_STDEV/GRIDMET_YR_AVG/*{szn}_{var}*.csv')
    else:
        fl_lst = glob(f'../data/ClimateData/{src}*/{scn}/zonal_avg/*{szn}_{var}*.csv')

    # loop through files and get climate var for each year
    for i in range(len(fl_lst)):
        fl = pd.read_csv(fl_lst[i])
        if src == 'GRIDMET':
            col_name= 'mean'
            yr = fl['Yr_Szn'][0][:4]
        else:
            col_name = f'AVG_{var}'
            yr = fl['YEAR'][0]

        if i == 0:
            full_df = fl[['huc8', col_name]]
            full_df = full_df.rename(columns={col_name:yr})
        else:
            full_df = full_df.join(fl[['huc8', col_name]].set_index('huc8'), on='huc8')
            full_df = full_df.rename(columns={col_name:yr})

    # save CSV
    full_df.to_csv(outpath)
    print(f'{os.path.basename(outpath)} saved.')
    return()


def mk_diff_csv(gridmet_path=str, src_path=str, src=str, szn=str, var=str, scn=str):

    gridmet_df = pd.read_csv(gridmet_path, index_col=0)
    gridmet_df.loc[:, 'huc4'] = [str(i)[0:4] for i in gridmet_df['huc8'].to_list()]
    src_df = pd.read_csv(src_path, index_col=0)
    src_df.loc[:, 'huc4'] = [str(i)[0:4] for i in src_df['huc8'].to_list()]
    huc4_lst = gridmet_df.huc4.unique()
    
    outpath_lst = [f'../data/ClimateData/macav2livneh_GRIDMET_diff_CSVs/{src}/{szn}/{var}/{scn}/{src}_{szn}_{var}_DIFF_{scn}_{huc4}.csv'\
        for huc4 in huc4_lst]
    
    for outpath in outpath_lst:
        if not os.path.exists(outpath):
            if not os.path.exists(os.path.dirname(outpath)):
                if not os.path.exists(os.path.dirname(os.path.dirname(outpath))):
                    if not os.path.exists(os.path.dirname(os.path.dirname(os.path.dirname(outpath)))):
                        if not os.path.exists(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(outpath))))):
                            os.mkdir(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(outpath)))))
                        os.mkdir(os.path.dirname(os.path.dirname(os.path.dirname(outpath))))
                    os.mkdir(os.path.dirname(os.path.dirname(outpath)))
                os.mkdir(os.path.dirname(outpath))
            huc4 = outpath[:-4].split('_')[-1]  # get huc4 name
            gridmet_huc_df = gridmet_df[gridmet_df['huc4']==huc4]   # get gridmet data for huc4
            gridmet_huc_df = gridmet_huc_df.set_index('huc8')
            src_huc_df = src_df[src_df['huc4']==huc4]   # get src data for huc4
            src_huc_df = src_huc_df.set_index('huc8')
            yr_lst = [i for i in gridmet_huc_df.columns if i in src_huc_df.columns] # get years that both gridmet and src have
            yr_lst.sort()
            yr_lst = yr_lst[:-1]
            diff_df = src_huc_df[yr_lst] - gridmet_huc_df[yr_lst]
            diff_df.to_csv(outpath)
            print(f'{os.path.basename(outpath)} saved.')

    return()


### MAIN ###
# 1) make all CSVs OBS and PROJ
# 2) calculate diff between obs and proj and save to csv
#    - raw diff, mean percent error, median percent error
#    - try to figure out which is best to use for MC bootstrapping
# 3) plot error dist for each szn, huc4, variable
# 4) check for temporal trend in error
# 5) fit distributions...

def main():
    src_lst = ['GRIDMET', 'GFDL']#, 'HadGEM2', 'IPSL', 'MIROC5', 'NorESM1']
    obs_proj_dict = {'OBS': [['Sp', 'Su', 'Fa', 'Wi'], ['Pr', 'maxTemp', 'minTemp']],\
                     'PROJ': [['SPRING', 'SUMMER', 'FALL', 'WINTER'], ['PRECIP', 'MAX_TEMP', 'MIN_TEMP']]}

    ###############################################
    #### make csvs ####
    ###############################################
    for src in src_lst:
        if src == 'GRIDMET':
            tag = 'OBS'
            scn_lst = ['OBS']
        else:
            tag = 'PROJ'
            scn_lst = ['HISTORICAL_PROJ', 'RCP45_PROJ', 'RCP85_PROJ']
        szn_lst = obs_proj_dict[tag][0]
        var_lst = obs_proj_dict[tag][1] 

        outpath_lst = [f'../data/ClimateData/macav2livneh_GRIDMET_CSVs/{src}/{src}_{szn}_{var}_AVG_{scn}.csv'\
                        for szn in szn_lst for var in var_lst for scn in scn_lst]

        for outpath in outpath_lst:
            if not os.path.exists(os.path.dirname(outpath)):
                os.mkdir(os.path.dirname(outpath))
            if not os.path.exists(outpath):
                mk_year_season_csv(outpath)
    print('All OBS and PROJ csv have been saved.')
    ###############################################
    
    ###############################################
    #### make diff csvs ####
    ###############################################
    gridmet_lst = glob('../data/ClimateData/macav2livneh_GRIDMET_CSVs/GRIDMET/*.csv')
    gridmet_info_dict = {'Sp':'SPRING', 'Su':'SUMMER', 'Fa':'FALL', 'Wi':'WINTER',\
        'Pr':'PRECIP', 'maxTemp':'MAX_TEMP', 'minTemp':'MIN_TEMP'}
    scn_lst = ['HISTORICAL', 'RCP45', 'RCP85']

    for gridmet_path in gridmet_lst:
        info_lst = os.path.basename(gridmet_path)[:-4].split('_')
        szn = gridmet_info_dict[info_lst[1]]
        var = gridmet_info_dict[info_lst[2]]

        for scn in scn_lst:
            for src in src_lst:
                if src == 'GRIDMET':
                    continue

                src_path = glob(f'../data/ClimateData/macav2livneh_GRIDMET_CSVs/{src}/*{szn}_{var}*{scn}*.csv')[0]
                mk_diff_csv(gridmet_path, src_path, src, szn, var, scn)
    print('All DIFF csv have been saved.')
    ###############################################

    ###############################################
    #### make plots ####
    ###############################################
