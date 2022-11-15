

import os
import pandas as pd
import numpy as np
from glob import glob

#################################################################
########################## Climate ##############################
#################################################################

# B2 - RCP45
# A1B - RCP85


sres_rcp_pairs = [('B2','RCP45'),('A1B','RCP85')]
gcm = 'GFDLESM2M'
lclu = 'FORESCE'

for k in range(len(sres_rcp_pairs)):

    print('\nStarting {}-{}'.format(sres_rcp_pairs[k][0],sres_rcp_pairs[k][1]))

    outpath = '../data/FutureData/2006_2099_{0}_{1}_{2}_{3}.csv'.format(sres_rcp_pairs[k][0], lclu, sres_rcp_pairs[k][1], gcm)

    if os.path.exists(outpath):
        print(os.path.basename(outpath), 'exists.')

    all_yr_szn_df = pd.DataFrame()

    for i in range(2006,2100):

        print('Year:', i)

        # anom_rcp45_yr_csvs = glob('../data/ClimateData/GFDL*/RCP45/zonal_anom/*{}*ANOM.csv'.format(i))
        # lclu_B2_yr_csvs = glob('../data/LandCover/FORE-SCE/*Landcover*/zonal_lclu_csv/{}*B2*.csv'.format(i))

        anom_yr_csvs = glob('../data/ClimateData/GFDL*/{}/zonal_anom/*{}*ANOM.csv'.format(sres_rcp_pairs[k][1],i))
        lclu_yr_csvs = glob('../data/LandCover/FORE-SCE/*Landcover*/zonal_lclu_csv/{}*{}*.csv'.format(i,sres_rcp_pairs[k][0]))


        lclu_B2_df = pd.read_csv(lclu_yr_csvs[0])

        szns = ['SPRING', 'SUMMER', 'FALL', 'WINTER']

        szn_dfs = {}

        for szn in szns:
            szn_anom = [x for x in anom_yr_csvs if szn in x]

            if len(szn_anom) != 0:

                for j in range(len(szn_anom)):
                    anom_path = szn_anom[j]
                    path_element_lst = os.path.basename(anom_path).split('_')
                    climate_var = path_element_lst[-4]
                    szn = path_element_lst[-2]
                    temp_df = pd.read_csv(anom_path, index_col=0)
                    if j == 0:
                        new_df = lclu_B2_df.merge(temp_df[['huc8','SEASON',climate_var+'_ANOM']], on='huc8')
                    else:
                        new_df = new_df.merge(temp_df[['huc8',climate_var+'_ANOM']], on='huc8')
                        # new_df = new_df.merge(lclu_B2_df[['huc8','PR_AG','PR_NAT','PR_INT']], on='huc8')
                    
                    del(temp_df)
                    
                szn_dfs[szn] = new_df

                del(new_df)

        yr_df = pd.concat([szn_dfs['SPRING'],szn_dfs['SUMMER']])
        yr_df = pd.concat([yr_df,szn_dfs['FALL']])
        if len(szn_dfs) == 4:
            yr_df = pd.concat([yr_df,szn_dfs['WINTER']])

        all_yr_szn_df = pd.concat([all_yr_szn_df, yr_df])

        del(yr_df)

    all_yr_szn_df.to_csv(outpath)
    print(os.path.basename(outpath), 'saved.')

    del(all_yr_szn_df)