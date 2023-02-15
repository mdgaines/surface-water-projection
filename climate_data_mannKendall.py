import os
import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt
from glob import glob

import pymannkendall as mk


def clean_df(df): #, huc=False, add_rand=False):
    
    # df['HUC_SEASON'] = df[huc].astype('str') + '_' + df['SEASON']
    df.loc[df.SEASON == "Spring", "YR_SZN"] = df.YEAR * 100 + 0
    df.loc[df.SEASON == "Summer", "YR_SZN"] = df.YEAR * 100 + 25
    df.loc[df.SEASON == "Fall", "YR_SZN"] = df.YEAR * 100 + 50
    df.loc[df.SEASON == "Winter", "YR_SZN"] = df.YEAR * 100 + 75
    df = df.sort_values(by=['YR_SZN'])
    # if add_rand:
    #     df['PRED_WATER'] = df['PRED_LOG_WATER'] #np.exp(df['PRED_LOG_WATER']+np.random.normal(0,1))
    # else:
    #     df['PRED_WATER'] = df['PRED_LOG_WATER'] #np.exp(df['PRED_LOG_WATER'])
    # if huc:
    #     df = df[(df[huc]==3020201)&(df['YEAR']>2005)]

    return(df)


#### DATA PATHS
precip_paths = glob('../data/ClimateData/macav2livneh_studyArea_avgs/*PRECIP.csv')

mxtemp_paths = glob('../data/ClimateData/macav2livneh_studyArea_avgs/*MAX-TEMP.csv')


#### GFDL PRECIP
gfdl_85_precip_path = '../data/ClimateData/macav2livneh_studyArea_avgs/GFDL-ESM2M_RCP85_PRECIP.csv'
gfdl_45_precip_path = '../data/ClimateData/macav2livneh_studyArea_avgs/GFDL-ESM2M_RCP45_PRECIP.csv'
gfdl_HISTORICAL_precip_path = '../data/ClimateData/macav2livneh_studyArea_avgs/GFDL-ESM2M_HISTORICAL_PRECIP.csv'


gfdl_85_precip = pd.read_csv(gfdl_85_precip_path, index_col=0)
gfdl_85_precip['YEAR'] = gfdl_85_precip.apply(lambda row: row['DATE'][0:4], axis=1)

mk.seasonal_test(gfdl_85_precip['PRECIP_AVG'], period=12)


mk_res_lst = [0] * 30
fl_lst = precip_paths + mxtemp_paths
for i in range(len(mk_res_lst)):
    
    temp_df = pd.read_csv(fl_lst[i], index_col=0)
    if i < 15:
        var = 'PRECIP_AVG'
    else:
        var = 'MAX_TEMP_AVG'
    
    if 'HIST' in fl_lst[i]:
        res = mk.seasonal_test(temp_df[504:][var], period=12)
    else:
        res = mk.seasonal_test(temp_df[var], period=12)


    mk_res_lst[i] = [os.path.basename(fl_lst[i])[:-4], res.trend, res.h, res.p, res.z, res.Tau, res.s, res.var_s, res.slope, res.intercept]

    # mk_savepng(np.asarray(temp_df['PR_WATER']), res, huc_lst[i])

mk_res_df = pd.DataFrame(mk_res_lst, columns=['HUC','TREND_DIR','TREND_PRES','P_VALUE','Z','TAU','S','VAR_S','SLOPE','INTERCEPT'])
mk_res_df
