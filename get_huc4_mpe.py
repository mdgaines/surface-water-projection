import os
import pandas as pd
import numpy as np
from glob import glob
import seaborn as sns
from scipy.stats import pearsonr
import json

import matplotlib.pyplot as plt


def main():



    obs_lst = glob('../data/ClimateData/macav2livneh_GRIDMET_CSVs/GRIDMET/*.csv')
    obs_info_lst = {'Sp':'SPRING', 'Su':'SUMMER', 'Fa':'FALL', 'Wi':'WINTER',\
        'Pr':'PRECIP', 'maxTemp':'MAX_TEMP', 'minTemp':'MIN_TEMP'}
    scn_lst = ['RCP45', 'RCP85'] #'HISTORICAL', 
    src_lst = ['GFDL', 'HadGEM2', 'IPSL', 'MIROC5', 'NorESM1']

    # src_lst = ['GRIDMET', 'GFDL', 'HadGEM2', 'IPSL', 'MIROC5', 'NorESM1', 'FORESCE', 'NLCDCDL']
    obs_proj_dict = {'OBS': [['Sp', 'Su', 'Fa', 'Wi'], ['Pr', 'maxTemp', 'minTemp']],\
                     'PROJ': [['SPRING', 'SUMMER', 'FALL', 'WINTER'], ['PRECIP', 'MAX_TEMP', 'MIN_TEMP']]}
    var_group_lst = ['Climate']#, 'LandCover']


    huc4_lst = ['3150', '3160', '3140', '8020', '8040', '3100', '3090', '3080',
                '3110', '3130', '3060', '3070', '3120', '8090', '8080', '8070',
                '8050', '8060', '3180', '8030', '3170', '3040', '3030', '3020',
                '3010', '3050', '6010', '6020', '8010', '6040', '6030']
    
    var_szn_lst = []

    for obs_path in obs_lst:
        obs_info = os.path.basename(obs_path).split('_')
        szn = obs_info_lst[obs_info[1]]
        var = obs_info_lst[obs_info[2]]

        # if var == 'MIN_TEMP':
        #     continue

        obs_df = pd.read_csv(obs_path, index_col=0)
        obs_df.loc[:, 'huc4'] = [str(i)[0:4] for i in obs_df['huc8'].to_list()]

        gcm_rcp_dict = {}
        for scn in scn_lst:
            for src in src_lst:
                huc_mpe_dict = {}
                for huc4 in huc4_lst:
                    diff_df = pd.read_csv(f'../data/ClimateData/macav2livneh_GRIDMET_diff_CSVs/{src}/{szn}/{var}/{scn}/{src}_{szn}_{var}_DIFF_{scn}_{huc4}.csv')
                    diff_df.loc[:, 'huc4'] = [str(i)[0:4] for i in diff_df['huc8'].to_list()]

                    obs_huc_df = obs_df[obs_df['huc4']==huc4]   # get obs data for huc4
                    obs_huc_df = obs_huc_df.set_index('huc8')
                    
                    src_huc_df = diff_df[diff_df['huc4']==huc4]   # get src data for huc4
                    src_huc_df = src_huc_df.set_index('huc8')
                    
                    yr_lst = [i for i in obs_huc_df.columns if i in src_huc_df.columns] # get years that both obs and src have
                    yr_lst.sort()
                    yr_lst = yr_lst[:-5]

                    huc_mpe_dict[huc4] = np.mean(np.mean(src_huc_df[yr_lst] / obs_huc_df[yr_lst]))
                
                huc4_df = pd.DataFrame(list(huc_mpe_dict.items()), columns=['HUC4', 'MPE'])
                gcm_rcp_dict[src+'_'+scn] = [huc4_df['MPE'].mean(), var, szn]
        
        var_szn_lst.append([gcm_rcp_dict])


    # GET HUC 04 MPE per GCM
    gcm_rcp_dict = []
    for gcm_rcp in ['GFDL_RCP45', 'HadGEM2_RCP45', 'IPSL_RCP45', 'MIROC5_RCP45', 'NorESM1_RCP45',\
                    'GFDL_RCP85', 'HadGEM2_RCP85', 'IPSL_RCP85', 'MIROC5_RCP85', 'NorESM1_RCP85']:
    
        df = pd.DataFrame(columns=['MPE','VAR','SZN'])
        for i in var_szn_lst:
            df.loc[len(df.index)] = i[0][gcm_rcp]
        gcm_rcp_dict.append([gcm_rcp, df[df['VAR']=='MAX_TEMP']['MPE'].mean(), df[df['VAR']=='PRECIP']['MPE'].mean()])

    gcm_rcp_df = pd.DataFrame(gcm_rcp_dict, columns = ['GCM_RCP', 'MAXT_MPE', 'PRECIP_MPE'])



################## Land Cover #####################
    
    obs_lst = glob('../data/LandCover/FORESCE_NLCDCDL_CSVs/NLCDCDL/*.csv')

    scn_lst = ['A1B', 'A2', 'B1', 'B2'] 
    var_lst = ['AGRI', 'FRST', 'INTS']

    huc4_lst = ['3150', '3160', '3140', '8020', '8040', '3100', '3090', '3080',
                '3110', '3130', '3060', '3070', '3120', '8090', '8080', '8070',
                '8050', '8060', '3180', '8030', '3170', '3040', '3030', '3020',
                '3010', '3050', '6010', '6020', '8010', '6040', '6030']
    
    var_szn_lst = []

    for obs_path in obs_lst:
        if '2011' in obs_path:
            continue

        obs_info = os.path.basename(obs_path).split('_')
        # szn = obs_info_lst[obs_info[1]]
        var = obs_info[1]

        # if var == 'MIN_TEMP':
        #     continue

        obs_df = pd.read_csv(obs_path, index_col=0)
        obs_df.loc[:, 'huc4'] = [str(i)[0:4] for i in obs_df['huc8'].to_list()]

        var_rcp_dict = {}
        for scn in scn_lst:
            huc_mpe_dict = {}
            for huc4 in huc4_lst:
                diff_df = pd.read_csv(f'../data/LandCover/FORESCE_NLCDCDL_diff_CSVs/{scn}/{var}/FORESCE_{var}_DIFF_{scn}_{huc4}.csv')
                diff_df.loc[:, 'huc4'] = [str(i)[0:4] for i in diff_df['huc8'].to_list()]

                obs_huc_df = obs_df[obs_df['huc4']==huc4]   # get obs data for huc4
                obs_huc_df = obs_huc_df.set_index('huc8')
                
                src_huc_df = diff_df[diff_df['huc4']==huc4]   # get src data for huc4
                src_huc_df = src_huc_df.set_index('huc8')
                
                yr_lst = [i for i in obs_huc_df.columns if i in src_huc_df.columns] # get years that both obs and src have
                yr_lst.sort()
                yr_lst = yr_lst[:-5]

                huc_mpe_dict[huc4] = np.mean(np.mean(src_huc_df[yr_lst] / obs_huc_df[yr_lst]))
            
            huc4_df = pd.DataFrame(list(huc_mpe_dict.items()), columns=['HUC4', 'MPE'])
            # Replacing infinite with nan 
            huc4_df.replace([np.inf, -np.inf], np.nan, inplace=True) 
            # Dropping all the rows with nan values 
            huc4_df.dropna(inplace=True) 

            var_rcp_dict[scn] = [huc4_df['MPE'].mean(), var]
        
        var_szn_lst.append([var_rcp_dict])


    # GET HUC 04 MPE per GCM
    var_rcp_dict = []
    for scn in scn_lst:
    
        df = pd.DataFrame(columns=['MPE','VAR'])
        for i in var_szn_lst:
            df.loc[len(df.index)] = i[0][scn]
        var_rcp_dict.append([scn, df[df['VAR']=='AGRI']['MPE'].mean(), df[df['VAR']=='INTS']['MPE'].mean(),  df[df['VAR']=='FRST']['MPE'].mean()])

    lclu_scn_df = pd.DataFrame(var_rcp_dict, columns = ['scn', 'AGRI_MPE', 'INTS_MPE', 'FRST_MPE'])
