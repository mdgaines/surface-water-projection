import os
import pandas as pd
import numpy as np
from glob import glob
import seaborn as sns
from scipy.stats import pearsonr
import json

import matplotlib.pyplot as plt


# function to add to JSON modified from https://www.geeksforgeeks.org/append-to-json-file-using-python/
def write_json(new_data=dict, filename=str, new_json=bool):

    if new_json:
        origin_dict = {"bivariate_info": [new_data]}
        with open(filename, 'w') as file:
            json.dump(origin_dict, file) 
    else:
        with open(filename,'r+') as file:
            # First we load existing data into a dict.
            file_data = json.load(file)
            # Join new_data with file_data inside emp_details
            file_data["bivariate_info"].append(new_data)
            # Sets file's current position at offset.
            file.seek(0)
            # convert back to json.
            json.dump(file_data, file, indent = 4)
    return


def mk_year_season_csv(outpath = str):
    '''
        Makes CSVs with data (projected or observed) from 1985-2022 for each season
        and climate/land cover variable.
        
        parameter:
            outpath: path to where the csv will be saved
                climate: '../data/ClimateData/macav2livneh_GRIDMET_CSVs/{src}/{src}_{szn}_{var}_AVG_{scn (opitonal)}_{tag}.csv'
                lclu   : '../data/LandCover/FORESCE_NLCDCDL_CSVs/{scn}/{src}_{var}_FRAC_{scn}_{tag}.csv'
    '''
    var_group = outpath.split('/')[2]
    info_lst = os.path.basename(outpath)[:-4].split('_')
    src = info_lst[0] # data source (GRIDMET, GFDL, etc.; FORESCE, NLCDCDL for LCLU)
    scn = info_lst[-2] # scenario (HISTORICAL, RCP45, RCP85 for GCMs; A1B, A2, B1, B2 for LCLU)

    if var_group == 'ClimateData':
        szn = info_lst[1] # season (Sp, Su, Fa, Wi for GRIDEMT; SPRING, SUMMER, FALL, WINTER for GCMs)
        if 'TEMP' in outpath:
            var = '_'.join([info_lst[2], info_lst[3]])
        else:
            var = info_lst[2] # variable (Pr, maxTemp, minTemp for GRIDMET; PRECIP, MAX_TEMP, MIN_TEMP for GCMs)

        if src=='GRIDMET':
            fl_lst = glob(f'../data/ClimateData/GRIDMET_AVG_STDEV/GRIDMET_YR_AVG/*{szn}_{var}*.csv')
        else:
            fl_lst = glob(f'../data/ClimateData/{src}*/{scn}/zonal_avg/*{szn}_{var}*.csv')
    
    else: # LandCover
        var_dict = {'AGRI': ['AG', 'agro'],\
                    'INTS': ['INT', 'intense'],\
                    'FRST': ['NAT', 'natural']}
        if src == 'FORESCE':
            var = var_dict[info_lst[1]][0]
            fl_lst = glob(f'../data/LandCover/FORE-SCE/*{scn}/zonal_lclu_csv/*.csv')
        else:
            var = var_dict[info_lst[1]][1]
            fl_lst = glob(f'../data/LandCover/CDL_paper2/*{var}*.csv')

    # loop through files and get climate var for each year
    for i in range(len(fl_lst)):
        fl = pd.read_csv(fl_lst[i])
        if src == 'GRIDMET':
            col_name = 'mean'
            yr = fl['Yr_Szn'][0][:4]
        elif src != 'GRDIMET' and var_group == 'ClimateData':
            col_name = f'AVG_{var}'
            yr = fl['YEAR'][0]
        elif src == 'FORESCE':
            col_name = f'PR_{var}'
            yr = str(fl['YEAR'][0])
        elif src == 'NLCDCDL':
            fl['PR_LC'] = (fl['sum'] * 0.0009 ) / fl['areasqkm'] # get percent LCLU
            col_name = 'PR_LC'
            yr = fl['Yr_Szn'][0][:4]

        if i == 0:
            full_df = fl[['huc8', col_name]]
            full_df = full_df.rename(columns={col_name:yr})
        else:
            full_df = full_df.join(fl[['huc8', col_name]].set_index('huc8'), on='huc8')
            full_df = full_df.rename(columns={col_name:yr})

    if var_group == 'LandCover':
        full_df = full_df.reindex(sorted(full_df.columns), axis=1)
    # save CSV
    full_df.to_csv(outpath)
    print(f'{os.path.basename(outpath)} saved.')
    return()


def mk_diff_csv(obs_path=str, src_path=str, src=str, szn=str, var=str, scn=str):

    var_group = obs_path.split('/')[2]

    obs_df = pd.read_csv(obs_path, index_col=0)
    obs_df.loc[:, 'huc4'] = [str(i)[0:4] for i in obs_df['huc8'].to_list()]
    src_df = pd.read_csv(src_path, index_col=0)
    src_df.loc[:, 'huc4'] = [str(i)[0:4] for i in src_df['huc8'].to_list()]
    huc4_lst = obs_df.huc4.unique()
    
    if var_group == 'ClimateData':
        outpath_lst = [f'../data/ClimateData/macav2livneh_GRIDMET_diff_CSVs/{src}/{szn}/{var}/{scn}/{src}_{szn}_{var}_DIFF_{scn}_{huc4}.csv'\
            for huc4 in huc4_lst]
    elif var_group == 'LandCover':
        outpath_lst = [f'../data/LandCover/FORESCE_NLCDCDL_diff_CSVs/{scn}/{var}/{src}_{var}_DIFF_{scn}_{huc4}.csv'\
            for huc4 in huc4_lst]

    for outpath in outpath_lst:
        if not os.path.exists(outpath):
            if not os.path.exists(os.path.dirname(outpath)):
                os.makedirs(os.path.dirname(outpath), exist_ok=True)

            huc4 = outpath[:-4].split('_')[-1]  # get huc4 name
            obs_huc_df = obs_df[obs_df['huc4']==huc4]   # get obs data for huc4
            obs_huc_df = obs_huc_df.set_index('huc8')
            src_huc_df = src_df[src_df['huc4']==huc4]   # get src data for huc4
            src_huc_df = src_huc_df.set_index('huc8')
            yr_lst = [i for i in obs_huc_df.columns if i in src_huc_df.columns] # get years that both obs and src have
            yr_lst.sort()
            yr_lst = yr_lst[:-1]
            diff_df = src_huc_df[yr_lst] - obs_huc_df[yr_lst]
            diff_df.to_csv(outpath)
            print(f'{os.path.basename(outpath)} saved.')

    return()


def save_error_png(outpath=str, diff_path = str, dist=True):

    if not os.path.exists(outpath):
        if not os.path.exists(os.path.dirname(outpath)):
            os.makedirs(os.path.dirname(outpath), exist_ok=True)

        diff_df = pd.read_csv(diff_path, index_col='huc8')
        if dist:
            fig, ax1 = plt.subplots(figsize=(12, 8))
            diff_df.stack().hist(ax=ax1, bins=20)
            ax1.set_title(os.path.basename(diff_path)[:-4], fontsize=14)
            ax1.set_xlabel('PROJ - OBS', fontsize=12)
            ax1.set_ylabel('Count', fontsize=12)
            fig.savefig(outpath, bbox_inches='tight', facecolor='white')

        else:
            fig, ax1 = plt.subplots(figsize=(12, 8))
            diff_df.T.plot(ax=ax1)
            ax1.set_title(os.path.basename(diff_path)[:-4], fontsize=14)
            ax1.set_xlabel('Year', fontsize=12)
            ax1.set_ylabel('PROJ - OBS', fontsize=12)
            ax1.legend(fontsize=10)
            fig.savefig(outpath, bbox_inches='tight', facecolor='white')

        plt.close(fig)

    return()


def calc_error_correlation(precip_path=str, mxTemp_path=str, png_outpath=str, \
                           src=str, szn=str, scn=str, huc4=str):
    '''
        Only for MaxTemp and Precip. We can look into MinTemp and tri-variate correlation another day.
    '''
    bivar_json_path = f'../data/ClimateData/macav2livneh_GRIDMET_bivar_JSONs/{src}/{szn}/{src}_{szn}_PREICP_MAXT_BIVAR_{scn}.json'
    new_json= False
    if not os.path.exists(bivar_json_path):
        if not os.path.exists(os.path.dirname(bivar_json_path)):
            os.makedirs(os.path.dirname(bivar_json_path), exist_ok=True)
        new_json = True

    precip = pd.read_csv(precip_path)
    new_precip_df = pd.DataFrame()
    precip = precip.set_index('huc8')
    new_precip_df['HUC_YR'] = ['_'.join([str(i), yr]) for i in precip.index for yr in precip.columns]
    new_precip_df['PRECIP_ERROR'] = precip.stack().values

    mxtemp = pd.read_csv(mxTemp_path)
    new_mxtemp_df = pd.DataFrame()
    mxtemp = mxtemp.set_index('huc8')
    new_mxtemp_df['HUC_YR'] = ['_'.join([str(i), yr]) for i in mxtemp.index for yr in mxtemp.columns]
    new_mxtemp_df['MAX_TEMP_ERROR'] = mxtemp.stack().values

    joint_df = new_precip_df.merge(new_mxtemp_df, on='HUC_YR')

    rho, p_val = pearsonr(joint_df.PRECIP_ERROR, joint_df.MAX_TEMP_ERROR)

    json_dict = {'HUC04':huc4}

    mu_x = joint_df.PRECIP_ERROR.mean()
    sigma_x = joint_df.PRECIP_ERROR.std()

    mu_y = joint_df.MAX_TEMP_ERROR.mean()
    sigma_y = joint_df.MAX_TEMP_ERROR.std()
    
    if p_val < 0.01:
        json_dict['SIGMA'] = [[sigma_x**2, rho*sigma_x*sigma_y], [rho*sigma_x*sigma_y, sigma_y**2]]
    
    else:
        print(f'MAX_TEMP and PRECIP errors are independent for {os.path.basename(png_outpath)[:-4]}')
        json_dict['SIGMA'] = [sigma_x, sigma_y]

    json_dict['MU'] = [mu_x, mu_y]
    json_dict['RHO'] = rho

    write_json(json_dict, bivar_json_path, new_json)    

    if not os.path.exists(png_outpath):
        if not os.path.exists(os.path.dirname(png_outpath)):
            os.makedirs(os.path.dirname(png_outpath), exist_ok=True)
        
        # plot and save with R on graph
        # fig, ax1 = plt.subplots(figsize=(12, 8))
        fig = sns.jointplot(x=joint_df.PRECIP_ERROR, y=joint_df.MAX_TEMP_ERROR)
        plt.text(joint_df.PRECIP_ERROR.min(), joint_df.MAX_TEMP_ERROR.min(), \
            f"R: {round(rho,4)}\np-val: {round(p_val,4)}",\
            horizontalalignment='left', size='medium', color='black', weight='semibold')
        plt.suptitle(f'{os.path.basename(png_outpath)[:-4]}\n', fontsize=14)
        plt.savefig(png_outpath, bbox_inches='tight', facecolor='white')
        plt.close()

    return

def calc_trivar_error_correlation(agri_path=str, ints_path=str, frst_path=str, src=str, scn=str, huc4=str):

    trivar_json_path = f'../data/LandCover/FORESCE_NLCDCDL_trivar_JSONs/{src}_PREICP_MAXT_trivar_{scn}.json'
    new_json= False
    if not os.path.exists(trivar_json_path):
        if not os.path.exists(os.path.dirname(trivar_json_path)):
            os.makedirs(os.path.dirname(trivar_json_path), exist_ok=True)
        new_json = True

    agri = pd.read_csv(agri_path)
    new_agri_df = pd.DataFrame()
    agri = agri.set_index('huc8')
    new_agri_df['HUC_YR'] = ['_'.join([str(i), yr]) for i in agri.index for yr in agri.columns]
    new_agri_df['AGRI_ERROR'] = agri.stack().values

    ints = pd.read_csv(ints_path)
    new_ints_df = pd.DataFrame()
    ints = ints.set_index('huc8')
    new_ints_df['HUC_YR'] = ['_'.join([str(i), yr]) for i in ints.index for yr in ints.columns]
    new_ints_df['INTS_ERROR'] = ints.stack().values

    frst = pd.read_csv(frst_path)
    new_frst_df = pd.DataFrame()
    frst = frst.set_index('huc8')
    new_frst_df['HUC_YR'] = ['_'.join([str(i), yr]) for i in frst.index for yr in frst.columns]
    new_frst_df['FRST_ERROR'] = frst.stack().values

    joint_df = new_agri_df.merge(new_ints_df, on='HUC_YR')
    joint_df = joint_df.merge(new_frst_df, on='HUC_YR')

    rho_xy, p_val_xy = pearsonr(joint_df.AGRI_ERROR, joint_df.INTS_ERROR)
    rho_xz, p_val_xz = pearsonr(joint_df.AGRI_ERROR, joint_df.FRST_ERROR)
    rho_yz, p_val_yz = pearsonr(joint_df.INTS_ERROR, joint_df.FRST_ERROR)

    p_val_lst = [p_val_xy, p_val_xz, p_val_yz]

    json_dict = {'HUC04':huc4}

    mu_x = joint_df.AGRI_ERROR.mean()
    sigma_x = joint_df.AGRI_ERROR.std()

    mu_y = joint_df.INTS_ERROR.mean()
    sigma_y = joint_df.INTS_ERROR.std()

    mu_z = joint_df.FRST_ERROR.mean()
    sigma_z = joint_df.FRST_ERROR.std()
    
    if len([i for i in p_val_lst if i < 0.01]) >= 2:
        json_dict['SIGMA_ALL'] = [[sigma_x**2, rho_xy*sigma_x*sigma_y, rho_xz*sigma_x*sigma_z], 
                              [rho_xy*sigma_x*sigma_y, sigma_y**2, rho_yz*sigma_y*sigma_z],
                              [rho_xz*sigma_x*sigma_z, rho_yz*sigma_y*sigma_z, sigma_z**2]]
    elif p_val_xy < 0.01:
        json_dict['SIGMA_AGRI_INTS'] = [[sigma_x**2, rho_xy*sigma_x*sigma_y],
                                        [rho_xy*sigma_x*sigma_y, sigma_y**2]]
        json_dict['SIGMA_FRST'] = sigma_z
    elif p_val_xz < 0.01:
        json_dict['SIGMA_AGRI_FRST'] = [[sigma_x**2, rho_xz*sigma_x*sigma_z],
                                        [rho_xz*sigma_x*sigma_z, sigma_z**2]]
        json_dict['SIGMA_INTS'] = sigma_y
    elif p_val_yz < 0.01:
        json_dict['SIGMA_INTS_FRST'] = [[sigma_y**2, rho_yz*sigma_y*sigma_z],
                                        [rho_yz*sigma_y*sigma_z, sigma_z**2]]
        json_dict['SIGMA_AGRI'] = sigma_x
    
    else:
        print(f'MAX_TEMP and AGRI errors are independent for {os.path.basename(png_outpath)[:-4]}')
        json_dict['SIGMA'] = [sigma_x, sigma_y]

    json_dict['MU'] = [mu_x, mu_y, mu_z]
    json_dict['RHO'] = [rho_xy, rho_xz, rho_yz]
    json_dict['P_VALS'] = p_val_lst

    write_json(json_dict, trivar_json_path, new_json)    

    return

### MAIN ###
# 1) make all CSVs OBS and PROJ
# 2) calculate diff between obs and proj and save to csv
#    - raw diff, mean percent error, median percent error
#    - try to figure out which is best to use for MC bootstrapping
# 3) plot error dist for each szn, huc4, variable
# 4) check for correlation/calculate mu and sigma
# 5) REPEAT for LCLU

# ) check for temporal trend in error
# New PY file) fit distributions...

def main():
    src_lst = ['GRIDMET', 'GFDL', 'HadGEM2', 'IPSL', 'MIROC5', 'NorESM1', 'FORESCE', 'NLCDCDL']
    obs_proj_dict = {'OBS': [['Sp', 'Su', 'Fa', 'Wi'], ['Pr', 'maxTemp', 'minTemp']],\
                     'PROJ': [['SPRING', 'SUMMER', 'FALL', 'WINTER'], ['PRECIP', 'MAX_TEMP', 'MIN_TEMP']]}

    ###############################################
    #### make csvs ####
    ###############################################
    for src in src_lst:
        if src == 'GRIDMET' or src == 'NLCDCDL':
            tag = 'OBS'
            scn_lst = ['OBS']
        elif src == 'FORESCE':
            tag = 'PROJ'
            scn_lst = ['A1B', 'A2', 'B1', 'B2']
        else:
            tag = 'PROJ'
            scn_lst = ['HISTORICAL_PROJ', 'RCP45_PROJ', 'RCP85_PROJ']
        szn_lst = obs_proj_dict[tag][0]
        var_lst = obs_proj_dict[tag][1]

        outpath_lst = [f'../data/ClimateData/macav2livneh_GRIDMET_CSVs/{src}/{src}_{szn}_{var}_AVG_{scn}.csv'\
                        for szn in szn_lst for var in var_lst for scn in scn_lst]

        if src == 'FORESCE':
            var_lst = ['AGRI', 'INTS', 'FRST']
            outpath_lst = [f'../data/LandCover/FORESCE_NLCDCDL_CSVs/{scn}/{src}_{var}_FRAC_{scn}_{tag}.csv' \
                            for var in var_lst for scn in scn_lst]
        elif src == 'NLCDCDL':
            var_lst = ['AGRI', 'INTS', 'FRST']
            outpath_lst = [f'../data/LandCover/FORESCE_NLCDCDL_CSVs/{src}/{src}_{var}_FRAC_{tag}.csv' \
                            for var in var_lst]

        for outpath in outpath_lst:
            if not os.path.exists(os.path.dirname(outpath)):
                os.makedirs(os.path.dirname(outpath))
            if not os.path.exists(outpath):
                mk_year_season_csv(outpath)
    print('All OBS and PROJ csv have been saved.')
    ###############################################
    
    ###############################################
    #### make diff csvs ####
    ###############################################
    var_group_lst = ['Climate', 'LandCover']
    for var_group in var_group_lst:
        if var_group == 'Climate':
            obs_lst = glob('../data/ClimateData/macav2livneh_GRIDMET_CSVs/GRIDMET/*.csv')
            obs_info_lst = {'Sp':'SPRING', 'Su':'SUMMER', 'Fa':'FALL', 'Wi':'WINTER',\
                'Pr':'PRECIP', 'maxTemp':'MAX_TEMP', 'minTemp':'MIN_TEMP'}
            scn_lst = ['HISTORICAL', 'RCP45', 'RCP85']
            src_lst = ['GRIDMET', 'GFDL', 'HadGEM2', 'IPSL', 'MIROC5', 'NorESM1']

        elif var_group == 'LandCover':
            obs_lst = glob('../data/LandCover/FORESCE_NLCDCDL_CSVs/NLCDCDL/*.csv')
            scn_lst = ['A1B', 'A2', 'B1', 'B2']

        for obs_path in obs_lst:
            info_lst = os.path.basename(obs_path)[:-4].split('_')
            if var_group == 'Climate':
                szn = obs_info_lst[info_lst[1]]
                var = obs_info_lst[info_lst[2]]

                for scn in scn_lst:
                    for src in src_lst:
                        if src == 'GRIDMET':
                            continue

                        src_path = glob(f'../data/ClimateData/macav2livneh_GRIDMET_CSVs/{src}/*{szn}_{var}*{scn}*.csv')[0]
                        mk_diff_csv(obs_path, src_path, src, szn, var, scn)
            elif var_group == 'LandCover':
                var = info_lst[1]
                for scn in scn_lst:
                    src_path = glob(f'../data/LandCover/FORESCE_NLCDCDL_CSVs/{scn}/*{var}*{scn}*.csv')[0]
                    mk_diff_csv(obs_path, src_path, src='FORESCE', var=var, scn=scn)

    print('All DIFF csv have been saved.')
    ###############################################

    ###############################################
    #### make plots ####
    ###############################################
    huc4_lst = ['3150', '3160', '3140', '8020', '8040', '3100', '3090', '3080',
                '3110', '3130', '3060', '3070', '3120', '8090', '8080', '8070',
                '8050', '8060', '3180', '8030', '3170', '3040', '3030', '3020',
                '3010', '3050', '6010', '6020', '8010', '6040', '6030']
    for var_group in var_group_lst:
        for huc4 in huc4_lst:
            if var_group == 'Climate':
                huc4_diff_lst = glob(f'../data/ClimateData/macav2livneh_GRIDMET_diff_CSVs/*/*/*/*/*{huc4}*.csv')
            else:
                huc4_diff_lst = glob(f'../data/LandCover/FORESCE_NLCDCDL_diff_CSVs/*/*/*{huc4}*.csv')
            for huc4_diff_path in huc4_diff_lst:
                info_lst = os.path.basename(huc4_diff_path)[:-4].split('_')
                src = info_lst[0] # data source (GFDL, etc.)
                scn = info_lst[-2] # scenario (HISTORICAL, RCP45, RCP85 for GCMs)
                if var_group == 'Climate':
                    szn = info_lst[1] # season (SPRING, SUMMER, FALL, WINTER for GCMs)
                    if 'TEMP' in huc4_diff_path:
                        var = '_'.join([info_lst[2], info_lst[3]])
                    else:
                        var = info_lst[2] # variable (PRECIP, MAX_TEMP, MIN_TEMP for GCMs)
                    
                    outpath = f'../imgs/Paper2/error_temporal/{src}/{szn}/{var}/{scn}/{src}_{szn}_{var}_DIFF_{scn}_{huc4}_error_temporal.png'    
                    save_error_png(outpath, huc4_diff_path, dist=False)

                    outpath = f'../imgs/Paper2/error_dist/{src}/{szn}/{var}/{scn}/{src}_{szn}_{var}_DIFF_{scn}_{huc4}_error_dist.png'    
                    save_error_png(outpath, huc4_diff_path, dist=True)
                else:
                    var = info_lst[1]
                    outpath =  f'../imgs/Paper2/error_temporal/{src}/{scn}/{var}/{src}_{var}_DIFF_{scn}_{huc4}_error_temporal.png'
                    save_error_png(outpath, huc4_diff_path, dist=False)

                    outpath =  f'../imgs/Paper2/error_dist/{src}/{scn}/{var}/{src}_{var}_DIFF_{scn}_{huc4}_error_dist.png'
                    save_error_png(outpath, huc4_diff_path, dist=True)

    print('All DIFF PNGs have been saved.')
    ###############################################

    ###############################################
    #### make bivariate plots and save bivaraite variables ####
    ###############################################
    for var_group in var_group_lst:
        for huc4 in huc4_lst:
            if var_group == 'Climate':
                huc4_precip_lst = glob(f'../data/ClimateData/macav2livneh_GRIDMET_diff_CSVs/*/*/PRECIP/*/*{huc4}*.csv')
                huc4_precip_lst.sort()
                huc4_mxTemp_lst = glob(f'../data/ClimateData/macav2livneh_GRIDMET_diff_CSVs/*/*/MAX_TEMP/*/*{huc4}*.csv')    
                huc4_mxTemp_lst.sort()

                for i in range(len(huc4_precip_lst)):
                    info_lst = huc4_mxTemp_lst[i].split('\\')
                    src = info_lst[1]
                    szn = info_lst[2]
                    scn = info_lst[4]

                    png_outpath = f'../imgs/Paper2/error_bivar_dist/{src}/{szn}/PRECIP_MAXT/{scn}/{src}_{szn}_PRECIP_MAXT_BIVAR_{scn}_{huc4}_error_dist.png'
                    
                    calc_error_correlation(precip_path=huc4_precip_lst[i],
                                        mxTemp_path=huc4_mxTemp_lst[i],
                                        png_outpath=png_outpath,
                                        src=src, szn=szn, scn=scn, huc4=huc4)
            else:
                huc4_agri_lst = glob(f'../data/LandCover/FORESCE_NLCDCDL_diff_CSVs/*/AGRI/*{huc4}*.csv')
                huc4_agri_lst.sort()
                huc4_ints_lst = glob(f'../data/LandCover/FORESCE_NLCDCDL_diff_CSVs/*/INTS/*{huc4}*.csv')
                huc4_ints_lst.sort()
                huc4_frst_lst = glob(f'../data/LandCover/FORESCE_NLCDCDL_diff_CSVs/*/FRST/*{huc4}*.csv')
                huc4_frst_lst.sort()

                for i in range(len(huc4_agri_lst)):
                    info_lst = huc4_agri_lst[i].split('\\')
                    scn = info_lst[1]

                    png_outpath = f'../imgs/Paper2/error_multivar_dist/{src}/{scn}/AGRI_INTS_FRST/{src}_AGRI_INTS_FRST_MULTIIVAR_{scn}_{huc4}_error_dist.png'

                    calc_trivar_error_correlation(agri_path=huc4_agri_lst[i],
                                        ints_path=huc4_ints_lst[i],
                                        frst_path=huc4_frst_lst[i],
                                        src=src, scn=scn, huc4=huc4)

    print('All BIVAR JSONs and PNGs have been saved.')
    ###############################################


if __name__ == '__main__':
    main()