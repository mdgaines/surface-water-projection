import os
import numpy as np 
import pandas as pd 
from glob import glob
import json

import concurrent.futures
import multiprocessing
import threading

from merf import MERF
from sklearn.model_selection import train_test_split, KFold
from sklearn.ensemble import RandomForestRegressor

from merf.viz import plot_merf_training_stats
from sklearn.inspection import plot_partial_dependence
import shap
import math
import time

import seaborn as sns
from matplotlib import cm
from matplotlib.colors import ListedColormap
import matplotlib.pyplot as plt

from sklearn.metrics import r2_score


def print_time(start, end, process = ""):
    """
        prints message with elapsed time of process
    """
    elapsed = end - start
    time_str = time.strftime("%H hr %M min %S sec", time.gmtime(elapsed))
    print(process, "completed in", time_str)
    return


def _get_Z_matrix(Z_values=np.array, Z_vars=list):
    '''
        calculate Z matrix from input variables and selected random effects
    '''
    for i in range(len(Z_vars)):
        Z_vars_array = Z_values[Z_vars[i]].unique()
        if i == 0:
            # Z_vars_array = Z_values[Z_vars[i]].unique()
            Z_df = pd.DataFrame(data=0, columns=Z_vars_array, index=Z_values.index)
        # else:
            # Z_vars_array = np.append(Z_vars_array, Z_values[Z_vars[i]].unique())
        for col in Z_vars_array:
            Z_df[col] = np.where(Z_values[Z_vars[i]] == col, 1, 0)

    return(Z_df)


def get_merf_model(train, Z_vars:list=['SEASON', 'HUC04'], clusters:str='HUC08'):
    
    rf_fe_b = RandomForestRegressor(n_estimators = 1000, random_state = 42)

    mrf = MERF(rf_fe_b, max_iterations=200)
    X_train = train[['PRECIP', 'MAX_TMP', 'PR_AG', 'PR_INT', 'PR_NAT']]
    clusters_train = train[clusters]
    y_train = train['LOG_PR_WATER']

    if Z_vars is None:
        Z_train = np.ones((len(X_train), 1))
    else:
        Z_values = train[Z_vars]
        Z_train = _get_Z_matrix(Z_values, Z_vars)

    mrf.fit(X_train, Z_train, clusters_train, y_train)

    return(mrf)


def get_dswe_split(test_split:float=0.2):
    dswe = pd.read_csv('../data/all_data_0118_p2.csv', index_col=0)
    dswe['HUC_SEASON'] = dswe['HUC08'].astype('str') + '_' + dswe['SEASON']

    dswe['LOG_PR_WATER'] = np.log(dswe['PR_WATER'] + 10e-6)

    # save orignial values pre-center/standardize
    dswe['MAX_TMP_OG'] = dswe['MAX_TMP']
    dswe['PRECIP_OG'] = dswe['PRECIP']
    dswe['PR_AG_OG'] = dswe['PR_AG']
    dswe['PR_NAT_OG'] = dswe['PR_NAT']
    dswe['PR_INT_OG'] = dswe['PR_INT']

    # normalize climate data
    dswe.loc[dswe['SEASON'] == 'Spring',['PRECIP']] = normalize_climate_vars(dswe.loc[dswe['SEASON'] == 'Spring',['HUC08','PRECIP']], \
        szn='SPRING', var='PRECIP')
    dswe.loc[dswe['SEASON'] == 'Spring',['MAX_TMP']] = normalize_climate_vars(dswe.loc[dswe['SEASON'] == 'Spring',['HUC08','MAX_TMP']], \
        szn='SPRING', var='MAX_TEMP')

    dswe.loc[dswe['SEASON'] == 'Summer',['PRECIP']] = normalize_climate_vars(dswe.loc[dswe['SEASON'] == 'Summer',['HUC08','PRECIP']], \
        szn='SUMMER', var='PRECIP')
    dswe.loc[dswe['SEASON'] == 'Summer',['MAX_TMP']] = normalize_climate_vars(dswe.loc[dswe['SEASON'] == 'Summer',['HUC08','MAX_TMP']], \
        szn='SUMMER', var='MAX_TEMP')

    dswe.loc[dswe['SEASON'] == 'Fall',['PRECIP']] = normalize_climate_vars(dswe.loc[dswe['SEASON'] == 'Fall',['HUC08','PRECIP']], \
        szn='FALL', var='PRECIP')
    dswe.loc[dswe['SEASON'] == 'Fall',['MAX_TMP']] = normalize_climate_vars(dswe.loc[dswe['SEASON'] == 'Fall',['HUC08','MAX_TMP']], \
        szn='FALL', var='MAX_TEMP')

    dswe.loc[dswe['SEASON'] == 'Winter',['PRECIP']] = normalize_climate_vars(dswe.loc[dswe['SEASON'] == 'Winter',['HUC08','PRECIP']], \
        szn='WINTER', var='PRECIP')
    dswe.loc[dswe['SEASON'] == 'Winter',['MAX_TMP']] = normalize_climate_vars(dswe.loc[dswe['SEASON'] == 'Winter',['HUC08','MAX_TMP']], \
        szn='WINTER', var='MAX_TEMP')


    # center and standardize independent variables
    dswe['MAX_TMP'] = (dswe['MAX_TMP'] - np.mean(dswe['MAX_TMP'])) / np.std(dswe['MAX_TMP'])
    dswe['PRECIP'] = (dswe['PRECIP'] - np.mean(dswe['PRECIP'])) / np.std(dswe['PRECIP'])

    dswe['PR_AG'] = (dswe['PR_AG'] - np.mean(dswe['PR_AG'])) / np.std(dswe['PR_AG'])
    dswe['PR_NAT'] = (dswe['PR_NAT'] - np.mean(dswe['PR_NAT'])) / np.std(dswe['PR_NAT'])
    dswe['PR_INT'] = (dswe['PR_INT'] - np.mean(dswe['PR_INT'])) / np.std(dswe['PR_INT'])

    dswe['HUC04'] = dswe['HUC08'].apply(lambda x: str(x)[0:3].zfill(4))

    train, test = train_test_split(dswe, test_size=test_split, shuffle=True, random_state=42)

    return(train, test, dswe)


def get_merf_error_stats(
        mrf,
        test:pd.DataFrame, 
        Z_vars:list=['SEASON', 'HUC04'], 
        clusters:str='HUC08'):

    X_test = test[['PRECIP', 'MAX_TMP', 'PR_AG', 'PR_INT', 'PR_NAT']]
    clusters_test = test[clusters]
    y_test = test['LOG_PR_WATER']

    Z_values = test[Z_vars]
    Z_test = _get_Z_matrix(Z_values, Z_vars)

    y_hat_test = mrf.predict(X_test, Z_test, clusters_test)

    mse_test = np.mean((y_test - y_hat_test) ** 2)
    rmspe_test = math.sqrt(np.mean( (y_test - y_hat_test) ** 2)) * 100
    mpe_test = np.mean(((y_test-y_hat_test) / y_test) * 100)
    r2_test = r2_score(y_test, y_hat_test)

    return(mse_test, rmspe_test, mpe_test, r2_test)


def get_merf_variable_importance(
        mrf,
        test:pd.DataFrame):
    
    X_test = test[['PRECIP', 'MAX_TMP', 'PR_AG', 'PR_INT', 'PR_NAT']]

    explainer = shap.TreeExplainer(mrf.trained_fe_model)
    shap_vals = explainer.shap_values(X_test)

    shap.summary_plot(shap_vals, X_test)

    ABS_SHAP(shap_vals, X_test)

    return shap_vals


def ABS_SHAP(df_shap,df):
    #import matplotlib as plt
    # Make a copy of the input data
    shap_v = pd.DataFrame(df_shap)
    feature_list = df.columns
    shap_v.columns = feature_list
    df_v = df.copy().reset_index().drop('index',axis=1)
    
    # Determine the correlation in order to plot with different colors
    corr_list = list()
    for i in feature_list:
        b = np.corrcoef(shap_v[i],df_v[i])[1][0]
        corr_list.append(b)
    corr_df = pd.concat([pd.Series(feature_list),pd.Series(corr_list)],axis=1).fillna(0)
    # Make a data frame. Column 1 is the feature, and Column 2 is the correlation coefficient
    corr_df.columns  = ['Variable','Corr']
    corr_df['Sign'] = np.where(corr_df['Corr']>0,'#2a8be1','#ed5b89')
    
    # Plot it
    shap_abs = np.abs(shap_v)
    k=pd.DataFrame(shap_abs.mean()).reset_index()
    k.columns = ['Variable','SHAP_abs']
    k2 = k.merge(corr_df,left_on = 'Variable',right_on='Variable',how='inner')
    k2 = k2.sort_values(by='SHAP_abs',ascending = True)
    colorlist = k2['Sign']
    ax = k2.plot.barh(x='Variable',y='SHAP_abs',color = colorlist, figsize=(5,6),legend=False)
    ax.set_xlabel("SHAP Value\nBlue = Positive Impact\nRed = Negative Impact")


def setup_merf_future(
        future_df:pd.DataFrame,
        Z_vars:list,
        clusters:str,
        obs_data:pd.DataFrame):
    # obs_data = pd.read_csv('../data/all_data_0118_p2.csv', index_col=0)

    # center and standardize independent variables based on the obs data used to train the model
    future_df['MAX_TMP'] = (future_df['MAX_TMP'] - np.mean(obs_data['MAX_TMP_OG'])) / np.std(obs_data['MAX_TMP_OG'])
    future_df['PRECIP'] = (future_df['PRECIP'] - np.mean(obs_data['PRECIP_OG'])) / np.std(obs_data['PRECIP_OG'])

    future_df['PR_AG'] = (future_df['PR_AG'] - np.mean(obs_data['PR_AG_OG'])) / np.std(obs_data['PR_AG_OG'])
    future_df['PR_NAT'] = (future_df['PR_NAT'] - np.mean(obs_data['PR_NAT_OG'])) / np.std(obs_data['PR_NAT_OG'])
    future_df['PR_INT'] = (future_df['PR_INT'] - np.mean(obs_data['PR_INT_OG'])) / np.std(obs_data['PR_INT_OG'])

    X_future = future_df[['PRECIP', 'MAX_TMP', 'PR_AG', 'PR_INT', 'PR_NAT']]
    Z_values = future_df[Z_vars]
    Z_future = _get_Z_matrix(Z_values, Z_vars)

    clusters_future = future_df[clusters]

    return(X_future, Z_future, clusters_future)
    

def get_random_error_dict_climate(json_path=str):

    # Opening JSON file
    with open(json_path) as json_file:
        data = json.load(json_file)
        json_dict = {}
        for i in range(len(data['bivariate_info'])):
            json_dict[data['bivariate_info'][i]['HUC04']] = data['bivariate_info'][i]

    rand_num_dict = {}
    for key in json_dict:
        mu = json_dict[key]['MU']
        sigma = json_dict[key]['SIGMA']

        if np.array(json_dict[key]['SIGMA']).shape == (2,2):
            rand_num_dict[key] = np.random.multivariate_normal(mu, sigma, 10000)
        else:
            rand_x = np.random.normal(mu[0], sigma[0], 10000)
            rand_y = np.random.normal(mu[1], sigma[1], 10000)
            rand_num_dict[key] = [[rand_x[i], rand_y[i]] for i in range(10000)]

    return(rand_num_dict)


def add_random_error(x, json_dict=dict, i=int, var=int):
    # huc04 = str(x)[:4]
    return(json_dict[str(x)[:4]][i][var])


def get_random_error_dict_lclu(json_path=str):
    
    # Opening JSON file
    with open(json_path) as json_file:
        data = json.load(json_file)
        json_dict = {}
        for i in range(len(data['bivariate_info'])):
            json_dict[data['bivariate_info'][i]['HUC04']] = data['bivariate_info'][i]

    rand_num_dict = {}
    for key in json_dict:
        mu = json_dict[key]['MU']

        if len(json_dict[key]) == 5:
            sigma = json_dict[key]['SIGMA_ALL']
            mu = json_dict[key]['MU']
            rand_num_dict[key] = np.random.multivariate_normal(mu, sigma, 10000)
        elif 'SIGMA_FRST' in json_dict[key].keys() and len(json_dict[key]) == 6:
            # AGRI and INTS are related (xy)
            rand_xy = np.random.multivariate_normal([mu[0], mu[1]], json_dict[key]['SIGMA_AGRI_INTS'], 10000)
            rand_z = np.random.normal(mu[2], json_dict[key]['SIGMA_FRST'], 10000)
            rand_num_dict[key] = [[rand_xy[i][0], rand_xy[i][1], rand_z[i]] for i in range(10000)]
        elif 'SIGMA_INTS' in json_dict[key].keys() and len(json_dict[key]) == 6:
            # AGRI and FRST are related (xz)
            rand_xz = np.random.multivariate_normal([mu[0], mu[2]], json_dict[key]['SIGMA_AGRI_FRST'], 10000)
            rand_y = np.random.normal(mu[1], json_dict[key]['SIGMA_INTS'], 10000)
            rand_num_dict[key] = [[rand_xz[i][0], rand_y[i], rand_xz[i][1]] for i in range(10000)]
        elif 'SIGMA_AGRI' in json_dict[key].keys() and len(json_dict[key]) == 6:
            # INTS and FRST are related (yz)
            rand_yz = np.random.multivariate_normal([mu[1], mu[2]], json_dict[key]['SIGMA_INTS_FRST'], 10000)
            rand_x = np.random.normal(mu[0], json_dict[key]['SIGMA_AGRI'], 10000)
            rand_num_dict[key] = [[rand_x[i], rand_yz[i][0], rand_yz[i][1]] for i in range(10000)]
        else:
            # all independent
            rand_x = np.random.normal(mu[0], json_dict[key]['SIGMA_AGRI'], 10000)
            rand_y = np.random.normal(mu[1], json_dict[key]['SIGMA_INTS'], 10000)
            rand_z = np.random.normal(mu[2], json_dict[key]['SIGMA_FRST'], 10000)
            rand_num_dict[key] = [[rand_x[i], rand_y[i], rand_z[i]] for i in range(10000)]

    return(rand_num_dict)


def normalize_climate_vars(df=pd.DataFrame, szn=str, var=str):

    avgStd_df = pd.read_csv(f'../data/ClimateData/GRIDMET_AVG_STDEV/1979_2008_{szn}_{var}_AVG_STDV.csv', index_col=0)
    avgStd_df = avgStd_df.sort_values('huc8')

    temp_df = df.merge(avgStd_df, left_on='HUC08', right_on='huc8')

    temp_df.iloc[:,1] = (temp_df.iloc[:,1] - temp_df['mean']) / temp_df['std']

    return(temp_df.iloc[:,0:2])


def process(data_fl, 
            outpath, 
            merf_model, 
            dswe, 
            spring_cl_dict, 
            summer_cl_dict, 
            fall_cl_dict, 
            winter_cl_dict, 
            lclu_dict):
    '''
        Processes each Monte Carlo run and saves the output file
    '''

    i = int(outpath.split('_')[-1].split('.')[0])

    # Add random variance to climate data
    data_fl.loc[data_fl['SEASON'] == 'Spring','PRECIP'] += data_fl[data_fl['SEASON'] == 'Spring']['HUC08'].apply(add_random_error, \
        json_dict=spring_cl_dict, i=i, var=0)
    data_fl.loc[data_fl['SEASON'] == 'Spring','MAX_TMP'] += data_fl[data_fl['SEASON'] == 'Spring']['HUC08'].apply(add_random_error, \
        json_dict=spring_cl_dict, i=i, var=1)

    data_fl.loc[data_fl['SEASON'] == 'Summer','PRECIP'] += data_fl[data_fl['SEASON'] == 'Summer']['HUC08'].apply(add_random_error, \
        json_dict=summer_cl_dict, i=i, var=0)
    data_fl.loc[data_fl['SEASON'] == 'Summer','MAX_TMP'] += data_fl[data_fl['SEASON'] == 'Summer']['HUC08'].apply(add_random_error, \
        json_dict=summer_cl_dict, i=i, var=1)

    data_fl.loc[data_fl['SEASON'] == 'Fall','PRECIP'] += data_fl[data_fl['SEASON'] == 'Fall']['HUC08'].apply(add_random_error, \
        json_dict=fall_cl_dict, i=i, var=0)
    data_fl.loc[data_fl['SEASON'] == 'Fall','MAX_TMP'] += data_fl[data_fl['SEASON'] == 'Fall']['HUC08'].apply(add_random_error, \
        json_dict=fall_cl_dict, i=i, var=1)

    data_fl.loc[data_fl['SEASON'] == 'Winter','PRECIP'] += data_fl[data_fl['SEASON'] == 'Winter']['HUC08'].apply(add_random_error, \
        json_dict=winter_cl_dict, i=i, var=0)
    data_fl.loc[data_fl['SEASON'] == 'Winter','MAX_TMP'] += data_fl[data_fl['SEASON'] == 'Winter']['HUC08'].apply(add_random_error, \
        json_dict=winter_cl_dict, i=i, var=1)

    # normalize climate data
    data_fl.loc[data_fl['SEASON'] == 'Spring',['PRECIP']] = normalize_climate_vars(data_fl.loc[data_fl['SEASON'] == 'Spring',['HUC08','PRECIP']], \
        szn='SPRING', var='PRECIP')
    data_fl.loc[data_fl['SEASON'] == 'Spring',['MAX_TMP']] = normalize_climate_vars(data_fl.loc[data_fl['SEASON'] == 'Spring',['HUC08','MAX_TMP']], \
        szn='SPRING', var='MAX_TEMP')

    data_fl.loc[data_fl['SEASON'] == 'Summer',['PRECIP']] = normalize_climate_vars(data_fl.loc[data_fl['SEASON'] == 'Summer',['HUC08','PRECIP']], \
        szn='SUMMER', var='PRECIP')
    data_fl.loc[data_fl['SEASON'] == 'Summer',['MAX_TMP']] = normalize_climate_vars(data_fl.loc[data_fl['SEASON'] == 'Summer',['HUC08','MAX_TMP']], \
        szn='SUMMER', var='MAX_TEMP')

    data_fl.loc[data_fl['SEASON'] == 'Fall',['PRECIP']] = normalize_climate_vars(data_fl.loc[data_fl['SEASON'] == 'Fall',['HUC08','PRECIP']], \
        szn='FALL', var='PRECIP')
    data_fl.loc[data_fl['SEASON'] == 'Fall',['MAX_TMP']] = normalize_climate_vars(data_fl.loc[data_fl['SEASON'] == 'Fall',['HUC08','MAX_TMP']], \
        szn='FALL', var='MAX_TEMP')

    data_fl.loc[data_fl['SEASON'] == 'Winter',['PRECIP']] = normalize_climate_vars(data_fl.loc[data_fl['SEASON'] == 'Winter',['HUC08','PRECIP']], \
        szn='WINTER', var='PRECIP')
    data_fl.loc[data_fl['SEASON'] == 'Winter',['MAX_TMP']] = normalize_climate_vars(data_fl.loc[data_fl['SEASON'] == 'Winter',['HUC08','MAX_TMP']], \
        szn='WINTER', var='MAX_TEMP')

    # add random variace for lclu data
    data_fl['PR_AG'] += data_fl['HUC08'].apply(add_random_error, json_dict=lclu_dict, i=i, var=0)
    data_fl['PR_INT'] += data_fl['HUC08'].apply(add_random_error, json_dict=lclu_dict, i=i, var=1)
    data_fl['PR_NAT'] += data_fl['HUC08'].apply(add_random_error, json_dict=lclu_dict, i=i, var=2)

    # Set up MERF model
    X_future, Z_future, clusters_future = setup_merf_future(data_fl, Z_vars=['SEASON'], clusters='HUC08', obs_data=dswe)
    
    # Run MERF model
    y_hat_test = merf_model.predict(X_future, Z_future, clusters_future)
    data_fl['PRED_LOG_WATER'] = y_hat_test
    # Save file
    data_fl.to_csv(outpath)

    return()


def param_wrapper(p):
    return process(*p)


def main():

    GCM_LST = ['GFDL', 'HadGEM2', 'IPSL', 'MIROC5', 'NorESM1']
    SCENARIO_LST = ['RCP45', 'RCP85']
    FORESCE_LST = ['A1B', 'A2', 'B1', 'B2']

    train, test, dswe = get_dswe_split(test_split=0.2)

    merf_model = get_merf_model(train=train, Z_vars=['SEASON'], clusters='HUC08')

    mse_test, rmspe_test, mpe_test, r2_test = get_merf_error_stats(merf_model, test, Z_vars=['SEASON'], clusters='HUC08')

    print('MSE:', mse_test)
    print('RMSPE:', rmspe_test)
    print('MPE:', mpe_test)
    print('R2:', r2_test)

    shap_vals = get_merf_variable_importance(mrf=merf_model, test=test)
    print(shap_vals)

    for gcm in GCM_LST:
        for foresce in FORESCE_LST:
            for scn in SCENARIO_LST:

                start_time_scenario = time.time()

                data_fl = pd.read_csv(f'../data/FutureData/GCM_FORESCE_CSVs/{gcm}_{scn}_{foresce}_ALL.csv', index_col=0)

                json_cl_lst = glob(f'../data/ClimateData/*JSONs/{gcm}/*/{gcm}*{scn}*.json')
                spring_cl_dict = get_random_error_dict_climate([i for i in json_cl_lst if 'SPRING' in i][0])
                summer_cl_dict = get_random_error_dict_climate([i for i in json_cl_lst if 'SUMMER' in i][0])
                fall_cl_dict = get_random_error_dict_climate([i for i in json_cl_lst if 'FALL' in i][0])
                winter_cl_dict = get_random_error_dict_climate([i for i in json_cl_lst if 'WINTER' in i][0])

                json_lclu = glob(f'../data/LandCover/*JSONs/*{foresce}.json')[0]
                lclu_dict = get_random_error_dict_lclu(json_lclu)

                to_proc_outpath_lst = []

                for i in range(1000):
                    outpath = f'../data/FutureData/GCM_FORESCE_CSVs/{gcm}/{gcm}_{scn}_{foresce}/{gcm}_{scn}_{foresce}_{i}.csv'

                    if not os.path.exists(outpath):
                        if not os.path.exists(os.path.dirname(outpath)):
                            os.makedirs(os.path.dirname(outpath))
                    
                        to_proc_outpath_lst.append(outpath)

                # Add random variance to climate data
                params = ((data_fl, path, merf_model, dswe, \
                           spring_cl_dict, summer_cl_dict, fall_cl_dict, winter_cl_dict, \
                           lclu_dict) for path in to_proc_outpath_lst)

                # # can use ProcessPoolExecutor because we are reading and writing different files in each core
                with concurrent.futures.ProcessPoolExecutor(
                    max_workers=7
                ) as executor:
                    executor.map(param_wrapper, params)

                end_time_scenario = time.time()
                print_time(start_time_scenario, end_time_scenario)
                print(f'{gcm}_{scn}_{foresce} done')

if __name__ == '__main__':
    main()