import os
import sys
import numpy as np
import pandas as pd
from glob import glob

from cli import parse_get_huc_scn_info


def clean_df(df, huc=False, pred=True):
    
    # df['HUC_SEASON'] = df[huc].astype('str') + '_' + df['SEASON']
    df.loc[df.SEASON == 'Spring', 'YR_SZN'] = df.loc[df.SEASON == 'Spring', 'YEAR'] * 100 + 0
    df.loc[df.SEASON == 'Summer', 'YR_SZN'] = df.loc[df.SEASON == 'Summer', 'YEAR'] * 100 + 25
    df.loc[df.SEASON == 'Fall', 'YR_SZN'] = df.loc[df.SEASON == 'Fall', 'YEAR'] * 100 + 50
    df.loc[df.SEASON == 'Winter', 'YR_SZN'] = df.loc[df.SEASON == 'Winter', 'YEAR'] * 100 + 75
    df = df.sort_values(by=['YR_SZN'])
    
    if pred:
        df['PR_WATER'] = np.exp(df['PRED_LOG_WATER']) - 10e-6
    else:
        df['OBS_WATER'] = df['PR_WATER']

    if huc:
        df = df[(df[huc]==3020201)&(df['YEAR']>2005)]

    return(df)


OBS_DF = pd.read_csv('../data/all_data_0118_p2.csv', index_col=0)
OBS_DF = clean_df(OBS_DF, pred=False)


def check_inputs():

    args = parse_get_huc_scn_info()

    if args.scenario not in ['all', 'RCP45_A1B', 'RCP45_A2', 'RCP45_B1', 'RCP45_B2', \
                        'RCP85_A1B', 'RCP85_A2', 'RCP85_B1', 'RCP85_B2']:
        print(f'{args.scenario} not an acceptible input, please try again.')
        sys.exit()
    elif args.scenario == 'all':
        scn_lst = ['RCP45_A1B', 'RCP45_A2', 'RCP45_B1', 'RCP45_B2', \
                   'RCP85_A1B', 'RCP85_A2', 'RCP85_B1', 'RCP85_B2']
    else:
        scn_lst = [args.scenario]

    for huc in args.huc_lst:
        if huc not in OBS_DF.HUC08.unique():
            print(f'{huc} is not a valid HUC. Please try again.')
            sys.exit()

    var_lst = [0] * len(args.variables)
    for i in range(len(args.variables)):
        var = args.variables[i]
        if var.upper() not in ['MEAN', 'CI', 'TREND', 'P_VALUE', 'Z', 'TAU', 'S', 'VAR_S', 'SLOPE', 'INTERCEPT']:
            print(f'{var} is not a valid variable. Please enter a list of at least one of the following:\
                  \nMEAN, CI, TREND, P_VALUE, Z, TAU, S, VAR_S, SLOPE, INTERCEPT')
            sys.exit()
        else:
            var_lst[i] = var.upper()

    if args.years not in ['all', '2040', '2070', '2099']:
        print(f'{args.years} not an acceptible input, please try again.')
        sys.exit()
    
    if args.season not in ['spring', 'summer', 'fall', 'winter']:
        print(f'{args.season} not an acceptible input, please try again.')
        sys.exit()

    return(scn_lst, args.huc_lst, var_lst, args.years, args.season)


def print_yr_szn_mean_ci(season:str, df:pd.DataFrame, year:str, mk_dict:dict, mk_vars:list):

    if season == 'all':
        season_lst = ['spring','summer','fall', 'winter']
    else:
        season_lst = [season]

    if year == 'all':
        year_lst = [2006, 2040, 2070, 2099]
        key_lst = [2040, 2070, 2099, '2006-2099']
    else:
        year_lst = [int(year)]
        key_lst = [int(year)]

    for szn in season_lst:
        if szn == 'spring':
            yr_szn_lst = [yr * 100 for yr in year_lst]

        elif szn == 'summer':
            yr_szn_lst = [yr * 100 + 25 for yr in year_lst]

        elif szn == 'fall':
            yr_szn_lst = [yr * 100 + 50 for yr in year_lst]

        elif szn == 'winter':
            yr_szn_lst = [yr * 100 + 75 for yr in year_lst]

        for i in range(len(yr_szn_lst)):
            yr = yr_szn_lst[i]
            key = key_lst[i]
            mk_df = mk_dict[key]
            avg = df[df.YR_SZN == yr].MEAN.values[0] * 100
            ci = (df[df.YR_SZN == yr].MEAN.values[0] - df[df.YR_SZN == yr].LOWER_95_CI.values[0]) * 100
            print(f'{str(yr)[0:4]}:\t{round(avg, 4)}\t{round(ci, 4)}', end=" ")
            for var in mk_vars:
                if var == 'TREND':
                    col = 'TREND_DIR'
                    print(f'\t{mk_df[col].values[0]}', end=" ")
                else:
                    col = var
                    print(f'\t{round(mk_df[col].values[0], 6)}\t', end=" ")

            print()

def main():
    scn_lst, huc_lst, var_lst, years, season = check_inputs()

    for scn in scn_lst:
        print(f'\n\n{scn} INFO:\n')
        mk_df_path_lst = glob(f'../data/FutureData/GCM_FORESCE_CSVs/HUC_MK/MULTIMODEL_{scn}*.csv')
        mk_df_path_lst.sort()
        mk_full_df_dict = {}
        for i in range(4):
            yr = [2040, 2070, 2099, '2006-2099'][i]
            df = pd.read_csv(mk_df_path_lst[i], index_col=0)
            mk_full_df_dict[yr] = df

        full_huc_mk_df = pd.read_csv(f'../data/FutureData/GCM_FORESCE_CSVs/HUC_CI/MULTIMODEL_{scn}_MC_HUC_CI95.csv', index_col=0)

        for huc in huc_lst:
            print(f'\n{huc}:')
            if 'MEAN' in var_lst or 'CI' in var_lst:
                mk_var_lst = var_lst.copy()
                mk_var_lst.remove('MEAN')
                mk_var_lst.remove('CI')

                huc_mk_df = full_huc_mk_df[full_huc_mk_df.HUC08 == huc]
                print(season)
                print(f'      \tMEAN\t+/-', end=" ")
                for i in range(len(mk_var_lst)):
                    print(f'\t{mk_var_lst[i]}\t', end=" ")
                print()
                if years == 'all':
                    mk_df_dict = {}
                    for key in mk_full_df_dict.keys():
                        df = mk_full_df_dict[key]     
                        df = df[df.HUC == huc]
                        mk_df_dict[key] = df

                    print_yr_szn_mean_ci(season, huc_mk_df, years, mk_df_dict, mk_var_lst)


if __name__ == '__main__':
    main()