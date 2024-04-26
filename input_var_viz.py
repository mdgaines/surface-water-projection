import os
import numpy as np 
import pandas as pd 

import math

import seaborn as sns
from matplotlib import cm
from matplotlib.colors import ListedColormap
from matplotlib.ticker import (MultipleLocator, AutoMinorLocator, IndexLocator)
import matplotlib.lines as mlines
import matplotlib.pyplot as plt
import pandas as pd

from glob import glob
#hello

plt.rcParams["font.family"] = "Times New Roman"
plt.rcParams["font.size"] = "20"

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


# import data
def import_foresce_data(WRR=False):
    ###### Historical ######
    hist_frst_df = pd.read_csv(glob('../data/LandCover/FORESCE_NLCDCDL_CSVs/Historical/*FRST*.csv')[0], index_col=0)
    hist_frst_df = hist_frst_df.set_index('huc8')
    hist_frst_avg = hist_frst_df.mean() * 100

    hist_ints_df = pd.read_csv(glob('../data/LandCover/FORESCE_NLCDCDL_CSVs/Historical/*INTS*.csv')[0], index_col=0)
    hist_ints_df = hist_ints_df.set_index('huc8')
    hist_ints_avg = hist_ints_df.mean() * 100

    hist_agri_df = pd.read_csv(glob('../data/LandCover/FORESCE_NLCDCDL_CSVs/Historical/*AGRI*.csv')[0], index_col=0)
    hist_agri_df = hist_agri_df.set_index('huc8')
    hist_agri_avg = hist_agri_df.mean() * 100

    ###### A1B ######
    A1B_frst_df = pd.read_csv(glob('../data/LandCover/FORESCE_NLCDCDL_CSVs/A1B/*FRST*.csv')[0], index_col=0)
    A1B_frst_df = A1B_frst_df.set_index('huc8')
    A1B_frst_avg = A1B_frst_df.mean() * 100

    A1B_ints_df = pd.read_csv(glob('../data/LandCover/FORESCE_NLCDCDL_CSVs/A1B/*INTS*.csv')[0], index_col=0)
    A1B_ints_df = A1B_ints_df.set_index('huc8')
    A1B_ints_avg = A1B_ints_df.mean() * 100

    A1B_agri_df = pd.read_csv(glob('../data/LandCover/FORESCE_NLCDCDL_CSVs/A1B/*AGRI*.csv')[0], index_col=0)
    A1B_agri_df = A1B_agri_df.set_index('huc8')
    A1B_agri_avg = A1B_agri_df.mean() * 100

    #### A2 ####
    A2_frst_df = pd.read_csv(glob('../data/LandCover/FORESCE_NLCDCDL_CSVs/A2/*FRST*.csv')[0], index_col=0)
    A2_frst_df = A2_frst_df.set_index('huc8')
    A2_frst_avg = A2_frst_df.mean() * 100

    A2_ints_df = pd.read_csv(glob('../data/LandCover/FORESCE_NLCDCDL_CSVs/A2/*INTS*.csv')[0], index_col=0)
    A2_ints_df = A2_ints_df.set_index('huc8')
    A2_ints_avg = A2_ints_df.mean() * 100

    A2_agri_df = pd.read_csv(glob('../data/LandCover/FORESCE_NLCDCDL_CSVs/A2/*AGRI*.csv')[0], index_col=0)
    A2_agri_df = A2_agri_df.set_index('huc8')
    A2_agri_avg = A2_agri_df.mean() * 100

    #### B1 ####
    B1_frst_df = pd.read_csv(glob('../data/LandCover/FORESCE_NLCDCDL_CSVs/B1/*FRST*.csv')[0], index_col=0)
    B1_frst_df = B1_frst_df.set_index('huc8')
    B1_frst_avg = B1_frst_df.mean() * 100

    B1_ints_df = pd.read_csv(glob('../data/LandCover/FORESCE_NLCDCDL_CSVs/B1/*INTS*.csv')[0], index_col=0)
    B1_ints_df = B1_ints_df.set_index('huc8')
    B1_ints_avg = B1_ints_df.mean() * 100

    B1_agri_df = pd.read_csv(glob('../data/LandCover/FORESCE_NLCDCDL_CSVs/B1/*AGRI*.csv')[0], index_col=0)
    B1_agri_df = B1_agri_df.set_index('huc8')
    B1_agri_avg = B1_agri_df.mean() * 100

    #### B2 ####
    B2_frst_df = pd.read_csv(glob('../data/LandCover/FORESCE_NLCDCDL_CSVs/B2/*FRST*.csv')[0], index_col=0)
    B2_frst_df = B2_frst_df.set_index('huc8')
    B2_frst_avg = B2_frst_df.mean() * 100

    B2_ints_df = pd.read_csv(glob('../data/LandCover/FORESCE_NLCDCDL_CSVs/B2/*INTS*.csv')[0], index_col=0)
    B2_ints_df = B2_ints_df.set_index('huc8')
    B2_ints_avg = B2_ints_df.mean() * 100

    B2_agri_df = pd.read_csv(glob('../data/LandCover/FORESCE_NLCDCDL_CSVs/B2/*AGRI*.csv')[0], index_col=0)
    B2_agri_df = B2_agri_df.set_index('huc8')
    B2_agri_avg = B2_agri_df.mean() * 100

    #### NLCDCDL OBSERVED ####
    NLCD_frst_df = pd.read_csv(glob('../data/LandCover/FORESCE_NLCDCDL_CSVs/NLCD*/*FRST*.csv')[0], index_col=0)
    NLCD_frst_df = NLCD_frst_df.set_index('huc8')
    NLCD_frst_avg = NLCD_frst_df.mean() * 100

    NLCD_ints_df = pd.read_csv(glob('../data/LandCover/FORESCE_NLCDCDL_CSVs/NLCD*/*INTS*.csv')[0], index_col=0)
    NLCD_ints_df = NLCD_ints_df.set_index('huc8')
    NLCD_ints_avg = NLCD_ints_df.mean() * 100

    NLCD_agri_df = pd.read_csv(glob('../data/LandCover/FORESCE_NLCDCDL_CSVs/NLCD*/*AGRI*.csv')[0], index_col=0)
    NLCD_agri_df = NLCD_agri_df.set_index('huc8')
    NLCD_agri_avg = NLCD_agri_df.mean() * 100

    return(hist_frst_avg, hist_ints_avg, hist_agri_avg,\
            NLCD_frst_avg, NLCD_ints_avg, NLCD_agri_avg,\
            A1B_frst_avg, A1B_ints_avg, A1B_agri_avg,\
            A2_frst_avg, A2_ints_avg, A2_agri_avg,\
            B1_frst_avg, B1_ints_avg, B1_agri_avg,\
            B2_frst_avg, B2_ints_avg, B2_agri_avg)

# Plot
# Study area HEX colors:
# green: #028239
# pink: #ff7e90
# tan: #fde48f

def save_foresce_png(hist_frst_avg, hist_ints_avg, hist_agri_avg,\
            NLCD_frst_avg, NLCD_ints_avg, NLCD_agri_avg,\
            A1B_frst_avg, A1B_ints_avg, A1B_agri_avg,\
            A2_frst_avg, A2_ints_avg, A2_agri_avg,\
            B1_frst_avg, B1_ints_avg, B1_agri_avg,\
            B2_frst_avg, B2_ints_avg, B2_agri_avg):
    
    fig, ax1 = plt.subplots(figsize=(16, 8))

    # ax1.plot(hist_frst_avg, linewidth = 3, alpha=0.75, color='#006723', label='Historical', linestyle=(5, (10, 3)))
    # ax1.plot(hist_ints_avg, linewidth = 3, alpha=0.75, color='#b62b53', linestyle=(5, (10, 3)))
    # ax1.plot(hist_agri_avg, linewidth = 3, alpha=0.75, color='#bca447', linestyle=(5, (10, 3)))

    ax1.plot(NLCD_frst_avg, linewidth = 3, alpha=1, color='#006723', label='Forest-dominated', linestyle='solid')
    ax1.plot(NLCD_ints_avg, linewidth = 3, alpha=1, color='#b62b53', label='Intensive', linestyle='solid')
    ax1.plot(NLCD_agri_avg, linewidth = 3, alpha=1, color='#bca447', label='Agriculture', linestyle='solid')

    ax1.plot(B1_frst_avg, linewidth = 3, alpha=0.75, color='#006723', linestyle = 'dashdot', label='B1' )
    ax1.plot(B1_ints_avg, linewidth = 3, alpha=0.75, color='#b62b53', linestyle = 'dashdot')
    ax1.plot(B1_agri_avg, linewidth = 3, alpha=0.75, color='#bca447', linestyle = 'dashdot')

    ax1.plot(B2_frst_avg, linewidth = 3, alpha=0.75, color='#006723', linestyle = (0, (3, 1,1,1,1,1)), label='B2' )
    ax1.plot(B2_ints_avg, linewidth = 3, alpha=0.75, color='#b62b53', linestyle = (0, (3, 1,1,1,1,1)))
    ax1.plot(B2_agri_avg, linewidth = 3, alpha=0.75, color='#bca447', linestyle = (0, (3, 1,1,1,1,1)))

    ax1.plot(A1B_frst_avg, linewidth = 3, alpha=0.75, color='#006723', linestyle = (0,(5,5)), label='A1B' )
    ax1.plot(A1B_ints_avg, linewidth = 3, alpha=0.75, color='#b62b53', linestyle = (0,(5,5)))
    ax1.plot(A1B_agri_avg, linewidth = 3, alpha=0.75, color='#bca447', linestyle = (0,(5,5)))

    ax1.plot(A2_frst_avg, linewidth = 3, alpha=0.75, color='#006723', linestyle = (0,(5,1)), label='A2')
    ax1.plot(A2_ints_avg, linewidth = 3, alpha=0.75, color='#b62b53', linestyle = (0,(5,1)))
    ax1.plot(A2_agri_avg, linewidth = 3, alpha=0.75, color='#bca447', linestyle = (0,(5,1)))

    ax1.set_title('FORE-SCE Land Cover Projections')
    ax1.set_xlabel('Year', size=24)
    ax1.set_ylabel('Percent Land Cover', size=24)
    ax1.set_ylim(0,75)

    # Make a plot with major ticks that are multiples of 10 and minor ticks that
    # are multiples of 5.  Label major ticks with '.0f' formatting but don't label
    # minor ticks.  The string is used directly, the `StrMethodFormatter` is
    # created automatically.
    ax1.yaxis.set_major_locator(MultipleLocator(10))
    ax1.yaxis.set_major_formatter('{x:.0f}')
    # For the minor ticks, use no labels; default NullFormatter.
    ax1.yaxis.set_minor_locator(MultipleLocator(5))
    ax1.yaxis.grid(True, which='major', linestyle = (0, (1, 5)))

    ax1.xaxis.set_major_locator(IndexLocator(base=10, offset=-1))
    # ax1.xaxis.set_major_formatter('{x:.0f}')
    # For the minor ticks, use no labels; default NullFormatter.
    ax1.xaxis.set_minor_locator(MultipleLocator(1))
    ax1.xaxis.grid(True, which='major', linestyle = (0, (1, 5)))

    # reordering the labels
    handles, labels = plt.gca().get_legend_handles_labels()

    # hist = mlines.Line2D([], [], color='black', linewidth=3, ls=(5, (10, 3)), label='Historical')
    a1b = mlines.Line2D([], [], color='black', linewidth=3, ls=(0,(5,5)), label='A1B')
    a2 = mlines.Line2D([], [], color='black', linewidth=3, ls=(0,(5,1)), label='A2')
    b1 = mlines.Line2D([], [], color='black', linewidth=3, ls='dashdot', label='B1')
    b2 = mlines.Line2D([], [], color='black', linewidth=3, ls=(0, (3, 1,1,1,1,1)), label='B2')
    # handles[0] = hist
    handles[5] = a1b
    handles[6] = a2
    handles[3] = b1
    handles[4] = b2

    # specify order
    order = [0, 1, 2, 3, 4, 5, 6]
    # pass handle & labels lists along with order as below
    ax1.legend([handles[i] for i in order], [labels[i] for i in order])

    ax1.text(2101, 70.7, 'B2')
    ax1.text(2101, 66.5, 'B1')
    ax1.text(2100, 53.8, 'A1B')
    ax1.text(2101, 48, 'A2')


    fig.savefig('../imgs/Paper2/var_projections/FORE-SCE.png', dpi=300,\
        facecolor='w', edgecolor='w', transparent=False, pad_inches=0)


def import_gcm_data(gcm=str, scn=str, var=str, WRR=False):

    if gcm == 'GRIDMET':
        gcm_spri = pd.read_csv(glob(f'../data/ClimateData/macav2livneh_GRIDMET_CSVs/{gcm}/*Sp*{var}*.csv')[0], index_col=0)
        gcm_summ = pd.read_csv(glob(f'../data/ClimateData/macav2livneh_GRIDMET_CSVs/{gcm}/*Su*{var}*.csv')[0], index_col=0)
        gcm_fall = pd.read_csv(glob(f'../data/ClimateData/macav2livneh_GRIDMET_CSVs/{gcm}/*Fa*{var}*.csv')[0], index_col=0)
        gcm_wint = pd.read_csv(glob(f'../data/ClimateData/macav2livneh_GRIDMET_CSVs/{gcm}/*Wi*{var}*.csv')[0], index_col=0)

    else:
        gcm_spri = pd.read_csv(glob(f'../data/ClimateData/macav2livneh_GRIDMET_CSVs/{gcm}/*SPRING*{var}*{scn}*.csv')[0], index_col=0)
        gcm_summ = pd.read_csv(glob(f'../data/ClimateData/macav2livneh_GRIDMET_CSVs/{gcm}/*SUMMER*{var}*{scn}*.csv')[0], index_col=0)
        gcm_fall = pd.read_csv(glob(f'../data/ClimateData/macav2livneh_GRIDMET_CSVs/{gcm}/*FALL*{var}*{scn}*.csv')[0], index_col=0)
        gcm_wint = pd.read_csv(glob(f'../data/ClimateData/macav2livneh_GRIDMET_CSVs/{gcm}/*WINTER*{var}*{scn}*.csv')[0], index_col=0)

    gcm_spri = gcm_spri.set_index('huc8')
    gcm_spri.columns = gcm_spri.columns + '00'

    gcm_summ = gcm_summ.set_index('huc8')
    gcm_summ.columns = gcm_summ.columns + '25'

    gcm_fall = gcm_fall.set_index('huc8')
    gcm_fall.columns = gcm_fall.columns + '50'

    gcm_wint = gcm_wint.set_index('huc8')
    gcm_wint.columns = gcm_wint.columns + '75'

    gcm_all = gcm_spri.merge(gcm_summ, on='huc8')
    gcm_all = gcm_all.merge(gcm_fall, on='huc8')
    gcm_all = gcm_all.merge(gcm_wint, on='huc8')

    gcm_avg = gcm_all.sort_index(axis = 1).mean()

    return(gcm_avg)


# rcp 45
GRIDMET_mxtemp_45 = import_gcm_data('GRIDMET', 'RCP45', 'maxTemp')

GFDL_mxtemp_45 = import_gcm_data('GFDL', 'RCP45', 'MAX_TEMP')
HadGEM2_mxtemp_45 = import_gcm_data('HadGEM2', 'RCP45', 'MAX_TEMP')
IPSL_mxtemp_45 = import_gcm_data('IPSL', 'RCP45', 'MAX_TEMP')
MIROC5_mxtemp_45 = import_gcm_data('MIROC5', 'RCP45', 'MAX_TEMP')
NorESM1_mxtemp_45 = import_gcm_data('NorESM1', 'RCP45', 'MAX_TEMP')

GFDL_mxtemp_85 = import_gcm_data('GFDL', 'RCP85', 'MAX_TEMP')
HadGEM2_mxtemp_85 = import_gcm_data('HadGEM2', 'RCP85', 'MAX_TEMP')
IPSL_mxtemp_85 = import_gcm_data('IPSL', 'RCP85', 'MAX_TEMP')
MIROC5_mxtemp_85 = import_gcm_data('MIROC5', 'RCP85', 'MAX_TEMP')
NorESM1_mxtemp_85 = import_gcm_data('NorESM1', 'RCP85', 'MAX_TEMP')


fig, ax1 = plt.subplots(figsize=(24, 8))

ax1.plot(GRIDMET_mxtemp_45[53:], color='black')

# ax1.plot(GFDL_mxtemp_45, alpha=0.75, color='dodgerblue', linestyle='dotted')
# ax1.plot(HadGEM2_mxtemp_45, alpha=0.75, color='dodgerblue', linestyle='dashed')
# ax1.plot(IPSL_mxtemp_45, alpha=0.75, color='dodgerblue', linestyle='dashdot')
# ax1.plot(NorESM1_mxtemp_45, alpha=0.75, color='dodgerblue', linestyle=(0,(5,10)))

# ax1.plot(GFDL_mxtemp_85, alpha=0.75, color='pink', linestyle='dotted')
# ax1.plot(HadGEM2_mxtemp_85, alpha=0.75, color='pink', linestyle='dashed')
# ax1.plot(IPSL_mxtemp_85, alpha=0.75, color='pink', linestyle='dashdot')
# ax1.plot(NorESM1_mxtemp_85, alpha=0.75, color='pink', linestyle=(0,(5,10)))

ax1.plot((GFDL_mxtemp_45 + HadGEM2_mxtemp_45 + IPSL_mxtemp_45 + NorESM1_mxtemp_45)/4,
        color='dodgerblue', alpha=0.5)

ax1.plot((GFDL_mxtemp_85 + HadGEM2_mxtemp_85 + IPSL_mxtemp_85 + NorESM1_mxtemp_85)/4,
        color='red', alpha=0.5)


ax1.yaxis.set_major_locator(MultipleLocator(5))
ax1.yaxis.set_major_formatter('{x:.0f}')
# For the minor ticks, use no labels; default NullFormatter.
ax1.yaxis.set_minor_locator(MultipleLocator(1))
ax1.yaxis.grid(True, which='major', linestyle = (0, (1, 5)))

ax1.xaxis.set_major_locator(IndexLocator(base=40, offset=-8))
# ax1.xaxis.set_major_formatter('{x:.0f}')
# For the minor ticks, use no labels; default NullFormatter.
ax1.xaxis.set_minor_locator(IndexLocator(base=4, offset=-24))
ax1.xaxis.grid(True, which='major', linestyle = (0, (1, 5)))







hist_frst_avg, hist_ints_avg, hist_agri_avg,\
    NLCD_frst_avg, NLCD_ints_avg, NLCD_agri_avg,\
    A1B_frst_avg, A1B_ints_avg, A1B_agri_avg,\
    A2_frst_avg, A2_ints_avg, A2_agri_avg,\
    B1_frst_avg, B1_ints_avg, B1_agri_avg,\
    B2_frst_avg, B2_ints_avg, B2_agri_avg = import_foresce_data()

for df in [hist_frst_avg, hist_ints_avg, hist_agri_avg,\
        NLCD_frst_avg, NLCD_ints_avg, NLCD_agri_avg,\
        A1B_frst_avg, A1B_ints_avg, A1B_agri_avg,\
        A2_frst_avg, A2_ints_avg, A2_agri_avg,\
        B1_frst_avg, B1_ints_avg, B1_agri_avg,\
        B2_frst_avg, B2_ints_avg, B2_agri_avg]:
    df.index = df.index.astype('int')

save_foresce_png(hist_frst_avg, hist_ints_avg, hist_agri_avg,\
            NLCD_frst_avg, NLCD_ints_avg, NLCD_agri_avg,\
            A1B_frst_avg, A1B_ints_avg, A1B_agri_avg,\
            A2_frst_avg, A2_ints_avg, A2_agri_avg,\
            B1_frst_avg, B1_ints_avg, B1_agri_avg,\
            B2_frst_avg, B2_ints_avg, B2_agri_avg)

# adjust y axis to go from 0 to 100 and have more ticks
# different dash marks for different projections
# solid line for observation

# repeat for WRRs

# assess Mann-Kendall across full study area avgs and WRRs (can do by HUC 4 or 8 later if needed)
import pymannkendall as mk


df_lst = [NLCD_frst_avg, NLCD_ints_avg, NLCD_agri_avg,\
            A1B_frst_avg, A1B_ints_avg, A1B_agri_avg,\
            A2_frst_avg, A2_ints_avg, A2_agri_avg,\
            B1_frst_avg, B1_ints_avg, B1_agri_avg,\
            B2_frst_avg, B2_ints_avg, B2_agri_avg]
df_names = ['NLCD_frst_avg', 'NLCD_ints_avg', 'NLCD_agri_avg',\
            'A1B_frst_avg', 'A1B_ints_avg', 'A1B_agri_avg',\
            'A2_frst_avg', 'A2_ints_avg', 'A2_agri_avg',\
            'B1_frst_avg', 'B1_ints_avg', 'B1_agri_avg',\
            'B2_frst_avg', 'B2_ints_avg', 'B2_agri_avg']
for i in range(len(df_lst)):
    print(df_names[i])
    df = df_lst[i]
    res_all = mk.original_test(df)
    print(f'\tALL: {res_all.trend}, {res_all.p}, {res_all.slope}')

    if 'NLCD' in df_names[i]:
        continue
    res_40 = mk.original_test(df.loc[df.index <= 2040])
    res_70 = mk.original_test(df.loc[(df.index > 2040) & (df.index <= 2070)])
    res_99 = mk.original_test(df.loc[(df.index > 2070)])
    print(f'\t2040: {res_40.trend}, {res_40.p}, {res_40.slope}')
    print(f'\t2070: {res_70.trend}, {res_70.p}, {res_70.slope}')
    print(f'\t2099: {res_99.trend}, {res_99.p}, {res_99.slope}\n')


# fig.savefig(outpath, bbox_inches='tight', facecolor='white')


# gfdl_mxtemp_lst = glob(f'../data/ClimateData/macav2livneh_GRIDMET_CSVs/GFDL/*MAX_TEMP*RCP45*.csv')

# gfdl_mxtemp_spri = pd.read_csv(gfdl_mxtemp_lst[0], index_col=0)
# gfdl_mxtemp_spri = gfdl_mxtemp_spri.set_index('huc8')
# gfdl_mxtemp_spri.columns = gfdl_mxtemp_spri.columns + '00'

# gfdl_mxtemp_summ = pd.read_csv(gfdl_mxtemp_lst[1], index_col=0)
# gfdl_mxtemp_summ = gfdl_mxtemp_summ.set_index('huc8')
# gfdl_mxtemp_summ.columns = gfdl_mxtemp_summ.columns + '25'

# gfdl_mxtemp_fall = pd.read_csv(gfdl_mxtemp_lst[2], index_col=0)
# gfdl_mxtemp_fall = gfdl_mxtemp_fall.set_index('huc8')
# gfdl_mxtemp_fall.columns = gfdl_mxtemp_fall.columns + '50'

# gfdl_mxtemp_wint = pd.read_csv(gfdl_mxtemp_lst[3], index_col=0)
# gfdl_mxtemp_wint = gfdl_mxtemp_wint.set_index('huc8')
# gfdl_mxtemp_wint.columns = gfdl_mxtemp_wint.columns + '75'

# gfdl_mxtemp_all = gfdl_mxtemp_spri.merge(gfdl_mxtemp_summ, on='huc8')
# gfdl_mxtemp_all = gfdl_mxtemp_all.merge(gfdl_mxtemp_fall, on='huc8')
# gfdl_mxtemp_all = gfdl_mxtemp_all.merge(gfdl_mxtemp_wint, on='huc8')

# gfdl_mxtemp_avg = gfdl_mxtemp_all.sort_index(axis = 1).mean()