import os
import numpy as np 
import pandas as pd 

import math

import seaborn as sns
from matplotlib import cm
from matplotlib.colors import ListedColormap
from matplotlib.ticker import (MultipleLocator, AutoMinorLocator, IndexLocator)
import matplotlib.pyplot as plt
import pandas as pd

from glob import glob

plt.rcParams["font.family"] = "Times New Roman"
plt.rcParams["font.size"] = "16"

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

scn_lst = ['Historical', 'A1B', 'A2', 'B1', 'B2']
lclu_lst = ['FRST', 'INTS', 'AGRI']

scn_dict = {'Historical': (5, (10, 3)), 
                   'A1B': (0, (5, 5)), 
                    'A2': (0, (5, 1)), 
                    'B1': 'dashdot', 
                    'B2': (0, (3, 1, 1, 1, 1, 1))}
lclu_dict = {'FRST': ['green', 'Forest-dominated'], 
             'INTS': ['red', 'Intensive'], 
             'AGRI': ['brown', 'Agriculture']}

fig, ax1 = plt.subplots(figsize=(16, 8))

for scn in scn_lst:
    for lclu in lclu_lst:
        fl = glob(f'../data/LandCover/FORESCE_NLCDCDL_CSVs/{scn}/*{lclu}*.csv')[0]
        df = pd.read_csv(fl, index_col=0)
        df = df.set_index('huc8')
        df_avg = df.mean() * 100

        if scn == 'Observed':
            ax1.plot(df, color=lclu_dict[lclu][0], linestyle=scn_dict[scn], 
                        label=lclu_dict[lclu][1], linewidth=2)
        elif lclu == 'FRST':
            ax1.plot(df, color=lclu_dict[lclu][0], linestyle=scn_dict[scn], 
                        label=scn, linewidth=2)
        else:
            ax1.plot(df, color=lclu_dict[lclu][0], linestyle=scn_dict[scn], 
                        linewidth=2)


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

# Plot
# Study area HEX colors:
# green: #028239
# pink: #ff7e90
# tan: #fde48f

fig, ax1 = plt.subplots(figsize=(16, 8))

ax1.plot(hist_frst_avg, linewidth = 3, alpha=0.75, color='#006723', label='Historical', linestyle=(5, (10, 3)))
ax1.plot(hist_ints_avg, linewidth = 3, alpha=0.75, color='#b62b53', linestyle=(5, (10, 3)))
ax1.plot(hist_agri_avg, linewidth = 3, alpha=0.75, color='#bca447', linestyle=(5, (10, 3)))

ax1.plot(A1B_frst_avg, linewidth = 3, alpha=0.75, color='#006723', linestyle = (0,(5,5)), label='A1B' )
ax1.plot(A1B_ints_avg, linewidth = 3, alpha=0.75, color='#b62b53', linestyle = (0,(5,5)))
ax1.plot(A1B_agri_avg, linewidth = 3, alpha=0.75, color='#bca447', linestyle = (0,(5,5)))

ax1.plot(A2_frst_avg, linewidth = 3, alpha=0.75, color='#006723', linestyle = (0,(5,1)), label='A2')
ax1.plot(A2_ints_avg, linewidth = 3, alpha=0.75, color='#b62b53', linestyle = (0,(5,1)))
ax1.plot(A2_agri_avg, linewidth = 3, alpha=0.75, color='#bca447', linestyle = (0,(5,1)))

ax1.plot(B1_frst_avg, linewidth = 3, alpha=0.75, color='#006723', linestyle = 'dashdot', label='B1' )
ax1.plot(B1_ints_avg, linewidth = 3, alpha=0.75, color='#b62b53', linestyle = 'dashdot')
ax1.plot(B1_agri_avg, linewidth = 3, alpha=0.75, color='#bca447', linestyle = 'dashdot')

ax1.plot(B2_frst_avg, linewidth = 3, alpha=0.75, color='#006723', linestyle = (0, (3, 1,1,1,1,1)), label='B2' )
ax1.plot(B2_ints_avg, linewidth = 3, alpha=0.75, color='#b62b53', linestyle = (0, (3, 1,1,1,1,1)))
ax1.plot(B2_agri_avg, linewidth = 3, alpha=0.75, color='#bca447', linestyle = (0, (3, 1,1,1,1,1)))

ax1.plot(NLCD_frst_avg, linewidth = 3, alpha=1, color='#006723', label='Forest-dominated', linestyle='solid')
ax1.plot(NLCD_ints_avg, linewidth = 3, alpha=1, color='#b62b53', label='Intensive', linestyle='solid')
ax1.plot(NLCD_agri_avg, linewidth = 3, alpha=1, color='#bca447', label='Agriculture', linestyle='solid')


ax1.set_title('FORE-SCE Land Cover Projections')
ax1.set_xlabel('Year')
ax1.set_ylabel('Percent Land Cover')
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

# ax1.set_xticks([1991, 2101])

ax1.xaxis.set_major_locator(IndexLocator(base=10, offset=-2))
# ax1.xaxis.set_major_formatter('{x:.0f}')
# For the minor ticks, use no labels; default NullFormatter.
ax1.xaxis.set_minor_locator(MultipleLocator(1))

ax1.xaxis.grid(True, which='major', linestyle = (0, (1, 5)))

# reordering the labels
handles, labels = plt.gca().get_legend_handles_labels()
# specify order
order = [5, 6, 7, 0, 1, 2, 3, 4]
# pass handle & labels lists along with order as below
ax1.legend([handles[i] for i in order], [labels[i] for i in order])

# adjust y axis to go from 0 to 100 and have more ticks
# different dash marks for different projections
# solid line for observation

# repeat for WRRs

# assess Mann-Kendall across full study area avgs and WRRs (can do by HUC 4 or 8 later if needed)

# fig.savefig(outpath, bbox_inches='tight', facecolor='white')