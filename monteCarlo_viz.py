import os
import numpy as np 
import pandas as pd 

import math

import seaborn as sns
from matplotlib import cm
from matplotlib.colors import ListedColormap
import matplotlib.pyplot as plt

from glob import glob

plt.rcParams["font.family"] = "Times New Roman"
plt.rcParams["font.size"] = "16"

def clean_df(df, huc=False, add_rand=False):

    # df['HUC_SEASON'] = df[huc].astype('str') + '_' + df['SEASON']
    df.loc[df.SEASON == "Spring", "YR_SZN"] = df.YEAR * 100 + 0
    df.loc[df.SEASON == "Summer", "YR_SZN"] = df.YEAR * 100 + 25
    df.loc[df.SEASON == "Fall", "YR_SZN"] = df.YEAR * 100 + 50
    df.loc[df.SEASON == "Winter", "YR_SZN"] = df.YEAR * 100 + 75
    df = df.sort_values(by=['YR_SZN'])
    if add_rand:
        df['PRED_WATER'] = df['PRED_LOG_WATER'] #np.exp(df['PRED_LOG_WATER']+np.random.normal(0,1))
    else:
        df['PRED_WATER'] = df['PRED_LOG_WATER'] #np.exp(df['PRED_LOG_WATER'])
    if huc:
        df = df[(df[huc]==3020201)&(df['YEAR']>2005)]

    return(df)

def get_ci(df):

    CI_df = df[['YEAR','SEASON','PRED_LOG_WATER','YR_SZN']]
    CI_df['MEAN'] = df.iloc[:,3:].drop(['YR_SZN'], axis=1).mean(axis=1)
    CI_df['STDEV'] = df.iloc[:,3:].drop(['YR_SZN'], axis=1).std(axis=1)
    n = len(df.iloc[:,3:].drop(['YR_SZN'], axis=1).columns)
    CI_df['LOWER_95_CI'] = CI_df['MEAN'] - ((1.96 * CI_df['STDEV']) / math.sqrt(n))
    CI_df['UPPER_95_CI'] = CI_df['MEAN'] + ((1.96 * CI_df['STDEV']) / math.sqrt(n))
    CI_df

    return(CI_df)


dswe = pd.read_csv('../data/all_data_0018.csv', index_col=0)
dswe['YR_SZN'] = dswe['YEAR'] * 100
dswe.loc[dswe.SEASON == "Spring", "YR_SZN"] += 0
dswe.loc[dswe.SEASON == "Summer", "YR_SZN"] += 25
dswe.loc[dswe.SEASON == "Fall", "YR_SZN"] += 50
dswe.loc[dswe.SEASON == "Winter", "YR_SZN"] += 75
dswe = dswe.sort_values(by=['YR_SZN'])
dswe['OBS_WATER'] = dswe['PR_WATER']
obs_temp_df = dswe[(dswe['HUC08']==3020201)&(dswe['YEAR']>2005)]
obs_temp_df = obs_temp_df.set_index('YR_SZN')

df = pd.read_csv('../data/FutureData/pred_03020201_2006_2099_A1B_FORESCE_RCP85_GFDLESM2M.csv')#, index_col=0)
df = clean_df(df)#, 'huc8')
df = df.set_index('YR_SZN')


df2 = pd.read_csv('../data/FutureData/pred_03020201_2006_2099_B2_FORESCE_RCP45_GFDLESM2M.csv')#, index_col=0)
df2 = clean_df(df2)#,'huc8')
df2 = df2.set_index('YR_SZN')


# plt.plot()
# plt.plot(df[['PRED_WATER']])
# plt.plot(df2[['PRED_WATER']])

a1b_rcp85_pred_lst = glob('../data/FutureData/*original/*RCP85*A1B.csv')
for i in range(len(a1b_rcp85_pred_lst)):
    df_run = pd.read_csv(a1b_rcp85_pred_lst[i], usecols=['YEAR','SEASON','PRED_LOG_WATER', 'HUC08'])
    df_run = clean_df(df_run, huc=False, add_rand=False)

    if i == 0:
        full_MC_df = df_run
    else:
        full_MC_df = full_MC_df.merge(df_run[['YR_SZN','PRED_WATER']], on='YR_SZN')
    
    del(df_run)

full_MC_info1 = get_ci(full_MC_df)


a1b_rcp85_pred_lst = glob('../data/FutureData/*03020201*A1B*RCP85*.csv')
for i in range(len(a1b_rcp85_pred_lst)):
    df_run = pd.read_csv(a1b_rcp85_pred_lst[i], usecols=['YEAR','SEASON','PRED_LOG_WATER'])
    df_run = clean_df(df_run, False, False)

    if i == 0:
        full_MC_df = df_run
    else:
        full_MC_df = full_MC_df.merge(df_run[['YR_SZN','PRED_WATER']], on='YR_SZN')
    
    del(df_run)

full_MC_info1 = get_ci(full_MC_df)


b2_rcp45_pred_lst = glob('../data/FutureData/*03020201*B2*RCP45*.csv')
for i in range(len(b2_rcp45_pred_lst)):
    df_run = pd.read_csv(b2_rcp45_pred_lst[i], usecols=['YEAR','SEASON','PRED_LOG_WATER'])
    df_run = clean_df(df_run, False, False)

    if i == 0:
        full_MC_df = df_run
    else:
        full_MC_df = full_MC_df.merge(df_run[['YR_SZN','PRED_WATER']], on='YR_SZN')
    
    del(df_run)

full_MC_info2 = get_ci(full_MC_df)



plt.plot()

lst = [a1b_rcp85_pred_lst, b2_rcp45_pred_lst]

for j in range(len(lst)):
    for i in range(len(lst[j])):
        # a1b_df = pd.read_csv(a1b_rcp85_pred_lst[i])
        df = pd.read_csv(lst[j][i])
        df = clean_df(df)

        # print(df[['PR_NAT','PR_AG','PR_INT']].head())

        # # a1b_df['PRED_WATER'] = np.exp(a1b_df['PRED_LOG_WATER'])
        # df['PRED_WATER'] = np.exp(df['PRED_LOG_WATER'])
        
        # df.loc[df.SEASON == "SPRING", "YR_SZN"] = df.YEAR * 100 + 0
        # df.loc[df.SEASON == "SUMMER", "YR_SZN"] = df.YEAR * 100 + 25
        # df.loc[df.SEASON == "FALL", "YR_SZN"] = df.YEAR * 100 + 50
        # df.loc[df.SEASON == "WINTER", "YR_SZN"] = df.YEAR * 100 + 75
        # df = df.sort_values(by=['YR_SZN'])
        # df = df.set_index('YR_SZN')

        # # if (j[i]==j[0]) & (i==0):
        # #     ax = df[['PRED_WATER']].plot(figsize = (16,10)) #.groupby('YEAR').mean()
        # # else:
        # #     df[['PRED_WATER']].plot(ax=ax)
    
        if j == 0:
            col = 'firebrick'
        if j == 1:
            col = 'dodgerblue'

        plt.plot(df[['PRED_WATER']], color=col, alpha=0.3)
    
        # if i==100:
        #     break
plt.plot(np.array(obs_temp_df['OBS_WATER']), color='black',linewidth=2)


fig, ax1 = plt.subplots(figsize=(12, 8))
# res = mk.seasonal_test(huc_data_water, period=4)

# ax1.plot(np.asarray(df['PRED_WATER']),color='firebrick',linewidth=2, alpha=0.5)
# ax1.plot(np.asarray(df2['PRED_WATER']),color='deepskyblue',linewidth=2, alpha=0.5)
x = [i for i in range(375)]

ax1.plot(np.asarray(full_MC_info1['MEAN']-0.02),color='firebrick',linewidth=2, label='SRES A1B - RCP 8.5')
ax1.fill_between(x, full_MC_info1['LOWER_95_CI']-0.02, full_MC_info1['UPPER_95_CI']-0.02, color='firebrick', alpha=.3, label='95% CI')

ax1.plot(np.asarray(full_MC_info2['MEAN']-0.02),color='deepskyblue',linewidth=2, label='SRES B2 - RCP 4.5')
ax1.fill_between(x, full_MC_info2['LOWER_95_CI']-0.02, full_MC_info2['UPPER_95_CI']-0.02, color='deepskyblue', alpha=.3, label='95% CI')


ax1.plot(np.array(obs_temp_df['OBS_WATER']), color='black',linewidth=2, label='Observed - (DSWE)')

# ax1.legend()

# reordering the labels
handles, labels = plt.gca().get_legend_handles_labels()
  
# specify order
order = [0, 3, 1, 4, 2]
  
# pass handle & labels lists along with order as below
ax1.legend([handles[i] for i in order], [labels[i] for i in order])
# plt.show()

ax1.set_xticks([0,16,36,56,76,96,116,136,156,176,196,216,236,256,276,296,316,336,356,375 ], minor=False)
ax1.set_xticklabels(['','2010','','2020','','2030','','2040','','2050','','2060','','2070','','2080','','2090','','2099'], size = 16)

ax1.xaxis.grid(True, which='major', linestyle = (0, (5, 10)))
ax1.set_ylabel('Percent Surface Water Area',size=20)
ax1.set_xlabel('Year', size=20)
ax1.set_title('HUC {0}'.format('03020201'), size=20)

plt.savefig('../imgs/Scenarios/A1B_RCP85_B2_RCP45_03020201.png', dpi=300,\
    facecolor='w', edgecolor='w', transparent=False, pad_inches=0)


######################################################################

b1_rcp45_pred_lst = glob('../data/FutureData/*original/*RCP45*B1.csv')
b2_rcp85_pred_lst = glob('../data/FutureData/*original/*RCP85*B2.csv')
a2_rcp45_pred_lst = glob('../data/FutureData/*original/*RCP85*A2.csv')

# plt.plot()
fig, ax1 = plt.subplots(figsize=(14, 10))

col_lst = ['#a1d99b', '#74c476','#41ab5d','#238b45','#005a32']
for i in range(len(b1_rcp45_pred_lst)):
    
    df_run = pd.read_csv(b1_rcp45_pred_lst[i], usecols=['HUC08','YEAR','SEASON','PRED_LOG_WATER'])
    df_run = clean_df(df_run, 'HUC08', False)
    plt.plot(df_run.YR_SZN, df_run.PRED_WATER, linewidth=4, color=col_lst[i], alpha=0.5, label = os.path.basename(b1_rcp45_pred_lst[i][:-4]))

col_lst = ['#9ecae1','#6baed6','#4292c6','#2171b5','#084594']
for i in range(len(b2_rcp85_pred_lst)):
    
    df_run = pd.read_csv(b2_rcp85_pred_lst[i], usecols=['HUC08','YEAR','SEASON','PRED_LOG_WATER'])
    df_run = clean_df(df_run, 'HUC08', False)
    plt.plot(df_run.YR_SZN, df_run.PRED_WATER, linewidth=4, color=col_lst[i], alpha=0.5, label = os.path.basename(b2_rcp85_pred_lst[i][:-4]))

col_lst = ['#c994c7','#df65b0','#e7298a','#ce1256','#91003f']
for i in range(len(a2_rcp45_pred_lst)):
    
    df_run = pd.read_csv(a2_rcp45_pred_lst[i], usecols=['HUC08','YEAR','SEASON','PRED_LOG_WATER'])
    df_run = clean_df(df_run, 'HUC08', False)
    plt.plot(df_run.YR_SZN, df_run.PRED_WATER, linewidth=4, color=col_lst[i], alpha=0.5, label = os.path.basename(a2_rcp45_pred_lst[i][:-4]))

# ax1.plot(np.array(obs_temp_df['OBS_WATER']), color='black',linewidth=2, label='Observed - (DSWE)')

# reordering the labels
# handles, labels = plt.gca().get_legend_handles_labels()
  
# # specify order
# order = [0, 3, 1, 4, 2]
  
# # pass handle & labels lists along with order as below
# ax1.legend([handles[i] for i in order], [labels[i] for i in order])
# plt.show()

ax1.legend(loc=[1.01,0.15])
# ax1.set_xticks([0,16,36,56,76,96,116,136,156,176,196,216,236,256,276,296,316,336,356,375 ], minor=False)
ax1.set_xticks([200000,201000,201500,202000,202500,203000,203500,204000,204500,205000,205500,206000,206500,207000,207500,208000,208500,209000,209500,209900 ], minor=False)
ax1.set_xticklabels(['','2010','','2020','','2030','','2040','','2050','','2060','','2070','','2080','','2090','','2099'], size = 16)

ax1.xaxis.grid(True, which='major', linestyle = (0, (5, 10)))
ax1.set_ylabel('Percent Surface Water Area',size=20)
ax1.set_xlabel('Year', size=20)
ax1.set_title('HUC {0}'.format('03020201'), size=20)

plt.savefig('../imgs/Paper2/all_original_03020201.png', dpi=300,\
    facecolor='w', edgecolor='w', transparent=False, pad_inches=0)



######################################################################

b1_rcp45_pred_lst = glob('../data/FutureData/GCM_FORESCE_CSVs/GFDL/*RCP45*B1/*.csv')
b2_rcp85_pred_lst = glob('../data/FutureData/GCM_FORESCE_CSVs/GFDL/*RCP85*B2/*.csv')
a2_rcp85_pred_lst = glob('../data/FutureData/GCM_FORESCE_CSVs/GFDL/*RCP85*A2/*.csv')

for i in range(len(b1_rcp45_pred_lst)):
    df_run = pd.read_csv(b1_rcp45_pred_lst[i], usecols=['YEAR','SEASON','PRED_LOG_WATER', 'HUC08'])
    df_run = clean_df(df_run, huc='HUC08', add_rand=False)

    if i == 0:
        full_MC_df = df_run
    else:
        full_MC_df = full_MC_df.merge(df_run[['YR_SZN','PRED_WATER']], on='YR_SZN')
    
    del(df_run)

full_MC_info1 = get_ci(full_MC_df)
del(full_MC_df)

for i in range(len(b2_rcp85_pred_lst)):
    df_run = pd.read_csv(b2_rcp85_pred_lst[i], usecols=['YEAR','SEASON','PRED_LOG_WATER', 'HUC08'])
    df_run = clean_df(df_run, huc='HUC08', add_rand=False)

    if i == 0:
        full_MC_df = df_run
    else:
        full_MC_df = full_MC_df.merge(df_run[['YR_SZN','PRED_WATER']], on='YR_SZN')
    
    del(df_run)

full_MC_info2 = get_ci(full_MC_df)
del(full_MC_df)

for i in range(len(a2_rcp85_pred_lst)):

    df_run = pd.read_csv(a2_rcp85_pred_lst[i], usecols=['YEAR','SEASON','PRED_LOG_WATER', 'HUC08'])
    df_run = clean_df(df_run, huc='HUC08', add_rand=False)

    if i == 0:
        full_MC_df = df_run
    else:
        full_MC_df = full_MC_df.merge(df_run[['YR_SZN','PRED_WATER']], on='YR_SZN')
    
    del(df_run)


full_MC_info3 = get_ci(full_MC_df)
del(full_MC_df)


full_MC_info1 = pd.read_csv('../data/FutureData/MC_CI_RCP45_B1_HUC.csv', index_col=0)
full_MC_info2 = pd.read_csv('../data/FutureData/MC_CI_RCP85_B2_HUC.csv', index_col=0)

fig, ax1 = plt.subplots(figsize=(12, 8))
# res = mk.seasonal_test(huc_data_water, period=4)

# ax1.plot(np.asarray(df['PRED_WATER']),color='firebrick',linewidth=2, alpha=0.5)
# ax1.plot(np.asarray(df2['PRED_WATER']),color='deepskyblue',linewidth=2, alpha=0.5)
x = [i for i in range(375)]

ax1.plot(np.asarray(full_MC_info1['MEAN']),color='#005a32',linewidth=2, label='RCP 4.5 - B1')
ax1.fill_between(x, full_MC_info1['LOWER_95_CI'], full_MC_info1['UPPER_95_CI'], color='#005a32', alpha=.3, label='95% CI')

ax1.plot(np.asarray(full_MC_info2['MEAN']),color='#084594',linewidth=2, label='RCP 8.5 - B2')
ax1.fill_between(x, full_MC_info2['LOWER_95_CI'], full_MC_info2['UPPER_95_CI'], color='#084594', alpha=.3, label='95% CI')

ax1.plot(np.asarray(full_MC_info3['MEAN']),color='#91003f',linewidth=2, label='RCP 8.5 - A2')
ax1.fill_between(x, full_MC_info3['LOWER_95_CI'], full_MC_info3['UPPER_95_CI'], color='#91003f', alpha=.3, label='95% CI')


ax1.plot(np.array(obs_temp_df['OBS_WATER']), color='black',linewidth=2, label='Observed - (DSWE)')

# ax1.legend()

# reordering the labels
handles, labels = plt.gca().get_legend_handles_labels()
  
# specify order
order = [0, 1, 2, 3, 4, 5, 6]
  
# pass handle & labels lists along with order as below
ax1.legend([handles[i] for i in order], [labels[i] for i in order])
# plt.show()

ax1.set_xticks([0,16,36,56,76,96,116,136,156,176,196,216,236,256,276,296,316,336,356,375 ], minor=False)
ax1.set_xticklabels(['','2010','','2020','','2030','','2040','','2050','','2060','','2070','','2080','','2090','','2099'], size = 16)

ax1.xaxis.grid(True, which='major', linestyle = (0, (5, 10)))
ax1.set_ylabel('Percent Surface Water Area',size=20)
ax1.set_xlabel('Year', size=20)
ax1.set_title('HUC {0}'.format('03020201'), size=20)

plt.savefig('../imgs/Paper2/HUC_MONTE_CARLO.png', dpi=300,\
    facecolor='w', edgecolor='w', transparent=False, pad_inches=0)











# plt.plot()
fig, ax1 = plt.subplots(figsize=(14, 10))

col_lst = ['#a1d99b', '#74c476','#41ab5d','#238b45','#005a32']
for i in range(len(b1_rcp45_pred_lst)):
    
    df_run = pd.read_csv(b1_rcp45_pred_lst[i], usecols=['HUC08','YEAR','SEASON','PRED_LOG_WATER'])
    df_run = clean_df(df_run, 'HUC08', False)
    plt.plot(df_run.YR_SZN, df_run.PRED_WATER, linewidth=4, color=col_lst[i], alpha=0.5, label = os.path.basename(b1_rcp45_pred_lst[i][:-4]))

col_lst = ['#9ecae1','#6baed6','#4292c6','#2171b5','#084594']
for i in range(len(b2_rcp85_pred_lst)):
    
    df_run = pd.read_csv(b2_rcp85_pred_lst[i], usecols=['HUC08','YEAR','SEASON','PRED_LOG_WATER'])
    df_run = clean_df(df_run, 'HUC08', False)
    plt.plot(df_run.YR_SZN, df_run.PRED_WATER, linewidth=4, color=col_lst[i], alpha=0.5, label = os.path.basename(b2_rcp85_pred_lst[i][:-4]))

col_lst = ['#c994c7','#df65b0','#e7298a','#ce1256','#91003f']
for i in range(len(a2_rcp85_pred_lst)):
    
    df_run = pd.read_csv(a2_rcp85_pred_lst[i], usecols=['HUC08','YEAR','SEASON','PRED_LOG_WATER'])
    df_run = clean_df(df_run, 'HUC08', False)
    plt.plot(df_run.YR_SZN, df_run.PRED_WATER, linewidth=4, color=col_lst[i], alpha=0.5, label = os.path.basename(a2_rcp45_pred_lst[i][:-4]))

# ax1.plot(np.array(obs_temp_df['OBS_WATER']), color='black',linewidth=2, label='Observed - (DSWE)')

# reordering the labels
# handles, labels = plt.gca().get_legend_handles_labels()
  
# # specify order
# order = [0, 3, 1, 4, 2]
  
# # pass handle & labels lists along with order as below
# ax1.legend([handles[i] for i in order], [labels[i] for i in order])
# plt.show()

ax1.legend(loc=[1.01,0.15])
# ax1.set_xticks([0,16,36,56,76,96,116,136,156,176,196,216,236,256,276,296,316,336,356,375 ], minor=False)
ax1.set_xticks([200000,201000,201500,202000,202500,203000,203500,204000,204500,205000,205500,206000,206500,207000,207500,208000,208500,209000,209500,209900 ], minor=False)
ax1.set_xticklabels(['','2010','','2020','','2030','','2040','','2050','','2060','','2070','','2080','','2090','','2099'], size = 16)

ax1.xaxis.grid(True, which='major', linestyle = (0, (5, 10)))
ax1.set_ylabel('Percent Surface Water Area',size=20)
ax1.set_xlabel('Year', size=20)
ax1.set_title('HUC {0}'.format('03020201'), size=20)

plt.savefig('../imgs/Paper2/all_original_03020201.png', dpi=300,\
    facecolor='w', edgecolor='w', transparent=False, pad_inches=0)