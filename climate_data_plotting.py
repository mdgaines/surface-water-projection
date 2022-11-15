
# import xarray as xr
# import rioxarray as riox
from glob import glob
# import geopandas as gpd
import numpy as np
import pandas as pd
import os

import matplotlib.pyplot as plt

#### DATA PATHS
precip_paths = glob('../data/ClimateData/macav2livneh_studyArea_avgs/*PRECIP.csv')



#### GFDL PRECIP
gfdl_85_precip_path = '../data/ClimateData/macav2livneh_studyArea_avgs/GFDL-ESM2M_RCP85_PRECIP.csv'
gfdl_45_precip_path = '../data/ClimateData/macav2livneh_studyArea_avgs/GFDL-ESM2M_RCP45_PRECIP.csv'
gfdl_HISTORICAL_precip_path = '../data/ClimateData/macav2livneh_studyArea_avgs/GFDL-ESM2M_HISTORICAL_PRECIP.csv'

gfdl_85_precip = pd.read_csv(gfdl_85_precip_path, index_col=0)
gfdl_45_precip = pd.read_csv(gfdl_45_precip_path, index_col=0)
gfdl_HISTORICAL_precip = pd.read_csv(gfdl_HISTORICAL_precip_path, index_col=0)


gfdl_85_precip['YEAR'] = gfdl_85_precip.apply(lambda row: row['DATE'][0:4], axis=1)
gfdl_45_precip['YEAR'] = gfdl_45_precip.apply(lambda row: row['DATE'][0:4], axis=1)
gfdl_HISTORICAL_precip['YEAR'] = gfdl_HISTORICAL_precip.apply(lambda row: row['DATE'][0:4], axis=1)

# monthly avg
fig, ax1 = plt.subplots(figsize=(12, 8))

ax1.plot(gfdl_HISTORICAL_precip['DATE'], gfdl_HISTORICAL_precip['PRECIP_AVG'], color='black')
ax1.plot(gfdl_85_precip['DATE'], gfdl_85_precip['PRECIP_AVG'], color='red')
ax1.plot(gfdl_45_precip['DATE'], gfdl_45_precip['PRECIP_AVG'], color='orange')

# yearly avg
fig, ax1 = plt.subplots(figsize=(18, 8))

ax1.plot(gfdl_HISTORICAL_precip.groupby('YEAR').mean(), color='black')
ax1.plot(gfdl_85_precip.groupby('YEAR').mean(), color='red')
ax1.plot(gfdl_45_precip.groupby('YEAR').mean(), color='orange')


#### RCP45 PRECIP
gfdl_45_precip_path = '../data/ClimateData/macav2livneh_studyArea_avgs/GFDL-ESM2M_RCP45_PRECIP.csv'
hadgem2_45_precip_path = '../data/ClimateData/macav2livneh_studyArea_avgs/HadGEM2-ES365_RCP45_PRECIP.csv'
# gfdl_45_precip_path = '../data/ClimateData/macav2livneh_studyArea_avgs/GFDL-ESM2M_RCP45_PRECIP.csv'
# gfdl_45_precip_path = '../data/ClimateData/macav2livneh_studyArea_avgs/GFDL-ESM2M_RCP45_PRECIP.csv'
# gfdl_45_precip_path = '../data/ClimateData/macav2livneh_studyArea_avgs/GFDL-ESM2M_RCP45_PRECIP.csv'

gfdl_45_precip = pd.read_csv(gfdl_45_precip_path, index_col=0)
hadgem2_45_precip = pd.read_csv(hadgem2_45_precip_path, index_col=0)


gfdl_45_precip['YEAR'] = gfdl_45_precip.apply(lambda row: row['DATE'][0:4], axis=1)
hadgem2_45_precip['YEAR'] = hadgem2_45_precip.apply(lambda row: row['DATE'][0:4], axis=1)


fig, ax1 = plt.subplots(figsize=(12, 8))


ax1.plot(gfdl_45_precip.groupby('YEAR').mean())
ax1.plot(hadgem2_45_precip.groupby('YEAR').mean())


#### ALL PRECIP
precip_paths.sort()

fig, ax1 = plt.subplots(figsize=(12, 8))

for fl_path in precip_paths:
    df = pd.read_csv(fl_path, index_col=0)
    df['YEAR'] = df.apply(lambda row: row['DATE'][2:3], axis=1)

    # if 'HISTORICAL' in os.path.basename(fl_path):
    #     ax1.plot(df.groupby('YEAR').mean(), color='black', alpha=0.5)

    if 'RCP45' in os.path.basename(fl_path):
        ax1.plot(df.groupby('YEAR').mean(), color='brown', alpha=0.5)

    # elif 'RCP85' in os.path.basename(fl_path):
    #     ax1.plot(df.groupby('YEAR').mean(), color='red', alpha=0.5)




#### GFDL MAX-TEMP
gfdl_85_MAXTEMP_path = '../data/ClimateData/macav2livneh_studyArea_avgs/GFDL-ESM2M_RCP85_MAX-TEMP.csv'
gfdl_45_MAXTEMP_path = '../data/ClimateData/macav2livneh_studyArea_avgs/GFDL-ESM2M_RCP45_MAX-TEMP.csv'
gfdl_HISTORICAL_MAXTEMP_path = '../data/ClimateData/macav2livneh_studyArea_avgs/GFDL-ESM2M_HISTORICAL_MAX-TEMP.csv'

gfdl_85_MAXTEMP = pd.read_csv(gfdl_85_MAXTEMP_path, index_col=0)
gfdl_45_MAXTEMP = pd.read_csv(gfdl_45_MAXTEMP_path, index_col=0)
gfdl_HISTORICAL_MAXTEMP = pd.read_csv(gfdl_HISTORICAL_MAXTEMP_path, index_col=0)


gfdl_85_MAXTEMP['YEAR'] = gfdl_85_MAXTEMP.apply(lambda row: row['DATE'][0:4], axis=1)
gfdl_45_MAXTEMP['YEAR'] = gfdl_45_MAXTEMP.apply(lambda row: row['DATE'][0:4], axis=1)
gfdl_HISTORICAL_MAXTEMP['YEAR'] = gfdl_HISTORICAL_MAXTEMP.apply(lambda row: row['DATE'][0:4], axis=1)


fig, ax1 = plt.subplots(figsize=(12, 8))

ax1.plot(gfdl_HISTORICAL_MAXTEMP.groupby('YEAR').mean())
ax1.plot(gfdl_85_MAXTEMP.groupby('YEAR').mean())
ax1.plot(gfdl_45_MAXTEMP.groupby('YEAR').mean())