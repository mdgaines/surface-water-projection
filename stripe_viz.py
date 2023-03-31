import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from matplotlib.collections import PatchCollection
from matplotlib.colors import ListedColormap
import pandas as pd
from glob import glob

### Climate Stripes code adapted from https://matplotlib.org/matplotblog/posts/warming-stripes/ ###

FIRST = 1950
LAST = 2022  # inclusive

# Reference period for the center of the color scale

FIRST_REFERENCE = 1979
LAST_REFERENCE = 2008
LIM = 0.7 # degrees

# data from

mxtemp_paths = glob('../data/ClimateData/macav2livneh_studyArea_avgs/*MAX-TEMP.csv')


anomaly = df.loc[FIRST:LAST, 'anomaly'].dropna()
reference = anomaly.loc[FIRST_REFERENCE:LAST_REFERENCE].mean()

# the colors in this colormap come from http://colorbrewer2.org

# the 8 more saturated colors from the 9 blues / 9 reds

cmap = ListedColormap([
    '#08306b', '#08519c', '#2171b5', '#4292c6',
    '#6baed6', '#9ecae1', '#c6dbef', '#deebf7',
    '#fee0d2', '#fcbba1', '#fc9272', '#fb6a4a',
    '#ef3b2c', '#cb181d', '#a50f15', '#67000d',
])

fig = plt.figure(figsize=(10, 1))

ax = fig.add_axes([0, 0, 1, 1])
ax.set_axis_off()


# create a collection with a rectangle for each year

col = PatchCollection([
    Rectangle((y, 0), 1, 1)
    for y in range(FIRST, LAST + 1)
])

# set data, colormap and color limits

col.set_array(anomaly)
col.set_cmap(cmap)
col.set_clim(reference - LIM, reference + LIM)
ax.add_collection(col)


ax.set_ylim(0, 1)
ax.set_xlim(FIRST, LAST + 1)

# fig.savefig('warming-stripes.png')