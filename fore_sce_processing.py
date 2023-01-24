import xarray as xr
from glob import glob
import geopandas as gpd
import rasterio as rio
import rioxarray as rxr
import numpy as np
import pandas as pd
# from osgeo import gdal
from shapely.geometry import mapping
import matplotlib.pyplot as plt


from rasterstats import zonal_stats

import os


##### Clip to study area #####

tif_paths = glob('../data/LandCover/FORE-SCE/CONUS_Landcover_*/*.tif')
huc_outline = gpd.read_file('../data/Shapefiles/outline_paper2/outline_clean_paper2.shp')
src = rxr.open_rasterio(tif_paths[0])
huc_outline_reproj = huc_outline.to_crs(crs=src.rio.crs.to_dict()['init'])
del(src)

for i in range(len(tif_paths)):
    in_path = tif_paths[i]

    sres_path = in_path.split('\\')[1]
    sres = sres_path.split('_')[2]
    yr = in_path.split('_')[4][1:5]

    out_path = os.path.join('../data/LandCover/FORE-SCE',sres_path,f'clipped/{yr}_{sres}_clipped.tif')
    if not os.path.exists(out_path):
        if not os.path.exists(os.path.dirname(out_path)):
            os.makedirs(os.path.dirname(out_path))

        src = rxr.open_rasterio(tif_paths[i])
       
        clipped = src.rio.clip(huc_outline_reproj.geometry.apply(mapping),
                                        # This is needed if your GDF is in a diff CRS than the raster data
                                        huc_outline_reproj.crs, drop=True, invert=False)

        clipped.rio.to_raster(out_path)
        del(src)
        del(clipped)
        print('{0}/{1} clips saved. {2}_{3}_clipped.tif'.format(i+1, len(tif_paths), yr, sres))

    else:
        print(f'{yr}_{sres}_clipped.tif exists.')
del(huc_outline, huc_outline_reproj)


##### Reclassify clips #####

print('\nStarting reclassification.')
tif_paths = glob('../data/LandCover/FORE-SCE/CONUS_Landcover_*/clipped/*.tif')

for i in range(len(tif_paths)):
    in_path = tif_paths[i]
    yr, sres, tif = os.path.basename(in_path).split('_')
    out_path = os.path.join(os.path.dirname(os.path.dirname(in_path)),f'reclass/{yr}_{sres}_reclass.tif')
    if not os.path.exists(out_path):
        if not os.path.exists(os.path.dirname(out_path)):
            os.makedirs(os.path.dirname(out_path))

        with rio.open(in_path) as src:
        # src = rio.open(in_path)

            src_data = src.read(1)
            np_shape = (3, src_data.shape[0], src_data.shape[1])
            suit_raster = np.zeros(np_shape, dtype=rio.int8)
            # Reclassify to natural land cover
            suit_raster[0] = np.select([src_data == 0, ((7 < src_data) & (src_data <= 12)) | ((13 < src_data) & (src_data <= 16)),\
                                                        (src_data < 7) | (src_data == 13) | (src_data > 16)], [0,1,0])
            # Reclassify to Agricultural land cover
            suit_raster[1] = np.select([src_data == 0, (src_data == 13),\
                                                        (src_data < 13) | (src_data > 13)], [0,1,0])
            # Reclassify to Intensive
            suit_raster[2] = np.select([src_data == 0, ((1 < src_data) & (src_data <= 7)),\
                                                        (src_data < 1) | (src_data > 7)], [0,1,0])

            profile = src.profile
            profile.update(count=3)
            # print(src.profile)
            # print(profile)

        with rio.open(out_path, "w", **profile) as dst:
            dst.write(suit_raster)

            dst.set_band_description(1, "NATURAL")
            dst.set_band_description(2, "AGRICULTURAL")
            dst.set_band_description(3, "INTENSIVE")

        del(src_data, suit_raster, src, dst)

        print('{0}/{1} reclass saved. {2}_{3}_reclass.tif'.format(i+1, len(tif_paths), yr, sres))

    else:
        print(f'{yr}_{sres}_reclass.tif exists.')


##### Zonal Stats on reclass #####

print('\nStarting zonal stats.')
tif_paths = glob('../data/LandCover/FORE-SCE/CONUS_Landcover_*/reclass/*.tif')

src = rio.open(tif_paths[0])
shp = gpd.read_file('../data/Shapefiles/HUC08/HUC08_paper2/HUC08_paper2.shp')
huc8_reproj = shp.to_crs(crs=src.crs.to_dict()['init'])
# point_reproj = pt_shp.to_crs(dataset.crs.to_dict())
del(shp)
huc8 = huc8_reproj[['huc8','geometry']]
del(huc8_reproj)

for i in range(len(tif_paths)):
    base_path = os.path.dirname(tif_paths[i])[:-7]
    yr = tif_paths[i].split('\\')[3].split('_')[0]
    sres = tif_paths[i].split('\\')[3].split('_')[1]
    
    out_path = os.path.join(base_path, 'zonal_lclu_csv', f'{yr}_{sres}_lclu.csv')
    if not os.path.exists(out_path):
        if not os.path.exists(os.path.dirname(out_path)):
            os.makedirs(os.path.dirname(out_path))

        src = rio.open(tif_paths[i])

        huc8_df = pd.DataFrame(huc8['huc8'])
        huc8_df['YEAR'] = int(yr)
        huc8_df['PR_AG'] = 0
        huc8_df['PR_NAT'] = 0
        huc8_df['PR_INT'] = 0

        affine = src.transform
        for j in [1,2,3]:
            array = src.read(j)
            
            zs = zonal_stats(huc8, array, affine=affine, stats=['sum', 'count'], nodata=-1, all_touched = False)

            zs_df = pd.DataFrame(zs)
            zs_df = zs_df.fillna(0)

            if j == 1:
                var = 'PR_NAT'
            elif j == 2:
                var = 'PR_AG'
            elif j == 3:
                var = 'PR_INT'

            huc8_df[var] += zs_df['sum'] / (zs_df['count'])
        
            del(array)

        src.close()

        huc8_df.to_csv(out_path, index=False)
        print('{0}/{1} zonal stats saved. {2}_{3}_lclu.csv'.format(i+1, len(tif_paths), yr, sres))

        del(huc8_df)    
    
    else:
        print(f'{yr}_{sres}_lclu.csv exists.')