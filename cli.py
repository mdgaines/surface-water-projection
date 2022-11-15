
import argparse
# from email.policy import default
from xmlrpc.client import Boolean

def parse_write_download_cmd():
    '''
        Parse the expected command-line arguments for write_download_cmd.py
    '''
    parser = argparse.ArgumentParser(description='Writes sh file to download MACA downscaled GCMs.')

    # parser.add_help()

    parser.add_argument('-i', '--in_file', type=str,
                        default='macav2livneh_urls.txt',
                        help='Input path to url textfile.')
    
    parser.add_argument('-o', '--out_dir', type=str,
                        default='../data/ClimateData/',
                        help='Output directory for where nc files will be saved.')

    return(parser.parse_args())


def parse_clip_nc():
    '''
        Parse the command-line arguments for .py for climate_data_processing.py
    '''
    parser = argparse.ArgumentParser(description='Clips climate NC files to study area, can calculate seasonal averages.')

    parser.add_argument('-cl', '--cutline', type=str,
                        default='../data/Shapefiles/outline_paper2/outline_clean_paper2.shp',
                        help='Outline with which to cut the NC files.')

    parser.add_argument('-dt', '--data_set', type=str,
                        default='all',
                        help='General path to climate files\' parent directory.\n\
                        Ex: \'../data/ClimateData/GFDL-ESM2M_macav2livneh\'\n\
                        OR \'all\'')
    
    parser.add_argument('-avg', '--seasonal_avg', type=Boolean,
                        default=True, 
                        help='True/False to calculate seasonal average.')

    parser.add_argument('-shp', '--shapefile', type=str,
                        default='../data/Shapefiles/HUC08/HUC08_paper2/HUC08_paper2.shp',
                        help='Path to the shapefile to use to calculate seasonal averages.')

    return(parser.parse_args())

