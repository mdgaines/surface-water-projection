
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
        Parse the command-line arguments for climate_data_processing.py
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


def parse_get_huc_scn_info():
    '''
        Parse the command-line arguments for get_huc_scn_info.py
    '''

    parser = argparse.ArgumentParser(description='Gets mean, 95\% CI, and Mann-Kendall trend and p-value for\
                                     a specified HUC and scenario.')
    
    parser.add_argument('-scn', '--scenario', type=str,
                        default='all',
                        help='RCP-SRES scenarios\
                            \n(options (str): all, \
                            \n\tRCP45_A1B, RCP45_A2, RCP45_B1, RCP45_B2, \
                            \n\tRCP85_A1B, RCP85_A2, RCP85_B1, RCP85_B2)')

    parser.add_argument('-huc', '--huc_lst', type=list,
                        default=[3130001, 3020201, 3100205, 6010201, 8090203],
                        help='List of HUC08s of interest')

    parser.add_argument('-vars', '--variables', type=list,
                        default=['MEAN', 'CI', 'TREND', 'P_VALUE'],
                        help='Variables of interest')

    parser.add_argument('-yr', '--years', type=str,
                        default='all',
                        help='Years for which to get the variable information\
                            \n(options (str): 2040, 2070, 2099, all)')
    
    parser.add_argument('-szn', '--season', type=str,
                        default='spring',
                        help='Season for which to get the variable information\
                            \n(options (str): Spring, Summer, Fall, Winter)')
    
    return(parser.parse_args())
