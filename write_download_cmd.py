

import os 

from cli import parse_write_download_cmd


def mk_climate_maca_wget_sh(maca_path, out_dir):
    '''
        Make a sh file using wget to download all MACA downscaled climate data.
    '''

    maca_name = os.path.basename(maca_path).split('_')[0]

    f = open(maca_path)
    f_lines = f.readlines()
    f.close()

    command = "#!/bin/bash \n"
    for line in f_lines:
        line = line.strip()
        if line == '':
            continue

        out_fl = line.split('/')[-1].split('?')[0]

        subfldr_lst = out_fl.split('_')

        subfldr_1 = out_dir + '/' + subfldr_lst[2] + '_' + subfldr_lst[0]
        print(subfldr_1)
        if not os.path.exists(subfldr_1):
            os.mkdir(subfldr_1)
        subfldr_2 = subfldr_1 + '/' + subfldr_lst[4].upper()
        print(subfldr_2)
        if not os.path.exists(subfldr_2):
            os.mkdir(subfldr_2)
        
        out_fl_path = subfldr_2 + '/' + out_fl

        command += 'wget -nc -c -nd "{0}" -O {1} \n'.format(line, out_fl_path)

    output = open('./{}_wget.sh'.format(maca_name), 'x')
    output.write(command)
    output.close()

    print('./{}_wget.sh saved'.format(maca_name))

    return()


def main():

    args = parse_write_download_cmd()

    in_fl = args.in_file
    out_dir = args.out_dir

    mk_climate_maca_wget_sh(in_fl, out_dir)

    return()

if __name__ == '__main__':
    main()