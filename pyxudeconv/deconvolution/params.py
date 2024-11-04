from argparse import ArgumentParser
import numpy as np
import json


def get_param():
    '''Get parameters for deconvolution'''
    parser = ArgumentParser(
        description='Deconvolution parameters',
        prog="params.py",
    )
    parser.add_argument(
        '--param_file',
        type=str,
        default=None,
        help=
        'Load a json param file INSTEAD of taking other command lines arguments',
    )
    parser.add_argument(
        '--datapath',
        type=str,
        default='./',
        help='Path to the data to deconvolve',
    )
    parser.add_argument(
        '--psfpath',
        type=str,
        default='/data/tampham/Zeiss/Airyscan_PSF.czi',
        help='Path to the point-spread function',
    )

    parser.add_argument(
        '--methods',
        type=str,
        nargs='+',
        default=('GARL', 'RLTV', 'RL', 'Tikhonov'),
        help='(Tuple of str) Method used for reconstructions',
    )

    parser.add_argument(
        '--fres',
        type=str,
        default='./results',
        help='Folder to save results',
    )

    parser.add_argument(
        '--psf_sz',
        type=int,
        nargs='+',
        default=(None, None, 64, 64),
        help=
        'Tuple (xc,yc,lx,ly) with (xc,yc) the PSF center and (lx,ly) for window size, middle of the whole FOV if (xc,yc) is (None,None) or negative'
    )

    parser.add_argument(
        '--nviews',
        type=int,
        default=32,
        help='Number of views used to deconvolve (Airyscan or equivalent)',
    )

    parser.add_argument(
        '--gpu',
        type=int,
        default=0,
        help='int for GPU (-1 for CPU)',
    )

    parser.add_argument(
        '--bufferwidth',
        type=int,
        nargs='+',
        default=(1, 3, 3),
        help=
        'To deconvolve properly at the border, optimize over a larger volume (Z,Y,X)',
    )
    parser.add_argument(
        '--phantom',
        type=str,
        default=None,
        help='Phantom filename, e.g., tube',
    )

    parser.add_argument(
        '--coi',
        nargs='+',
        type=int,
        default=0,
        help='Channel(s) for data',
    )
    parser.add_argument(
        '--coi_psf',
        nargs='+',
        type=int,
        default=None,
        help='Channel(s) for PSF. By default, same than coi.',
    )
    parser.add_argument(
        '--bg',
        type=float,
        default=None,
        help=
        'Minimum value of the deconvolved volume. If None, estimated from measurements.',
    )

    parser.add_argument(
        '--normalize_meas',
        type=int,
        default=1,
        choices=[0, 1],
        help='(Boolean) Normalize measurements.',
    )

    parser.add_argument(
        '--roi',
        type=int,
        nargs='+',
        default=(0, 0, None, None),  #298, 298),
        help='Tuple to deconvolve only a region of interest (x0,y0,w,h)',
    )

    parser.add_argument(
        '--Nepoch',
        type=int,
        default=50,
        help='Number of epoch (iterations) used for each optimization',
    )
    parser.add_argument(
        '--pxsz',
        type=float,
        nargs='+',
        default=(35.7, 35.7, 100.),
        help='Pixel sizes for Tiff metadata (X,Y,Z)',
    )

    parser.add_argument(
        '--pxunit',
        type=str,
        default='nm',
    )

    parser.add_argument(
        '--saveMeas',
        type=int,
        default=1,
        choices=[0, 1],
        help='(Boolean) Save the measurements',
    )

    parser.add_argument(
        '--saveIter',
        type=int,
        nargs='+',
        default=(50, 50),
        help=
        'Solutions are saved every saveIter iterations. Length should match the length of --methods arguments'
    )

    parser.add_argument(
        '--disp',
        type=int,
        default=10,
        help='Display every '
        'disp'
        ' iterations',
    )

    parser.add_argument(
        '--config_GARL',
        type=str,
        default='airyscan_params',
        help=
        'Name of the configuration file located in folder methods/configs/GARL for the method GARL',
    )

    par = parser.parse_args()
    if par.param_file is not None and par.param_file != '':
        print(f'Loading param file {par.param_file}')
        with open(par.param_file, 'r', encoding="utf-8") as f:
            par.__dict__ = json.load(f)
    if par.coi_psf is None:
        par.coi_psf = par.coi
    par.psf_sz = np.array(par.psf_sz)
    par.phantom = None if par.phantom == 'None' else par.phantom
    par.saveMeas = bool(par.saveMeas)
    par.normalize_meas = bool(par.normalize_meas)

    return par


if __name__ == '__main__':
    pars = get_param()
