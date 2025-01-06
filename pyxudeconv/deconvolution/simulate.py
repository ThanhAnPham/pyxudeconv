import numpy as np
import os
import importlib
import platform
import tifffile
import json
import pyxudeconv
import pyxudeconv.deconvolution.forward.convolution as forw
from argparse import ArgumentParser
import pyxu.operator as pxo
from pyxu.operator import FFTConvolve
from pyxudeconv.deconvolution.reader.get_reader import get_reader
from pyxudeconv.deconvolution.utils import convert2save


def simulate():
    parser = ArgumentParser(
        description='Convolution simulation parameters',
        prog="simulate",
    )
    parser.add_argument(
        '--param_file',
        type=str,
        default=None,
        help=
        'Load a json param file INSTEAD of taking other command lines arguments',
    )
    parser.add_argument('--gpu',
                        type=int,
                        default=0,
                        help='GPU id (-1 for CPU)')
    parser.add_argument(
        '--fres',
        type=str,
        default='../data/simulated',
        help='Folder to save results',
    )
    parser.add_argument(
        '--phantom',
        type=str,
        default='/data/tampham/calibration_3/sample_calib.ome.tif',
        help='Phantom filename, e.g., tube',
    )
    parser.add_argument(
        '--psfpath',
        type=str,
        default='/data/tampham/Zeiss/Airyscan_PSF.czi',
        help='Path to the point-spread function',
    )
    parser.add_argument(
        '--coi',
        nargs='+',
        type=int,
        default=2,
        help='Channel(s) for PSF from Airy',
    )
    parser.add_argument(
        '--bg',
        type=float,
        default=1e-4,
        help='Background value in the noiseless measurements.',
    )
    parser.add_argument(
        '--roi',
        type=int,
        nargs='+',
        default=(0, 0, None, None),  #298, 298),
        help='Tuple to deconvolve only a region of interest (x0,y0,w,h)',
    )

    parser.add_argument(
        '--psf_sz',
        type=int,
        nargs='+',
        default=(None, None, 64, 64),
        help=
        'Tuple (xc,yc,lx,ly) with (xc,yc) the PSF center and (lx,ly) for window size, middle of the whole FOV if (xc,yc) is (None,None)'
    )
    parser.add_argument(
        '--nviews',
        type=int,
        default=32,
        help='Number of views used to deconvolve (Airyscan or equivalent)',
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
        '--noise',
        type=float,
        nargs='+',
        default=(0.001, 2.5),
    )
    par = parser.parse_args()
    par.psf_sz = np.array(par.psf_sz)
    if par.param_file is not None and par.param_file != '':
        print(f'Loading param file {par.param_file}')
        with open(par.param_file, 'r', encoding="utf-8") as f:
            par.__dict__ = json.load(f)

    if par.gpu >= 0:
        ordi = platform.system()
        if ordi == 'Darwin':  # mac
            print('On mac')
            device_name = "mps"
        else:
            import cupy
            device_name = 'cuda:' + str(
                par.gpu) if cupy.cuda.is_available() else 'cpu'
    else:
        device_name = 'cpu'

    if 'cuda' in device_name:
        import cupy as xp
        on_gpu = True
        xp.cuda.Device(par.gpu).use()
    else:
        import numpy as xp
        on_gpu = False
    # 0. Define the read function for data and psf by checking the file extension

    read_psf = get_reader(par.psfpath)
    read_phantom = get_reader(par.phantom)
    roi = par.roi
    nviews = par.nviews
    hoi = tuple(range(nviews))
    phantom = xp.array(read_phantom(par.phantom))

    psf = xp.array(
        read_psf(
            par.psfpath,
            Hoi=hoi,
            Coi=par.coi,
            roi=par.psf_sz,
        ).astype("float32"))

    if psf.ndim > 3 + 1 * (isinstance(par.coi, tuple)):
        #Deconvolution problem has more than one view (Airyscan-like)
        psf = psf.squeeze()
        nviews = psf.shape[
            0]  #data might have less views than the expected number set by parameters

    # Normalize
    psf /= psf.sum(
        axis=(-3, -2, -1),
        keepdims=True)  # Each view and each channel integrates to 1. (2,3,4)
    if psf.shape[0] == 1:
        psf = psf.squeeze(0)

    print('Phantom shape', phantom.shape, 'min/max values', phantom.min(),
          phantom.max())
    print('PSF shape ', psf.shape, 'and min value ', psf.min())
    recon_shape = phantom.shape
    #Physical model
    if nviews > 1:
        forw_conv = pxo.stack([
            FFTConvolve(
                dim_shape=recon_shape,
                kernel=psf_view,
                center=tuple(np.array(psf_view.shape) // 2),
            ) for psf_view in psf
        ])
    else:
        forw_conv = FFTConvolve(
            dim_shape=recon_shape,
            kernel=psf,
            center=tuple(np.array(psf.shape) // 2),
        )
    phantom /= phantom.max()
    phantom += par.noise[0]
    g = forw_conv(phantom)
    g += par.bg
    rng = xp.random.default_rng(0)
    if par.noise[1] > 0:
        sig = xp.sqrt(g / xp.array(par.noise[1]))
        g = xp.maximum(g + sig * rng.standard_normal(size=g.shape), 0.)
    g2save = convert2save(g)
    psf2save = convert2save(psf if psf.ndim == 5 else xp.expand_dims(psf, 1))
    phantom_name = par.phantom[par.phantom.rfind('/') +
                               1:par.phantom.find('.', par.phantom.rfind('/'))]
    fid = f'{phantom_name}_nv_{nviews:d}_coi'

    if isinstance(par.coi, int):
        fid = f'{fid}_{par.coi:d}'
    else:
        for ccoi in par.coi:
            fid = f'{fid}_{ccoi:d}'
    fid = f'{fid}_n_{par.noise[0]}_{par.noise[1]}'
    fid = f'{fid}_bg_{par.bg}'
    if not os.path.exists(par.fres):
        os.makedirs(par.fres)
    tifffile.imwrite(
        f'{par.fres}/g_{fid}.ome.tif',
        g2save,
        imagej=True,
        resolution=(1 / par.pxsz[0], 1 / par.pxsz[1]),
        metadata={
            'axes': 'TZCYX',
            'spacing': par.pxsz[2],
            'unit': par.pxunit
        },
    )
    tifffile.imwrite(
        f'{par.fres}/psf_{fid}.ome.tif',
        psf2save,
        imagej=True,
        resolution=(1 / par.pxsz[0], 1 / par.pxsz[1]),
        metadata={
            'axes': 'TZCYX',
            'spacing': par.pxsz[2],
            'unit': par.pxunit
        },
    )
    phantom2save = convert2save(phantom)
    tifffile.imwrite(
        f'{par.fres}/phantom_{fid}.ome.tif',
        phantom2save,
        imagej=True,
        resolution=(1 / par.pxsz[0], 1 / par.pxsz[1]),
        metadata={
            'axes': 'TZCYX',
            'spacing': par.pxsz[2],
            'unit': par.pxunit
        },
    )
    with open(f'{par.fres}/params.json', 'w', encoding='utf-8') as f:
        #par.psf_sz
        par.psf_sz = par.psf_sz if isinstance(par.psf_sz, list) else list(
            par.psf_sz)
        json.dump(par.__dict__, f, indent=2)


if __name__ == '__main__':
    simulate()
