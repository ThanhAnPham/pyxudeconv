"""
Main script for 3D deconvolution.
"""
import logging
import importlib

import sys
import os
import json
import platform
import numpy as np
import matplotlib.pyplot as plt
from pyxudeconv.deconvolution.reader.get_reader import get_reader
import tifffile
from .utils import convert2save
import pyxudeconv.deconvolution.forward.convolution as forw

import pyxu.opt.stop as pxst

from .params import get_param

sys.path.append('../')


def deconvolve(par=None):
    '''
    Deconvolution methods.
    '''
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)

    if par is None:
        par = get_param()

    if par.gpu >= 0:
        ordi = platform.system()
        if ordi == 'Darwin':  # mac, uncertain compatibilities
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
        import torch
    else:
        import numpy as xp
        on_gpu = False

    print(f'Running on {device_name}')
    seed = 0
    xp.random.seed(seed)

    if hasattr(par, 'save_results'):
        save_results = par.save_results
    else:
        def save_results(vol, fname, pxsz, unit):
            if vol.ndim == 3:
                axes = 'ZYX'
            else:
                axes = 'TZCYX'
            if '.ome.tif' in fname:
                full_fname = fname
            else:
                full_fname = f'{fname}.ome.tif'
            tifffile.imwrite(
                full_fname,
                vol,
                imagej=True,
                resolution=(1 / pxsz[0], 1 / pxsz[1]),
                metadata={
                    'axes': axes,
                    'spacing': pxsz[2],
                    'unit': unit
                },
            )

    # Define the read function for data and psf by checking the file extension
    read_data = get_reader(par.datapath)
    read_psf = get_reader(par.psfpath)
    # Load phantom (if provided), measurements and create physical model from PSF data
    if par.phantom is not None:
        #Metrics
        from pyxudeconv.deconvolution.utils import rsnr as cmp_metrics
        read_phantom = get_reader(par.phantom)
        phantom = xp.array(read_phantom(par.phantom))
        phantom_name = par.phantom[par.phantom.rfind('/') + 1:par.phantom.
                                   find('.', par.phantom.rfind('/'))]
    else:
        print('No ground truth')
        phantom = None
        phantom_name = ''

        def cmp_metrics(_, __):
            return -np.inf

    fid = f'{phantom_name}'

    forw_model, g, trim_buffer, op4metrics, phantom, fid, psf, gnormalizer = forw.getModel(
        par.psfpath, par.datapath, par.psf_sz, par.nviews, par.coi_psf,
        par.roi, par.coi, par.bufferwidth, phantom, fid, xp, read_psf,
        read_data, par.normalize_meas)

    op4save = gnormalizer * trim_buffer

    if par.phantom is not None:
        print('Phantom taken from ' + par.phantom)
        print(f'Min/Max phantom {phantom.min():.4f}/{phantom.max():.4f}')
        print(f'Phantom size {phantom.shape}')
    print('Forward', forw_model)
    print(f'Reconstruction shape {forw_model.dim_shape}')

    pxsz = np.array(par.pxsz)

    if par.fres:
        if not os.path.exists(par.fres):
            os.makedirs(par.fres)
        with open(par.fres + '/params.json', 'w', encoding="utf-8") as f:
            cpar = par
            cpar.psf_sz = cpar.psf_sz if isinstance(
                cpar.psf_sz, list) else list(cpar.psf_sz)
            cpar = vars(cpar)
            json.dump(cpar, f, indent=2,default=int)

    if par.fres:
        fh = logging.FileHandler(f'{par.fres}/run.log', mode='w')
    else:
        fh = logging.NullHandler()
    logger.addHandler(fh)
    x0 = forw_model.adjoint(g)
    if par.bg is None or par.bg < 0:
        bg_est = xp.maximum(
            trim_buffer(x0).min() / (forw_model.codim_shape[0] if len(
                forw_model.codim_shape) > 3 else 1), xp.zeros(1))
        bg_est = bg_est[0]
    else:
        bg_est = xp.maximum(par.bg, xp.zeros(1))
        bg_est = bg_est[0]
    logger.info(f'Estimated background: {bg_est:.3e}')
    x0 = xp.maximum(x0, bg_est)
    x0_metric = cmp_metrics(phantom, op4metrics(x0))

    if par.saveMeas:
        g2save = convert2save(g * gnormalizer)
        save_results(g2save, f'{par.fres}/g_{fid}',pxsz,par.pxunit)
        psf2save = convert2save(psf if psf.ndim ==
                                5 else xp.expand_dims(psf, 1))
        save_results(psf2save, f'{par.fres}/psf_{fid}',pxsz,par.pxunit)

    if par.phantom is not None and par.saveMeas:
        save_results(phantom.get() if on_gpu else phantom, f'{par.fres}/xgt_{fid}',pxsz,par.pxunit)
    if hasattr(par, 'create_fname'):
        create_fname = par.create_fname
    else:
        def create_fname(
            meth,
            paramstr,
            metric=-np.inf,
        ):
            if np.isinf(metric) and metric < 0:
                return f'{par.fres}/{meth}_{fid}_{paramstr}.ome.tif'
            else:
                return f'{par.fres}/{meth}_{fid}_{paramstr}_{metric:.4e}.ome.tif'
    if par.saveMeas:
        save_results(trim_buffer(x0).get() if on_gpu else x0, create_fname('x0', '', x0_metric),pxsz,par.pxunit)
    
    stop_crit = pxst.MaxIter(par.Nepoch)
    ims = []
    imstit = []
    if phantom is not None:
        ims.append(phantom.copy().get() if on_gpu else phantom.copy())
        imstit = ['GT']

    ims.append((gnormalizer * g).copy().get() if on_gpu else gnormalizer *
               g.copy())
    imstit.append('Meas')
    
    if par.disp == 0:
        par.disp = 1e9
    
    dpar = vars(par)
    for meth_iter, method in enumerate(par.methods):
        module_class = importlib.import_module(f'pyxudeconv.deconvolution.methods.{method}')
        classmeth = getattr(module_class, method)
        cconfig = 'config_' + method
        if cconfig in dpar.keys():
            cdpar = dpar[cconfig]
        else:
            cdpar = ''
        cmeth = classmeth(
            forw_model,
            g,
            bg_est,
            trim_buffer,
            device_name,
            par.disp,
            cdpar,
            stop_crit,
            cmp_metrics,
            phantom,
            op4metrics,
            create_fname,
            op4save,
            save_results,
            pxsz=pxsz,
            pxunit=par.pxunit,
        )
        logger.info(f'Doing hyper-parameters optimization for {method}')

        bestrecon, bestparams, bestmetric = cmeth.optimize_hyperparams(
            x0, par.saveIter[np.minimum(meth_iter,
                                        len(par.saveIter) - 1)], logger)
        if par.phantom is None:
            cimstit = f'{method}'
        else:
            logger.info(f'{method} with params ' + ' | '.join([
                f'{c[0]} : {c[1]}'
                for c in zip(bestparams.keys(), bestparams.values())
            ]) + f' : {bestmetric}')
            cimstit = f'{method} {bestmetric}'

        ims.append(bestrecon.get() if on_gpu else bestrecon)
        imstit.append(cimstit)

    if on_gpu:
        phantom = None if phantom is None else phantom.get()
        x0 = x0.get()
        g = g.get()

    if np.size(par.coi) > 1:

        def predisp(x):
            ''' x.ndim > 2 '''
            return x.max(
                axis=(tuple(range(1, 4 - x.ndim, -1))),
                keepdims=False,
            ).transpose(1, 2, 0)
    else:

        def predisp(x):
            return x.max(axis=(tuple(range(0, x.ndim - 2))), keepdims=False)

    ims = [predisp(im) for im in ims]

    if par.fres:
        nrow = 2
        ncol = np.ceil(len(ims) / nrow).astype(int)
        _, axs = plt.subplots(nrows=nrow, ncols=ncol)
        
        for itera, ax in enumerate(axs.reshape(-1)):
            if itera < len(ims):
                plt.axes(ax)
                plt.imshow(ims[itera])
                plt.title(imstit[itera])
                plt.axis('off')

        plt.show()

        plt.savefig(f'{par.fres}/res_{fid}.png', dpi=500, bbox_inches='tight')
    logger.info('Deconvolution finished')
    if on_gpu:
        xp.cuda.Device().synchronize()
        xp._default_memory_pool.free_all_blocks()
        xp._default_pinned_memory_pool.free_all_blocks()
        torch.cuda.empty_cache()

    return ims


if __name__ == '__main__':
    deconvolve()
