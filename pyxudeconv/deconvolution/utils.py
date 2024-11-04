import numpy as np

try:
    import cupy as cp
    HAS_CP = True
except ImportError as e:
    HAS_CP = False

import pyxu.util as pxu


def convert2save(data, has_channel=False):
    #ToDo: Ask in input the original dimension order
    if HAS_CP:
        on_gpu = isinstance(data, cp.ndarray)
    else:
        on_gpu = False
    data = data.get() if on_gpu else data
    if data.ndim == 5:
        #TCZXY -> TZCYX
        data2save = np.transpose(
            data,
            axes=(0, 2, 1, 3, 4),
        ).astype('float32')
    elif data.ndim == 4:
        #CZXY -> TZCYX
        if has_channel:
            data2save = np.expand_dims(
                np.transpose(data, axes=(1, 0, 2, 3)),
                0,
            ).astype('float32')
        else:
            data2save = np.expand_dims(data, 2).astype('float32')
    elif data.ndim == 3:
        #ZXY -> TZCYX
        data2save = np.expand_dims(
            data,
            axis=(0, 2),
        ).astype('float32')
    else:
        data2save = data
    return data2save


def rsnr(xgt, xhat):
    xp = pxu.get_array_module(xgt)

    if xp.linalg.norm(xhat) == 0:
        b = xp.mean(xgt)
        a = 0
    else:
        xhatnorm2 = xp.linalg.norm(xhat)**2
        if (xp.mean(xgt) - xp.mean(xhat) * xp.vdot(xhat, xgt) / xhatnorm2
            ) == 0 and (1 - xp.sum(xhat) * xp.mean(xhat) / (xhatnorm2)) == 0:
            b = 0
        else:
            b = (xp.mean(xgt) - xp.mean(xhat) * xp.vdot(xhat, xgt) /
                 xhatnorm2) / (1 - xp.sum(xhat) * xp.mean(xhat) / (xhatnorm2))
        a = (xp.vdot(xhat, xgt) - b * xp.sum(xhat)) / xhatnorm2

    xhatr = a * xhat + b
    rsnro = 20 * xp.log10(xp.linalg.norm(xgt) / xp.linalg.norm(xgt - xhatr))
    return rsnro
