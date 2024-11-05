import numpy as numpy


def airyscan_params():
    params = dict()
    params['WCRnet'] = [
        'pyxudeconv/trained_models/3Dtubes/'
    ]
    params['epochoi'] = [40180]
    params['lmbd'] = [0.5, 1, 2]  #np.arange(0.6, 2.1, 0.2),
    params['sigWC'] = [1.2]  #np.arange(0.1, 2.1, 0.2)
    return params