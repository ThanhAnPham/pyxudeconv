import numpy as numpy


def widefield_params():
    #nameCRR = par.WCnet #'3Dkern333/3DWCRR-CNN'#'WCRR-CNN'#'confWCRR-CNN' #'confWCRR-CNN-one-scale' #'Sigma_25_t_5' #'WCRR-CNN' #CRR_CNN #confWCRR-CNN
        #epochoi = par.WCchkpts #40180
    params = dict()
    params['WCRnet'] = [
        '../trained_models/3Dtubes/'
    ]
    params['epochoi'] = [40180]
    params['lmbd'] = [0.1,0.5]  #np.arange(0.6, 2.1, 0.2)
    params['sigWC'] = [0.1,0.5]  #np.arange(0.1, 2.1, 0.2)
    return params