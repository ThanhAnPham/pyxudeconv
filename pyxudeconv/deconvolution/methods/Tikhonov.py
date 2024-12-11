# Accelerated Richardson-Lucy
#from libraries import *
#import pyxu

import numpy as np
from pyxu.operator import SquaredL2Norm, PositiveOrthant
from pyxu.opt.solver import PGD

from .ABC import HyperParametersDeconvolutionOptimizer

__all__ = [
    "Tikhonov",
]

class Tikhonov(HyperParametersDeconvolutionOptimizer):
    r"""
    Hyper parameters optimizer for Tikhonov
    """

    def get_hyperparams(self):
        if isinstance(self._param_method, dict):
            params = self._param_method
        else:
            params = dict()
            params['tau'] = [1e-5]
        return params

    def init_solver(self, param):
        loss = 0.5 * SquaredL2Norm(
            dim_shape=self._forw.codim_shape).argshift(-self._g) * self._forw
        loss += param['tau'] * SquaredL2Norm(
            dim_shape=self._trim_buffer.codim_shape) * self._trim_buffer
        
        print(f'Set constraint for Tikhonov [{self._bg_est},{np.inf}]')
        
        self._solver = PGD(
            loss,
            g=PositiveOrthant(self._forw.dim_shape).argshift(-self._bg_est),
            verbosity=self._disp,
            stop_rate=5,
            show_progress=False,
        )
        self._solver_param = {'acceleration': True}
