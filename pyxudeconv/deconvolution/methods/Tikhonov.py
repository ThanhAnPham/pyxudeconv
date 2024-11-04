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
        params = dict()
        params['tau'] = np.logspace(-3, 4, 3)#15)
        return params

    def init_solver(self, param):
        loss = 0.5 * SquaredL2Norm(
            dim_shape=self._forw.codim_shape).argshift(-self._g) * self._forw
        loss += param['tau'] * SquaredL2Norm(
            dim_shape=self._trim_buffer.codim_shape) * self._trim_buffer

        self._solver = PGD(
            loss,
            g=PositiveOrthant(self._forw.dim_shape),
            verbosity=self._disp,
            stop_rate=1,
            show_progress=False,
        )
