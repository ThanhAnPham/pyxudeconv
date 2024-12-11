# Accelerated Richardson-Lucy

from pyxu.operator import KLDivergence

# in future release, RL will be incorporated in Pyxu
import pyxudeconv.deconvolution.methods.solver as solver

from .ABC import HyperParametersDeconvolutionOptimizer

__all__ = [
    "RL",
]


class RL(HyperParametersDeconvolutionOptimizer):
    r"""
    Hyper parameters optimizer for Accelerated Richardson-Lucy
    """

    def get_hyperparams(self):
        if isinstance(self._param_method, dict):
            #Parameters are directly provided as a dictionary (possible if deconvolve is called from Python)
            if 'acceleration' not in self._param_method.keys():
                self._param_method['acceleration'] = [True]
            return self._param_method
        else:
            new_param = {
                'acceleration': [True],
            }
            return new_param

    def init_solver(self, param):
        lossRL = KLDivergence(self._g)
        self._solver = solver.RL(
            lossRL,
            self._forw,
            self._g,
            verbosity=self._disp,
            show_progress=False,
            stop_rate=1,
            bg=self._bg_est,
        )
        self._solver_param = {'acceleration': param['acceleration']}
