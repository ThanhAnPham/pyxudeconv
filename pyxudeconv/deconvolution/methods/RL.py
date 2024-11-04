# Accelerated Richardson-Lucy

from pyxu.operator import KLDivergence

# in future release, RL will be incorporated in Pyxu
import methods.solver as solver

from .ABC import HyperParametersDeconvolutionOptimizer

__all__ = [
    "RL",
]


class RL(HyperParametersDeconvolutionOptimizer):
    r"""
    Hyper parameters optimizer for Accelerated Richardson-Lucy
    """

    def get_hyperparams(self):
        return dict()

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
