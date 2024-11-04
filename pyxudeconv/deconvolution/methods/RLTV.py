# Total-Variation Accelerated Richardson-Lucy
#from libraries import *
#import pyxu
import numpy as np
import pyxu.abc as pxa
from pyxu.operator import KLDivergence, Gradient
import pyxu.info.ptype as pxt
import pyxu.util as pxu
# in future release, RRL will be incorporated in Pyxu
from methods.solver import RRL

from .ABC import HyperParametersDeconvolutionOptimizer

__all__ = [
    "RLTV",
]


class RLTV(HyperParametersDeconvolutionOptimizer):
    r"""
    Hyper parameters optimizer for Total-Variation Accelerated Richardson-Lucy
    """

    def get_hyperparams(self):
        params = dict()
        params['tau'] = np.logspace(-7, 0, 8)  #15)
        return params

    def init_solver(self, param):
        reg_shape = self._trim_buffer.codim_shape  #chooses where the regularization is enforced
        grad = Gradient(dim_shape=reg_shape,
                        mode='symmetric',
                        gpu='cuda' in self._device_name)
        
        l21 = diffL21Norm(dim_shape=grad.codim_shape, l2_axis=(0, ))
        reg = param['tau'] * l21 * grad * self._trim_buffer
        loss = KLDivergence(self._g)
        self._solver = RRL(
            loss,
            self._forw,
            self._g,
            reg,
            verbosity=self._disp,
            stop_rate=1,
            show_progress=False,
            bg=self._bg_est,
        )


class diffL21Norm(pxa.ProxDiffFunc):
    r"""
    Differentiable mixed :math:`\ell_{2}-\ell_{1}` norm, :math:`\Vert\mathbf{x}\Vert_{2, 1} := \sum_{i} \sqrt{\sum_{j} x_{i, j}^{2}+\epsilon}`.
    """

    def __init__(
            self,
            dim_shape: pxt.NDArrayShape,
            l2_axis: pxt.NDArrayAxis = (0, ),
            epsilon: pxt.Real = 1e-6,
    ):
        r"""
        Parameters
        ----------
        l2_axis: NDArrayAxis
            Axis (or axes) along which the :math:`\ell_{2}` norm is applied.
        epsilon: Real
            Nonnegative scalar (>0) to prevent division by 0
        """
        super().__init__(
            dim_shape=dim_shape,
            codim_shape=1,
        )
        assert self.dim_rank >= 2

        l2_axis = pxu.as_canonical_axes(l2_axis, rank=self.dim_rank)
        l1_axis = tuple(ax for ax in range(self.dim_rank) if ax not in l2_axis)

        self.lipschitz = np.inf
        self._l1_axis = l1_axis
        self._l2_axis = l2_axis
        assert epsilon > 0
        self._epsilon = epsilon

    def apply(self, arr: pxt.NDArray) -> pxt.NDArray:
        sh = arr.shape[:-self.dim_rank]

        l2_axis = tuple(len(sh) + ax for ax in self._l2_axis)
        x = (arr**2).sum(axis=l2_axis, keepdims=True)
        xp = pxu.get_array_module(arr)
        xp.sqrt(x + self._epsilon, out=x)

        l1_axis = tuple(len(sh) + ax for ax in self._l1_axis)
        out = x.sum(axis=l1_axis, keepdims=True)

        out = out.squeeze(l1_axis + l2_axis)[..., np.newaxis]
        return out

    def grad(self, arr: pxt.NDArray) -> pxt.NDArray:
        sh = arr.shape[:-self.dim_rank]

        l2_axis = tuple(len(sh) + ax for ax in self._l2_axis)
        x = (arr**2).sum(axis=l2_axis, keepdims=True)
        xp = pxu.get_array_module(arr)
        xp.sqrt(x + self._epsilon, out=x)
        out = arr / x
        return out

    def prox(self, arr: pxt.NDArray, tau: pxt.Real) -> pxt.NDArray:
        sh = arr.shape[:-self.dim_rank]

        l2_axis = tuple(len(sh) + ax for ax in self._l2_axis)
        n = (arr**2).sum(axis=l2_axis, keepdims=True)
        xp = pxu.get_array_module(arr)
        xp.sqrt(n + self._epsilon, out=n)

        out = arr.copy()
        out *= 1 - tau / xp.fmax(n, tau)
        return out
