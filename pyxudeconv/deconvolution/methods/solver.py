import itertools
import math

import pyxu.abc as pxa
import pyxu.info.ptype as pxt
import pyxu.info.warning as pxw
import pyxu.operator as pxo
import pyxu.util as pxu

__all__ = ["RL", "RRL"]


class RL(pxa.Solver):
    r"""
    Richardson-Lucy solver.

    RL solves minimization problems of the form ToDo

    where:

    * :math:`\mathcal{F}:\mathbb{R}^{M_{1} \times\cdots\times M_{D}}\rightarrow \mathbb{R}` is *convex* and
      *differentiable*, with :math:`\beta`-*Lipschitz continuous* gradient, for some :math:`\beta\in[0,+\infty[`.

    Remarks
    -------
    Parameters (``__init__()``)
    ---------------------------
    Parameters (``fit()``)
    ----------------------
    * **x0** (:py:attr:`~pyxu.info.ptype.NDArray`)
    """

    def __init__(
            self,
            f: pxa.ProxFunc = None,  #for loss computation only
            H: pxt.OpT = None,
            g: pxt.NDArray = None,
            epsi: float = 0.00001,
            bg: float = 0.00001,
            ub: float = float('inf'),
            lb: float = 1e-6,
            **kwargs,
    ):
        kwargs.update(log_var=kwargs.get("log_var", ("x", )), )
        super().__init__(**kwargs)

        if (f is None) or (H is None) or (g is None):
            msg = " ".join([
                "Cannot minimize always-0 functional.",
                " Parameters[f, H, g] must be specified.",
            ])
            raise NotImplementedError(msg)
        else:
            self._f = f
        self._H = H
        self._g = g
        self._epsi = epsi
        self._bg = bg
        self._lb = lb
        self._ub = ub
        xp = pxu.get_array_module(g)
        self._Ht1 = xp.maximum(H.adjoint(xp.ones(H.codim_shape)), self._epsi)
        print(f'Min value in HT1 for RL {self._Ht1.min()}')
        print(f'Set constraint for RL [{0},{self._ub}]')

    def m_init(
        self,
        x0: pxt.NDArray,
        acceleration: bool = False,
    ):
        mst = self._mstate  # shorthand
        mst["x"] = mst["x_prev"] = x0

        if acceleration:
            mst["acc"] = True
            mst["tk"] = 1
        else:
            mst["acc"] = False
            mst["a"] = itertools.repeat(0)

    def m_step(self):
        mst = self._mstate  # shorthand
        xp = pxu.get_array_module(mst["x"])
        # RL multiplicative update

        if mst["acc"]:
            tprev = mst["tk"]
            mst["tk"] = 0.5 * (1 + math.sqrt(1 + 4 * tprev**2))
            a = (tprev - 1) / mst["tk"]
        else:
            a = next(mst["a"])
        # In-place implementation of -----------------
        #   y = (1 + a) * mst["x"] - a * mst["x_prev"]
        p = mst["x"] - mst["x_prev"]
        p *= a
        p += mst["x"]
        p.clip(self._lb, self._ub, out=p)
        p *= self._H.adjoint(
            self._g / xp.maximum(self._H(p) + self._bg, self._epsi)) / self._Ht1
        p.clip(self._lb, self._ub, out=p)
        mst["x_prev"], mst["x"] = mst["x"], p

    def default_stop_crit(self) -> pxa.StoppingCriterion:
        from pyxu.opt.stop import RelError

        stop_crit = RelError(
            eps=1e-4,
            var="x",
            rank=self._f.dim_rank,
            f=None,
            norm=2,
            satisfy_all=True,
        )
        return stop_crit

    def objective_func(self) -> pxt.NDArray:
        func = lambda x: self._f.apply(self._H.apply(x) + self._bg)

        y = func(self._mstate["x"])
        self._mstate["f"] = y
        return y

    def solution(self) -> pxt.NDArray:
        """
        Returns
        -------
        x: NDArray
            (..., M1,...,MD) solution.
        """
        data, _ = self.stats()
        return data.get("x")


class RRL(pxa.Solver):
    r"""
    Richardson-Lucy solver + WCR.

    RL solves minimization problems of the form ToDo

    where:

    * :math:`\mathcal{F}:\mathbb{R}^{M_{1} \times\cdots\times M_{D}}\rightarrow \mathbb{R}` is *convex* and
      *differentiable*, with :math:`\beta`-*Lipschitz continuous* gradient, for some :math:`\beta\in[0,+\infty[`.

    Remarks
    -------
    Parameters (``__init__()``)
    ---------------------------
    Parameters (``fit()``)
    ----------------------
    * **x0** (:py:attr:`~pyxu.info.ptype.NDArray`)
    """

    def __init__(
            self,
            f: pxa.ProxFunc = None,  #for loss computation only
            H: pxt.OpT = None,
            g: pxt.NDArray = None,
            R: pxa.ProxFunc = None,  #needs gradient
            epsi: float = 0.00001,
            bg: float = 0.00001,
            ub: float = float('inf'),
            lb: float = 1e-6,
            **kwargs,
    ):
        kwargs.update(log_var=kwargs.get("log_var", ("x", )), )
        super().__init__(**kwargs)

        if (f is None) or (H is None) or (g is None) or (R is None):
            msg = " ".join([
                " Parameters[f, H, g, WCR] must be specified.",
            ])
            raise NotImplementedError(msg)
        else:
            self._f = f
        self._H = H
        self._g = g
        self._R = R
        self._epsi = epsi
        self._bg = bg
        self._ub = ub
        self._lb = lb
        xp = pxu.get_array_module(g)
        self._Ht1 = xp.maximum(H.adjoint(xp.ones(H.codim_shape)), self._epsi)
        print(f'Min value in HT1 for RRL {self._Ht1.min()}')
        print(f'Set constraint for RRL [{self._lb},{self._ub}]')

    def m_init(
        self,
        x0: pxt.NDArray,
        acceleration: bool = False,
    ):
        mst = self._mstate  # shorthand
        mst["x"] = mst["x_prev"] = x0

        if acceleration:
            mst["acc"] = True
            mst["tk"] = 1
        else:
            mst["acc"] = False
            mst["a"] = itertools.repeat(0)

    def m_step(self):
        mst = self._mstate  # shorthand
        xp = pxu.get_array_module(mst["x"])
        # RL multiplicative update

        if mst["acc"]:
            tprev = mst["tk"]
            mst["tk"] = 0.5 * (1 + math.sqrt(1 + 4 * tprev**2))
            a = (tprev - 1) / mst["tk"]
        else:
            a = next(mst["a"])
        # In-place implementation of -----------------
        #   y = (1 + a) * mst["x"] - a * mst["x_prev"]
        p = mst["x"] - mst["x_prev"]
        p *= a
        p += mst["x"]
        p.clip(self._lb, self._ub, out=p) # NOTE: minimum value 1e-6 
        p *= (self._H.adjoint(self._g / xp.maximum(self._H(p) + self._bg, self._epsi)) -
              self._R.grad(p)) / self._Ht1
        p.clip(self._lb, self._ub, out=p)
        mst["x_prev"], mst["x"] = mst["x"], p

    def default_stop_crit(self) -> pxa.StoppingCriterion:
        from pyxu.opt.stop import RelError

        stop_crit = RelError(
            eps=1e-4,
            var="x",
            rank=self._f.dim_rank,
            f=None,
            norm=2,
            satisfy_all=True,
        )
        return stop_crit

    def objective_func(self) -> pxt.NDArray:
        func = lambda x: self._f.apply(self._H.apply(x) + self._bg) + self._R.apply(x)
        y = func(self._mstate["x"])
        self._mstate["f"] = y
        return y

    def solution(self) -> pxt.NDArray:
        """
        Returns
        -------
        x: NDArray
            (..., M1,...,MD) solution.
        """
        data, _ = self.stats()
        return data.get("x")
