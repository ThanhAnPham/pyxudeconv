from abc import ABC, abstractmethod
import numpy as np

import pyxu
from pyxu.operator import KLDivergence
import pyxu.info.deps as pxd
from pyxu.abc.solver import SolverMode

import tifffile
import time
import itertools
import pyxu.info.ptype as pxt

__all__ = [
    "HyperParametersOptimizer",
]


class HyperParametersOptimizer(ABC):
    '''Abstract class to optimize hyperparameters of a Pyxu solver.
    There are two abstract methods: init_solver and get_hyperparams
    '''
    _solver: pxt.SolverT
    r"""
    Abstract class for hyper parameters optimization
    """

    def __init__(
        self,
        stop_crit,
        cmp_metrics,
        phantom,
        op4metrics,
        fct4name,
        op4save,
        save_results,
        pxsz=1,
        pxunit='N/A',
        **kwargs,
    ):
        self.stop_crit = stop_crit
        self.cmp_metrics = cmp_metrics
        self.phantom = phantom
        self.op4metrics = op4metrics
        self.fct4name = fct4name
        self.op4save = op4save
        self._save_results = save_results
        self._pxunit = pxunit
        self._pxsz = pxsz

    @abstractmethod
    def init_solver(self, param):
        '''Return a Pyxu solver ready to solve an optimization problem with the input param'''
        pass

    @abstractmethod
    def get_hyperparams(self):
        '''Return a dictionary with keys for each parameter required by the solver. Values are any value to be optimized over (by default, grid search over all combinations)'''
        pass

    def optimize_hyperparams(self, x0, save_iter, logger):
        '''Find the best set of hyperparameters among the combinations allowed by the method :func:`~ABC.HyperParametersOptimizer.get_hyperparams`'''
        meth = type(self).__name__
        param_meth = self.get_hyperparams()
        bestmetric = -1e9
        bestrecon = None
        logger.info(f'Saving results every {save_iter} iterations')
        for cparam in itertools.product(*param_meth.values()):
            cparamstr = '_'.join([
                '{} : {}'.format(c[0], c[1])
                for c in zip(param_meth.keys(), cparam)
            ])
            logger.info('Parameters | ' + cparamstr.replace('_', ' | '))
            cparamstr = cparamstr.replace(' : ', '_').replace('/', '_')
            cparam = dict(zip(param_meth.keys(), cparam))
            self.init_solver(cparam)

            t0 = time.time()
            self._solver.fit(
                mode=SolverMode.MANUAL,
                x0=x0,
                stop_crit=self.stop_crit,
                track_objective=True,
                acceleration=True,
            )
            for citer, data in enumerate(self._solver.steps()):
                citer += 1
                if citer % save_iter == 0:
                    recon = data['x'].copy()
                    recon_metric = self.cmp_metrics(self.phantom,
                                                    self.op4metrics(recon))
                    fname = self.fct4name(
                        meth,
                        cparamstr + f'_iter_{citer}',
                        recon_metric,
                    )

                    self.save_results(recon, fname)
                    if self.phantom is None:
                        logger.info('{:d} | {} in {:.2f}sec'.format(
                            citer,
                            meth,
                            time.time() - t0,
                        ))
                    else:
                        logger.info(
                            '{:d} | Metric {}: {:.4e} in {:.2f}sec'.format(
                                citer,
                                meth,
                                recon_metric,
                                time.time() - t0,
                            ))
                    if recon_metric > bestmetric:
                        bestmetric = recon_metric
                        bestrecon = recon
                        bestparams = cparam
                        bestparams['iter'] = citer

            recon = self._solver.solution()
            recon_metric = self.cmp_metrics(self.phantom,
                                            self.op4metrics(recon))
            fname = self.fct4name(meth, cparamstr + '_last', recon_metric)
            self.save_results(recon, fname)
            if (recon_metric > bestmetric) or self.phantom is None:
                bestmetric = recon_metric
                bestrecon = recon
                bestparams = cparam
        return self.op4save(bestrecon), bestparams, bestmetric

    def save_results(self, recon, fname):
        '''
        Save results as tiff file by default or do whatever self.
        '''
        self._save_results(self.op4save(recon)
            if isinstance(recon,np.ndarray) else self.op4save(recon).get(),fname,self._pxsz,self._pxunit)


class HyperParametersDeconvolutionOptimizer(HyperParametersOptimizer):
    r"""
    Abstract class for Hyper parameters optimizer for deconvolution
    """

    def __init__(
        self,
        forw,
        g,
        bg_est,
        trim_buffer,
        device_name,
        disp,
        param_method,
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self._forw = forw
        self._g = g
        self._bg_est = bg_est
        self._trim_buffer = trim_buffer
        self._device_name = device_name
        self._disp = disp
        self._param_method = param_method
