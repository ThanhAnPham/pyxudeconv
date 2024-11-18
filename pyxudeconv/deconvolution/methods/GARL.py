# Goujon Accelerated Richardson-Lucy
import numpy as np

#from libraries import *
#import pyxu
from pyxu.operator import KLDivergence
import os
import glob

import pyxu.abc as pxa
import pyxu.info.ptype as pxt
import torch
from pyxu.operator.interop import from_source
from pyxu.operator.interop import torch as pxotorch

from pyxudeconv.deconvolution.methods.models_GARL.wc_conv_net import WCvxConvNet, WCvxConvNet3DS, WCvxConvNet3D
import json

#in future release, RRL will be incorporated in Pyxu or at least as a plugin
from pyxudeconv.deconvolution.methods.solver import RRL

from .ABC import HyperParametersDeconvolutionOptimizer
import importlib

import inspect

parent_dir = os.path.dirname(
    os.path.abspath(inspect.getfile(inspect.currentframe())))

__all__ = [
    "GARL",
]


class GARL(HyperParametersDeconvolutionOptimizer):
    r"""
    Hyper parameters optimizer for Goujon Accelerated Richardson-Lucy
    """

    def get_hyperparams(self):
        if isinstance(self._param_method, dict):
            #Parameters are directly provided as a dictionary (possible if deconvolve is called from Python)
            if self._param_method['model'][0]=='':
                #load trained model in package if 'model' empty string ''
                curr_modality = 'widefield_params' if len(
                    self._forw.codim_shape) == len(
                        self._forw.dim_shape) else 'airyscan_params'
                module_config = importlib.import_module(
                    'pyxudeconv.deconvolution.methods.configs.GARL.' +
                    curr_modality)
                config_fct = getattr(module_config, curr_modality)
                subparams = config_fct()
                self._param_method['model'] = subparams['model']
            return self._param_method
        if isinstance(self._param_method, str):
            if '.json' in self._param_method[self._param_method.rfind('.'):]:
                with open(self._param_method, 'r', encoding="utf-8") as f:
                    params = json.load(f)

                new_params = params.copy()
                for k in params.keys():
                    if 'min' in k:
                        cp = k[:k.rfind('_min')]
                        new_params[cp] = np.linspace(
                            params[cp + '_min'],
                            params[cp + '_max'],
                            params[cp + '_nsteps'],
                        )
                        new_params.pop(cp + '_min')
                        new_params.pop(cp + '_max')
                        new_params.pop(cp + '_nsteps')
                return new_params
        else:
            #Load some default configurations in the package
            self._param_method = 'widefield_params' if len(
                self._forw.codim_shape) == len(
                    self._forw.dim_shape) else 'airyscan_params'
        module_config = importlib.import_module(
            'pyxudeconv.deconvolution.methods.configs.GARL.' +
            self._param_method)
        config_fct = getattr(module_config, self._param_method)
        params = config_fct()
        return params

    def init_solver(self, param):
        device = torch.device(self._device_name)
        lossRL = KLDivergence(self._g)
        reg_shape = self._trim_buffer.codim_shape  #chooses where the regularization is enforced
        NN, applyNN, gradNN, proxNN = load_model_weakly_convex(
            param['model'],
            sigma=torch.tensor(param['sigma'], dtype=torch.float32).to(device),
            device=self._device_name,
            epoch=param['epochoi'],
            do3D=True,
            doSplineActivation=False,
        )
        NN.to(device)
        NN.conv_layer.spectral_norm(mode="power_method", n_steps=200)
        R = pxotorch.from_torch(dim_shape=reg_shape,
                                codim_shape=(1),
                                apply=applyNN,
                                cls=pxa.ProxDiffFunc,
                                grad=gradNN,
                                prox=None)
        R.lipschitz = NN.get_mu().maximum(
            torch.tensor(1)).detach().cpu().numpy()

        R = param['lmbd'] * R * self._trim_buffer
        self._solver = RRL(
            lossRL,
            self._forw,
            self._g,
            R,
            verbosity=self._disp,
            stop_rate=1,
            show_progress=False,
            bg=self._bg_est,
        )


def load_model_weakly_convex(
    name,
    sigma=None,
    device='cuda:0',
    epoch=None,
    do3D=False,
    doSplineActivation=True,
    #current_directory=None,
):
    directory_checkpoints = os.path.join(name, 'checkpoints')

    # retrieve last checkpoint
    if epoch is None:
        files = glob.glob(os.path.join(directory_checkpoints, 'checkpoints',
                                       '*.pth'),
                          recursive=False)
        epochs = map(
            lambda x: int(x.split("/")[-1].split('.pth')[0].split('_')[1]),
            files)
        epoch = max(epochs)

    checkpoint_path = os.path.join(directory_checkpoints,
                                   f'checkpoint_{epoch}.pth')
    # config file
    config = json.load(
        open(os.path.join(name, 'config.json'), encoding="utf-8"))

    # build model

    model, _ = build_modelWC(config, do3D, doSplineActivation)

    checkpoint = torch.load(checkpoint_path,
                            map_location=device,
                            weights_only=True)

    #breakpoint()
    model.to(device)

    model.load_state_dict(checkpoint['state_dict'])
    model.conv_layer.spectral_norm()
    model.eval()

    if 'cuda' in device:
        import cupy as xp
    else:
        import numpy as xp

    def arrange(arr: pxt.NDArray) -> pxt.NDArray:
        arr = arr.to(torch.float32)
        if do3D:
            if arr.ndim == 2:
                arr = arr.unsqueeze(0).unsqueeze(0).unsqueeze(
                    0)  #doesn't make sense
            elif arr.ndim == 3:
                arr = arr.unsqueeze(0)
            elif arr.ndim == 4:  #multichannels, but processed separately
                if arr.shape[-4] > 1:  #concatenate all channels in samples dim
                    arr = arr.reshape(-1, 1, *arr.shape[-3:])
        else:
            if arr.ndim == 2:
                arr = arr.unsqueeze(0).unsqueeze(0)
            elif arr.ndim == 3:
                if arr.shape[-3] > 1:  #expects one channel only
                    arr = arr.unsqueeze(1)
                else:
                    arr = arr.unsqueeze(0)
            elif arr.ndim == 4:
                if arr.shape[-3] > 1:  #concatenate all in samples dim
                    arr = arr.reshape(-1, 1, *arr.shape[2:])
        return arr

    def applyNN(arr: pxt.NDArray) -> pxt.NDArray:
        with torch.no_grad():
            out = model.cost(arrange(arr), sigma=sigma)
            return out.sum()

    def gradNN(arr: pxt.NDArray) -> pxt.NDArray:
        with torch.no_grad():
            shape_og = arr.shape
            return model.grad(arrange(arr), sigma=sigma).reshape(shape_og)

    def proxNN(arr: pxt.NDArray, rho: pxt.Real) -> pxt.NDArray:
        with torch.no_grad():
            return model.forward(arrange(arr))  #ToDo

    return model, applyNN, gradNN, proxNN


def build_modelWC(config, do3D, doSplineActivation):
    # ensure consistency of the config file, e.g. number of channels, ranges + enforce constraints
    if doSplineActivation:
        # 1- Activation function (learnable spline)
        param_spline_activation = config['spline_activation']
        # non expansive increasing splines
        param_spline_activation["slope_min"] = 0
        param_spline_activation["slope_max"] = 1
        # antisymmetric splines
        param_spline_activation["antisymmetric"] = True
        # shared spline
        param_spline_activation["num_activations"] = 1

    # 2- Multi convolution layer
    param_multi_conv = config['multi_convolution']
    if len(param_multi_conv['num_channels']) != (
            len(param_multi_conv['size_kernels']) + 1):
        raise ValueError(
            "Number of channels specified is not compliant with number of kernel sizes"
        )

    param_spline_scaling = config['spline_scaling']
    param_spline_scaling["clamp"] = False
    param_spline_scaling["x_min"] = config['noise_range'][0]
    param_spline_scaling["x_max"] = config['noise_range'][1]
    param_spline_scaling["num_activations"] = config['multi_convolution'][
        'num_channels'][-1]

    if do3D:
        if doSplineActivation:
            model = WCvxConvNet3D(
                param_multi_conv=param_multi_conv,
                param_spline_activation=param_spline_activation,
                param_spline_scaling=param_spline_scaling,
                rho_wcvx=config['rho_wcvx'])
        else:
            model = WCvxConvNet3DS(param_multi_conv=param_multi_conv,
                                   param_spline_scaling=param_spline_scaling,
                                   rho_wcvx=config['rho_wcvx'])
    else:
        model = WCvxConvNet(param_multi_conv=param_multi_conv,
                            param_spline_activation=param_spline_activation,
                            param_spline_scaling=param_spline_scaling,
                            rho_wcvx=config['rho_wcvx'])

    return (model, config)
