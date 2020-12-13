"""
@author: jpzxshi
"""
import numpy as np
import torch

from .module import LossNN
from .fnn import FNN
from ..integrator.hamiltonian import SV
from ..utils import lazy_property, grad

class HNN(LossNN):
    '''Hamiltonian neural networks.
    '''
    def __init__(self, dim, layers=3, width=30, activation='tanh', initializer='orthogonal', integrator='midpoint'):
        super(HNN, self).__init__()
        self.dim = dim
        self.layers = layers
        self.width = width
        self.activation = activation
        self.initializer = initializer
        self.integrator = integrator
        
        self.modus = self.__init_modules()
    
    def criterion(self, x0h, x1):
        x0, h = (x0h[..., :-1], x0h[..., -1:])
        return self.__integrator_loss(x0, x1, h)
    
    def predict(self, x0, h, steps=1, keepinitx=False, returnnp=False):
        N = max(int(h * 10), 1)
        solver = SV(self.modus['H'], None, iterations=10, order=4, N=N)
        res = solver.flow(x0, h, steps) if keepinitx else solver.flow(x0, h, steps)[..., 1:, :].squeeze()
        return res.cpu().detach().numpy() if returnnp else res
    
    @lazy_property
    def J(self):
        d = int(self.dim / 2)
        res = np.eye(self.dim, k=d) - np.eye(self.dim, k=-d)
        return torch.tensor(res, dtype=self.Dtype, device=self.Device)
    
    def __init_modules(self):
        modules = torch.nn.ModuleDict()
        modules['H'] = FNN(self.dim, 1, self.layers, self.width, self.activation, self.initializer)
        return modules 
    
    def __integrator_loss(self, x0, x1, h):
        if self.integrator == 'midpoint':
            mid = ((x0 + x1) / 2).requires_grad_(True)
            gradH = grad(self.modus['H'](mid), mid)
            return torch.nn.MSELoss()((x1 - x0) / h, gradH @ self.J)
        else:
            raise NotImplementedError