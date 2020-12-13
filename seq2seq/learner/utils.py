"""
@author: jpzxshi
"""
from functools import wraps
import time

import numpy as np
import torch

#
# Useful tools.
#
def timing(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        t = time.time()
        result = func(*args, **kwargs)
        print('\'' + func.__name__ + '\'' + ' took {} s'.format(time.time() - t))
        return result
    return wrapper

class lazy_property:
    def __init__(self, func): 
        self.func = func 
        
    def __get__(self, instance, cls): 
        val = self.func(instance)
        setattr(instance, self.func.__name__, val)
        return val
    
#
# Numpy tools.
#
def softmax(x):
    e_x = np.exp(x - np.max(x, axis=-1, keepdims=True))
    return e_x / np.sum(e_x, axis=-1, keepdims=True)

#
# Torch tools.
#
def cross_entropy_loss(y_pred, y_label):
    if y_pred.size() == y_label.size():
        return torch.mean(-torch.sum(torch.log_softmax(y_pred, dim=-1) * y_label, dim=-1))
    else:
        return torch.nn.CrossEntropyLoss()(y_pred, y_label.long())

def grad(y, x, create_graph=True, keepdim=False):
    '''
    y: [N, Ny] or [Ny]
    x: [N, Nx] or [Nx]
    Return dy/dx ([N, Ny, Nx] or [Ny, Nx]).
    '''
    N = y.size(0) if len(y.size()) == 2 else 1
    Ny = y.size(-1)
    Nx = x.size(-1)
    z = torch.ones_like(y[..., 0])
    dy = []
    for i in range(Ny):
        dy.append(torch.autograd.grad(y[..., i], x, grad_outputs=z, create_graph=create_graph)[0])
    shape = np.array([N, Ny])[2-len(y.size()):]
    shape = list(shape) if keepdim else list(shape[shape > 1])
    return torch.cat(dy, dim=-1).view(shape + [Nx])