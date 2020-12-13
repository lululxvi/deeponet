"""
@author: jpzxshi
"""
import numpy as np
import torch

from ...utils import grad

class SV:
    '''Stormer-Verlet scheme.
    '''
    def __init__(self, H, dH, iterations=10, order=4, N=1):
        '''
        H: H(x) or None
        dH: dp,dq=dH(p,q) or None
        ``iterations`` is encouraged to be 1 if H is separable.
        '''
        self.H = H
        self.dH = dH
        self.iterations = iterations
        self.order = order
        self.N = N
        
    def __sv2(self, x, h):
        '''Order 2.
        x: np.ndarray or torch.Tensor of shape [dim] or [num, dim].
        h: int
        '''
        dim = x.shape[-1] if isinstance(x, np.ndarray) else x.size(-1)
        d = int(dim / 2)
        p0, q0 = (x[..., :d], x[..., d:])
        p1, q1 = p0, q0
        if callable(self.dH):
            for _ in range(self.iterations):
                p1 = p0 - h / 2 * self.dH(p1, q0)[1]
            q1 = q0 + h / 2 * self.dH(p1, q0)[0]
            p2, q2 = p1, q1 
            for _ in range(self.iterations):
                q2 = q1 + h / 2 * self.dH(p1, q2)[0]
            p2 = p1 - h / 2 * self.dH(p1, q2)[1]
            return np.hstack([p2, q2]) if isinstance(x, np.ndarray) else torch.cat([p2, q2], dim=-1)
        elif isinstance(x, torch.Tensor):
            for _ in range(self.iterations):
                x = torch.cat([p1, q0], dim=-1).requires_grad_(True)
                dH = grad(self.H(x), x, create_graph=False)
                p1 = p0 - h / 2 * dH[..., d:]
            q1 = q0 + h / 2 * dH[..., :d]
            p2, q2 = p1, q1
            for _ in range(self.iterations):
                x = torch.cat([p1, q2], dim=-1).requires_grad_(True)
                dH = grad(self.H(x), x, create_graph=False)
                q2 = q1 + h / 2 * dH[..., :d]
            p2 = p1 - h / 2 * dH[..., d:]
            return torch.cat([p2, q2], dim=-1)
        else:
            raise ValueError
    
    def __sv4(self, x, h):
        '''Order 4.
        '''
        r1 = 1 / (2 - 2 ** (1 / 3))
        r2 = - 2 ** (1 / 3) / (2 - 2 ** (1 / 3))
        return self.__sv2(self.__sv2(self.__sv2(x, r1 * h), r2 * h), r1 * h)
     
    def __sv6(self, x, h):
        '''Order 6
        '''
        r1 = 1 / (2 - 2 ** (1 / 5))
        r2 = - 2 ** (1 / 5) / (2 - 2 ** (1 / 5))
        return self.__sv4(self.__sv4(self.__sv4(x, r1 * h), r2 * h), r1 * h)
        
    def solve(self, x, h):
        if self.order == 2:
            solver = self.__sv2
        elif self.order == 4:
            solver = self.__sv4
        elif self.order == 6:
            solver = self.__sv6
        else:
            raise NotImplementedError
        for _ in range(self.N):
            x = solver(x, h / self.N)
        return x
    
    def flow(self, x, h, steps):
        dim = x.shape[-1] if isinstance(x, np.ndarray) else x.size(-1)
        size = len(x.shape) if isinstance(x, np.ndarray) else len(x.size())
        X = [x]
        for i in range(steps):
            X.append(self.solve(X[-1], h))
        shape = [steps + 1, dim] if size == 1 else [-1, steps + 1, dim]
        return np.hstack(X).reshape(shape) if isinstance(x, np.ndarray) else torch.cat(X, dim=-1).view(shape)