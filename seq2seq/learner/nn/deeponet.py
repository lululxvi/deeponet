"""
@author: jpzxshi
"""
import torch
import torch.nn as nn

from .module import StructureNN
from .fnn import FNN

class DeepONet(StructureNN):
    '''Deep operator network.
    Input: [batch size, branch_dim + trunk_dim]
    Output: [batch size, 1]
    '''
    def __init__(self, branch_dim, trunk_dim, branch_depth=2, trunk_depth=3, width=50,
                 activation='relu', initializer='Glorot normal'):
        super(DeepONet, self).__init__()
        self.branch_dim = branch_dim
        self.trunk_dim = trunk_dim
        self.branch_depth = branch_depth
        self.trunk_depth = trunk_depth
        self.width = width
        self.activation = activation
        self.initializer = initializer
        
        self.modus = self.__init_modules()
        self.params = self.__init_params()
        self.__initialize()
        
    def forward(self, x):
        x_branch, x_trunk = x[..., :self.branch_dim], x[..., self.branch_dim:]
        x_branch = self.modus['Branch'](x_branch)
        for i in range(1, self.trunk_depth):
            x_trunk = self.modus['TrActM{}'.format(i)](self.modus['TrLinM{}'.format(i)](x_trunk))
        return torch.sum(x_branch * x_trunk, dim=-1, keepdim=True) + self.params['bias']
        
    def __init_modules(self):
        modules = nn.ModuleDict()
        modules['Branch'] = FNN(self.branch_dim, self.width, self.branch_depth, self.width,
                                self.activation, self.initializer)
        modules['TrLinM1'] = nn.Linear(self.trunk_dim, self.width)
        modules['TrActM1'] = self.Act
        for i in range(2, self.trunk_depth):
            modules['TrLinM{}'.format(i)] = nn.Linear(self.width, self.width)
            modules['TrActM{}'.format(i)] = self.Act
        return modules
            
    def __init_params(self):
        params = nn.ParameterDict()
        params['bias'] = nn.Parameter(torch.zeros([1]))
        return params
            
    def __initialize(self):
        for i in range(1, self.trunk_depth):
            self.weight_init_(self.modus['TrLinM{}'.format(i)].weight)
            nn.init.constant_(self.modus['TrLinM{}'.format(i)].bias, 0)