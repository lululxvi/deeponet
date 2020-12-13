"""
@author: jpzxshi
"""
from .module import Module
from .module import StructureNN
from .module import LossNN
from .fnn import FNN
from .hnn import HNN
from .sympnet import LASympNet
from .sympnet import GSympNet
from .seq2seq import S2S
from .deeponet import DeepONet

__all__ = [
    'Module',
    'StructureNN',
    'LossNN',
    'FNN',
    'HNN',
    'LASympNet',
    'GSympNet',
    'S2S',
    'DeepONet',
]


