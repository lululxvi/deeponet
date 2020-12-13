"""
@author: jpzxshi
"""
import abc
import numpy as np
from sklearn import gaussian_process as gp
from scipy import interpolate
from scipy.integrate import solve_ivp

import learner as ln

class ODEData(ln.Data, abc.ABC):
    '''dy/dt=g(y,u,t), y(0)=s0, 0<=t<=T.
    '''
    def __init__(self, T, s0, sensor_in, sensor_out, length_scale, train_num, test_num):
        super(ODEData, self).__init__()
        self.T = T
        self.s0 = s0
        self.sensor_in = sensor_in
        self.sensor_out = sensor_out
        self.length_scale = length_scale
        self.train_num = train_num
        self.test_num = test_num
        self.__init_data()
        
    @abc.abstractmethod
    def g(self, y, u, t):
        pass
        
    def __init_data(self):
        features = 1000 * self.T
        train = self.__gaussian_process(self.train_num, features)
        test = self.__gaussian_process(self.test_num, features)
        self.X_train = self.__sense(train).reshape([-1, self.sensor_in, 1])
        self.y_train = self.__solve(train).reshape([-1, self.sensor_out, 1])
        self.X_test = self.__sense(test).reshape([-1, self.sensor_in, 1])
        self.y_test = self.__solve(test).reshape([-1, self.sensor_out, 1])
    
    def __gaussian_process(self, num, features):
        x = np.linspace(0, self.T, num=features)[:, None]
        K = gp.kernels.RBF(length_scale=self.length_scale)(x)
        L = np.linalg.cholesky(K + 1e-13 * np.eye(features))
        return (L @ np.random.randn(features, num)).transpose()
    
    def __sense(self, gps):
        x = np.linspace(0, self.T, num=gps.shape[1])
        res = map(
            lambda y: interpolate.interp1d(x, y, kind='cubic', copy=False, assume_sorted=True
            )(np.linspace(0, self.T, num=self.sensor_in)),
            gps)
        return np.vstack(list(res))
    
    def __solve(self, gps):
        x = np.linspace(0, self.T, num=gps.shape[1])
        interval = np.linspace(0, self.T, num=self.sensor_out) if self.sensor_out > 1 else [self.T]
        def solve(y):
            u = interpolate.interp1d(x, y, kind='cubic', copy=False, assume_sorted=True)
            return solve_ivp(lambda t, y: self.g(y, u(t), t), [0, self.T], self.s0, 'RK45', interval, max_step=0.05).y[0]
        return np.vstack(list(map(solve, gps)))

class AntideData(ODEData):
    '''Data for learning the antiderivative operator.
    g(y,u,t)=u.
    '''
    def __init__(self, T, s0, sensor_in, sensor_out, length_scale, train_num, test_num):
        super(AntideData, self).__init__(T, s0, sensor_in, sensor_out, length_scale, train_num, test_num)
    
    def g(self, y, u, t):
        return u
    
class PendData(ODEData):
    '''Data for learning the gravity pendulum.
    g(y,u,t)=[y[1], -np.sin(s[0]) + u].
    '''
    def __init__(self, T, s0, sensor_in, sensor_out, length_scale, train_num, test_num):
        super(PendData, self).__init__(T, s0, sensor_in, sensor_out, length_scale, train_num, test_num)
        
    def g(self, y, u, t):
        return [y[1], -np.sin(y[0]) + u]