from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import matplotlib.pyplot as plt
import numpy as np
from pathos.pools import ProcessPool
from sklearn import gaussian_process as gp

import config
from spaces import GRF, GRF_KL
from utils import eig, make_triple, timing, trapz


def KL():
    l = 0.2
    N = 1000
    kernel = gp.kernels.RBF(length_scale=l)
    # kernel = gp.kernels.Matern(length_scale=l, nu=0.5)  # AE
    # kernel = gp.kernels.Matern(length_scale=l, nu=2.5)

    eigval, eigfun = eig(kernel, 10, N, eigenfunction=True)
    print(eigval)

    variance = 0.999
    s = np.cumsum(eigval)
    idx = np.nonzero(s > variance)[0][1]
    print(idx + 1)

    x = np.linspace(0, 1, num=N)
    plt.plot(x, eigfun[:, 0])
    plt.plot(x, eigfun[:, idx - 1])
    plt.plot(x, eigfun[:, idx])
    plt.show()


class GRFs(object):
    def __init__(
        self, T, kernel, length_scale_min, length_scale_max, N=100, interp="linear"
    ):
        self.T = T
        self.kernel = kernel
        self.length_scale_min = length_scale_min
        self.length_scale_max = length_scale_max
        self.N = N
        self.interp = interp

    def random(self, n):
        return (self.length_scale_max - self.length_scale_min) * np.random.rand(
            n, 1
        ) + self.length_scale_min

    def eval_u_one(self, l, sensors, M):
        grf = GRF(
            self.T, kernel=self.kernel, length_scale=l[0], N=self.N, interp=self.interp
        )
        us = grf.random(M)
        ys = grf.eval_u(us, sensors)
        return np.ravel(ys)

    def eval_u(self, ls, sensors, M):
        return np.vstack([self.eval_u_one(l, sensors, M) for l in ls])

    def eval_KL_bases(self, ls, sensors, M):
        def helper(l):
            grf = GRF_KL(
                self.T,
                kernel=self.kernel,
                length_scale=l[0],
                num_eig=M,
                N=self.N,
                interp=self.interp,
            )
            return np.ravel(grf.bases(sensors))

        p = ProcessPool(nodes=config.processes)
        return np.vstack(p.map(helper, ls))


class SODESystem(object):
    def __init__(self, T, y0, Nx=None, npoints_output=None):
        """Stochastic ODE"""
        self.T = T
        self.y0 = y0
        self.Nx = Nx
        self.npoints_output = npoints_output

    @timing
    def gen_operator_data(self, space, Nx, M, num, representation):
        print("Generating operator data...", flush=True)
        features = space.random(num)
        sensors = np.linspace(0, self.T, num=Nx)[:, None]
        if representation == "samples":
            X = space.eval_u(features, sensors, M)
        elif representation == "KL":
            X = space.eval_KL_bases(features, sensors, M)
        t = self.T * np.random.rand(num)[:, None]
        y = self.eval_s(features, t)
        return [X, t], y

    @timing
    def gen_example_data(self, space, l, Nx, M, representation, num=100):
        print("Generating example operator data...", flush=True)
        features = np.full((num, 1), l)
        sensors = np.linspace(0, self.T, num=Nx)[:, None]
        if representation == "samples":
            X = space.eval_u(features, sensors, M)
        elif representation == "KL":
            X = space.eval_KL_bases(features, sensors, M)
        t = np.linspace(0, self.T, num=num)[:, None]
        y = self.eval_s(features, t)
        return [X, t], y

    def eval_s(self, features, t):
        sigma2 = 2 * features * t + 2 * features ** 2 * (np.exp(-t / features) - 1)
        # mean
        y = self.y0 * np.exp(1 / 2 * sigma2)
        # 2nd moment
        # y = self.y0**2 * np.exp(2 * sigma2)
        # 3rd moment
        # y = self.y0**3 * np.exp(9/2 * sigma2)
        # 4th moment
        # y = self.y0**4 * np.exp(8 * sigma2)
        return y

    @timing
    def gen_operator_data_path(self, space, Nx, M, num):
        print("Generating operator data...", flush=True)
        features = space.random(num)
        t = np.linspace(0, self.T, num=self.Nx)[:, None]
        bases = space.eval_KL_bases(features, t, M)
        rv = np.random.randn(num, M)
        # rv = np.clip(rv, -3.1, 3.1)
        p = ProcessPool(nodes=config.processes)
        s_values = np.array(p.map(self.eval_s_path, bases, rv))

        sensors = np.linspace(0, self.T, num=Nx)[:, None]
        sensor_values = space.eval_KL_bases(features, sensors, M)
        sensor_values = np.hstack((sensor_values, rv))
        res = [
            make_triple(sensor_values[i], t, s_values[i], self.npoints_output)
            for i in range(num)
        ]
        res = np.vstack(res)
        m = Nx * M
        return [res[:, :m], res[:, m:-1]], res[:, -1:]

    @timing
    def gen_example_data_path(self, space, l, Nx, M):
        print("Generating operator data...", flush=True)
        features = np.full((1, 1), l)
        t = np.linspace(0, self.T, num=self.Nx)[:, None]
        bases = space.eval_KL_bases(features, t, M)
        rv = np.random.randn(1, M)
        # rv = np.clip(rv, -3.1, 3.1)
        s_values = self.eval_s_path(bases[0], rv[0])

        sensors = np.linspace(0, self.T, num=Nx)[:, None]
        sensor_value = space.eval_KL_bases(features, sensors, M)
        return (
            [
                np.tile(sensor_value, (self.Nx, 1)),
                np.hstack((np.tile(rv, (self.Nx, 1)), t)),
            ],
            s_values[:, None],
        )

    def eval_s_path(self, bases, rv):
        bases = bases.reshape((-1, self.Nx))
        k = np.dot(rv, bases)
        h = self.T / (self.Nx - 1)
        K = trapz(k, h)
        return self.y0 * np.exp(K)


class SPDESystem(object):
    def __init__(self, T, f, Nx, M, npoints_output):
        """Stochastic PDE"""
        self.T = T
        self.f = f
        self.Nx = Nx
        self.M = M
        self.npoints_output = npoints_output

    def random_process(self, gp):
        # return np.exp(gp)
        return np.exp(0.1 * gp)

    @timing
    def gen_operator_data(self, space, Nx, M, num, representation):
        print("Generating operator data...", flush=True)
        features = space.random(num)
        # Generate outputs
        x = np.linspace(0, self.T, num=self.Nx)[:, None]
        sensor_values = self.random_process(space.eval_u(features, x, self.M))  # exp(b)
        p = ProcessPool(nodes=config.processes)
        s_values = np.array(p.map(self.eval_s, sensor_values))

        # Generate inputs
        sensors = np.linspace(0, self.T, num=Nx)[:, None]
        if representation == "samples":
            sensor_values = self.random_process(space.eval_u(features, sensors, M))
        elif representation == "KL":
            sensor_values = space.eval_KL_bases(features, sensors, M)
            # sensor_values = self.random_process(sensor_values)
        res = [
            make_triple(sensor_values[i], x, s_values[i], self.npoints_output)
            for i in range(num)
        ]
        res = np.vstack(res)
        m = sensor_values.shape[1]
        return [res[:, :m], res[:, m:-1]], res[:, -1:]

    @timing
    def gen_example_data(self, space, l, Nx, M, representation):
        print("Generating example operator data...", flush=True)
        features = np.full((1, 1), l)
        t = np.linspace(0, self.T, num=self.Nx)[:, None]
        sensor_values = self.random_process(space.eval_u(features, t, self.M))
        s_value = self.eval_s(sensor_values)

        sensors = np.linspace(0, self.T, num=Nx)[:, None]
        if representation == "samples":
            sensor_value = self.random_process(space.eval_u(features, sensors, M))
        elif representation == "KL":
            sensor_value = space.eval_KL_bases(features, sensors, M)
            # sensor_value = self.random_process(sensor_value)
        return [np.tile(sensor_value, (self.Nx, 1)), t], s_value[:, None]

    def eval_s(self, sensor_value):
        h = self.T / (self.Nx - 1)
        sensor_value = sensor_value.reshape((self.M, self.Nx))
        tmp = 1 / sensor_value  # exp(-b)
        v1 = trapz(tmp, h)
        tmp *= self.f * np.linspace(0, self.T, num=self.Nx)
        v2 = trapz(tmp, h)
        C = 1 / v1[:, -1:] * v2[:, -1:]
        v = C * v1 - v2
        return np.mean(v, axis=0)
        # return np.std(v, axis=0)
        # return np.mean(v ** 3, axis=0)
        # skewness
        # mean, std = np.mean(v, axis=0), np.std(v, axis=0)
        # std[0], std[-1] = 1, 1
        # return (np.mean(v ** 3, axis=0) - 3 * mean * std ** 2 - mean ** 3) / (std ** 3 + 1e-13)
        # return (np.mean(v ** 3, axis=0) - 3 * mean * std ** 2 - mean ** 3) / std ** 3
        # res = np.mean((v / std) ** 3, axis=0) - 3 * mean / std - (mean / std) ** 3
        # res[0], res[-1] = res[1], res[-2]
        # return res
        # kurtosis
        # return np.mean((v - mean) ** 4, axis=0) / (std ** 4 + 1e-13)

    @timing
    def gen_operator_data_path(self, space, Nx, M, num):
        print("Generating operator data...", flush=True)
        features = space.random(num)
        x = np.linspace(0, self.T, num=self.Nx)[:, None]
        bases = space.eval_KL_bases(features, x, M)
        rv = np.random.randn(num, M)
        # rv = np.clip(rv, -3.1, 3.1)
        p = ProcessPool(nodes=config.processes)
        s_values = np.array(p.map(self.eval_s_path, bases, rv))

        sensors = np.linspace(0, self.T, num=Nx)[:, None]
        sensor_values = space.eval_KL_bases(features, sensors, M)
        # sensor_values = self.random_process(sensor_values)
        sensor_values = np.hstack((sensor_values, rv))
        res = [
            make_triple(sensor_values[i], x, s_values[i], self.npoints_output)
            for i in range(num)
        ]
        res = np.vstack(res)
        m = Nx * M
        return [res[:, :m], res[:, m:-1]], res[:, -1:]

    @timing
    def gen_example_data_path(self, space, l, Nx, M):
        print("Generating operator data...", flush=True)
        features = np.full((1, 1), l)
        t = np.linspace(0, self.T, num=self.Nx)[:, None]
        bases = space.eval_KL_bases(features, t, M)
        rv = np.random.randn(1, M)
        # rv = np.clip(rv, -3.1, 3.1)
        s_values = self.eval_s_path(bases[0], rv[0])

        sensors = np.linspace(0, self.T, num=Nx)[:, None]
        sensor_value = space.eval_KL_bases(features, sensors, M)
        # sensor_value = self.random_process(sensor_value)
        return (
            [
                np.tile(sensor_value, (self.Nx, 1)),
                np.hstack((np.tile(rv, (self.Nx, 1)), t)),
            ],
            s_values[:, None],
        )

    def eval_s_path(self, bases, rv):
        bases = bases.reshape((-1, self.Nx))
        b = np.dot(rv, bases)
        h = self.T / (self.Nx - 1)
        tmp = np.exp(-b)
        v1 = trapz(tmp, h)
        tmp *= self.f * np.linspace(0, self.T, num=self.Nx)
        v2 = trapz(tmp, h)
        C = 1 / v1[-1] * v2[-1]
        v = C * v1 - v2
        return v


def main():
    # KL()
    # return

    # SODE statistical averages
    # system = SODESystem(1, 1)
    # representation = "samples"
    # space = GRFs(1, "AE", 1, 2, N=10, interp="linear")
    # Nx = 10
    # M = 10
    # representation = "KL"
    # space = GRFs(1, "AE", 1, 2, N=100, interp="linear")
    # Nx = 20
    # M = 5
    # X, y = system.gen_operator_data(space, Nx, M, 1000000, representation)
    # np.savez_compressed("train.npz", X_train0=X[0], X_train1=X[1], y_train=y)
    # X, y = system.gen_operator_data(space, Nx, M, 1000000, representation)
    # np.savez_compressed("test.npz", X_test0=X[0], X_test1=X[1], y_test=y)
    # X, y = system.gen_example_data(space, 1.5, Nx, M, representation, num=100)
    # np.savez_compressed("example.npz", X_test0=X[0], X_test1=X[1], y_test=y)

    # SPDE statistical averages
    system = SPDESystem(1, 10, 100, 20000, 10)
    space = GRFs(1, "RBF", 0.2, 2, N=100, interp="linear")
    # representation = "samples"
    # Nx = 10
    # M = 10
    representation = "KL"
    Nx = 30
    M = 8
    X, y = system.gen_operator_data(space, Nx, M, 1000, representation)
    np.savez_compressed("train.npz", X_train0=X[0], X_train1=X[1], y_train=y)
    X, y = system.gen_operator_data(space, Nx, M, 1000, representation)
    np.savez_compressed("test.npz", X_test0=X[0], X_test1=X[1], y_test=y)
    for i in range(10):
        X, y = system.gen_example_data(space, 0.2 + 0.2 * i, Nx, M, representation)
        np.savez_compressed(
            "example{}.npz".format(i), X_test0=X[0], X_test1=X[1], y_test=y
        )
    return

    # SODE/SPDE pathwise solution
    system = SODESystem(1, 1, Nx=100, npoints_output=100)
    # system = SPDESystem(1, 10, 100, None, 100)
    space = GRFs(1, "RBF", 1, 2, N=100, interp="linear")
    Nx = 20
    M = 5
    X, y = system.gen_operator_data_path(space, Nx, M, 10000)
    np.savez_compressed("train.npz", X_train0=X[0], X_train1=X[1], y_train=y)
    X, y = system.gen_operator_data_path(space, Nx, M, 10000)
    np.savez_compressed("test.npz", X_test0=X[0], X_test1=X[1], y_test=y)
    for i in range(10):
        X, y = system.gen_example_data_path(space, 1.5, Nx, M)
        np.savez_compressed(
            "example{}.npz".format(i), X_test0=X[0], X_test1=X[1], y_test=y
        )


if __name__ == "__main__":
    main()
