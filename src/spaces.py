from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import matplotlib.pyplot as plt
import numpy as np
from pathos.pools import ProcessPool
from scipy import linalg, interpolate
from sklearn import gaussian_process as gp

import config
from utils import eig


class FinitePowerSeries:
    def __init__(self, N=100, M=1):
        self.N = N
        self.M = M

    def random(self, n):
        return 2 * self.M * np.random.rand(n, self.N) - self.M

    def eval_u_one(self, a, x):
        return np.dot(a, x ** np.arange(self.N))

    def eval_u(self, a, sensors):
        mat = np.ones((self.N, len(sensors)))
        for i in range(1, self.N):
            mat[i] = np.ravel(sensors ** i)
        return np.dot(a, mat)


class FiniteChebyshev:
    def __init__(self, N=100, M=1):
        self.N = N
        self.M = M

    def random(self, n):
        return 2 * self.M * np.random.rand(n, self.N) - self.M

    def eval_u_one(self, a, x):
        return np.polynomial.chebyshev.chebval(2 * x - 1, a)

    def eval_u(self, a, sensors):
        return np.polynomial.chebyshev.chebval(2 * np.ravel(sensors) - 1, a.T)


class GRF(object):
    def __init__(self, T, kernel="RBF", length_scale=1, N=1000, interp="cubic"):
        self.N = N
        self.interp = interp
        self.x = np.linspace(0, T, num=N)[:, None]
        if kernel == "RBF":
            K = gp.kernels.RBF(length_scale=length_scale)
        elif kernel == "AE":
            K = gp.kernels.Matern(length_scale=length_scale, nu=0.5)
        self.K = K(self.x)
        self.L = np.linalg.cholesky(self.K + 1e-13 * np.eye(self.N))

    def random(self, n):
        """Generate `n` random feature vectors.
        """
        u = np.random.randn(self.N, n)
        return np.dot(self.L, u).T

    def eval_u_one(self, y, x):
        """Compute the function value at `x` for the feature `y`.
        """
        if self.interp == "linear":
            return np.interp(x, np.ravel(self.x), y)
        f = interpolate.interp1d(
            np.ravel(self.x), y, kind=self.interp, copy=False, assume_sorted=True
        )
        return f(x)

    def eval_u(self, ys, sensors):
        """For a list of functions represented by `ys`,
        compute a list of a list of function values at a list `sensors`.
        """
        if self.interp == "linear":
            return np.vstack([np.interp(sensors, np.ravel(self.x), y).T for y in ys])
        p = ProcessPool(nodes=config.processes)
        res = p.map(
            lambda y: interpolate.interp1d(
                np.ravel(self.x), y, kind=self.interp, copy=False, assume_sorted=True
            )(sensors).T,
            ys,
        )
        return np.vstack(list(res))


class GRF_KL(object):
    def __init__(
        self, T, kernel="RBF", length_scale=1, num_eig=10, N=100, interp="cubic"
    ):
        if not np.isclose(T, 1):
            raise ValueError("Only support T = 1.")

        self.num_eig = num_eig
        if kernel == "RBF":
            kernel = gp.kernels.RBF(length_scale=length_scale)
        elif kernel == "AE":
            kernel = gp.kernels.Matern(length_scale=length_scale, nu=0.5)
        eigval, eigvec = eig(kernel, num_eig, N, eigenfunction=True)
        eigvec *= eigval ** 0.5
        x = np.linspace(0, T, num=N)
        self.eigfun = [
            interpolate.interp1d(x, y, kind=interp, copy=False, assume_sorted=True)
            for y in eigvec.T
        ]

    def bases(self, sensors):
        return np.array([np.ravel(f(sensors)) for f in self.eigfun])

    def random(self, n):
        """Generate `n` random feature vectors.
        """
        return np.random.randn(n, self.num_eig)

    def eval_u_one(self, y, x):
        """Compute the function value at `x` for the feature `y`.
        """
        eigfun = [f(x) for f in self.eigfun]
        return np.sum(eigfun * y)

    def eval_u(self, ys, sensors):
        """For a list of functions represented by `ys`,
        compute a list of a list of function values at a list `sensors`.
        """
        eigfun = np.array([np.ravel(f(sensors)) for f in self.eigfun])
        return np.dot(ys, eigfun)


def space_samples(space, T):
    features = space.random(100000)
    sensors = np.linspace(0, T, num=1000)
    u = space.eval_u(features, sensors[:, None])

    plt.plot(sensors, np.mean(u, axis=0), "k")
    plt.plot(sensors, np.std(u, axis=0), "k--")
    plt.plot(sensors, np.cov(u.T)[0], "k--")
    plt.plot(sensors, np.exp(-0.5 * sensors ** 2 / 0.2 ** 2))
    for ui in u[:3]:
        plt.plot(sensors, ui)
    plt.show()


def main():
    # space = FinitePowerSeries(N=100, M=1)
    # space = FiniteChebyshev(N=20, M=1)
    # space = GRF(1, length_scale=0.2, N=1000, interp="cubic")
    # space = GRF_KL(1, length_scale=0.2, num_eig=10, N=100, interp="cubic")
    # space_samples(space, 1)

    space1 = GRF(1, length_scale=0.1, N=100, interp="cubic")
    space2 = GRF(1, length_scale=1, N=100, interp="cubic")
    W2 = np.trace(space1.K + space2.K - 2 * linalg.sqrtm(space1.K @ space2.K)) ** 0.5 / 100 ** 0.5
    print(W2)


if __name__ == "__main__":
    main()
