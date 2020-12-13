from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


def solve_ADVD(xmin, xmax, tmin, tmax, f, g, V, Nx, Nt):
    """Solve
    u_t + u_x - D * u_xx = 0
    u(x, 0) = V(x)
    """
    # Crank-Nicholson
    D = 0.1
    x = np.linspace(xmin, xmax, Nx)
    t = np.linspace(tmin, tmax, Nt)
    h = x[1] - x[0]
    dt = t[1] - t[0]
    lam = dt / h
    mu = dt / h ** 2
    u = np.zeros([Nx, Nt])
    u[:, 0] = V(x)

    I = np.eye(Nx - 1)
    I1 = np.roll(I, 1, axis=0)
    I2 = np.roll(I, -1, axis=0)
    A = (1 + D * mu) * I - (lam / 4 + D * mu / 2) * I1 + (lam / 4 - D * mu / 2) * I2
    B = 2 * I - A
    C = np.linalg.solve(A, B)

    for n in range(Nt - 1):
        u[1:, n + 1] = C @ u[1:, n]
    u[0, :] = u[-1, :]

    return x, t, u


def main():
    xmin, xmax = 0, 1
    tmin, tmax = 0, 1
    V = lambda x: np.sin(2 * np.pi * x)  # 1 - 2 * x#np.sin(2 * np.pi * x)
    f = None
    g = None
    D = 0.1

    u_true = lambda x, t: np.exp(-4 * np.pi ** 2 * D * t) * np.sin(
        2 * np.pi * (x - t)
    )  # V(np.sin(np.pi * (x - t)) ** 2)

    Nx, Nt = 100, 991
    x, t, u = solve_ADVD(xmin, xmax, tmin, tmax, f, g, V, Nx, Nt)

    print(np.max(abs(u - u_true(x[:, None], t))))
    print(np.average(abs(u - u_true(x[:, None], t))))
    # diff = u - u_true(x[:, None], t)
    # plt.plot(x, u)
    # plt.show()

    u_true = u_true(x[:, None], t)[:, 0:991:10]
    u = u[:, 0:991:10]
    error = abs(u - u_true)
    axis = plt.subplot(111)
    sns.heatmap(error, linewidths=0.00, ax=axis, cmap="rainbow")
    xlabel = [format(i, ".1f") for i in np.linspace(0, 1, num=11)]
    ylabel = [format(i, ".1f") for i in np.linspace(0, 1, num=11)]
    axis.set_xticks(range(0, 101, 10))
    axis.set_xticklabels(xlabel)
    axis.set_yticks(range(0, 101, 10))
    axis.set_yticklabels(ylabel)
    axis.set_xlabel("t")
    axis.set_ylabel("x")
    axis.set_title(r"Error", fontdict={"fontsize": 30}, loc="left")


if __name__ == "__main__":
    main()
