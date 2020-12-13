from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import matplotlib.pyplot as plt


def solve_CVC(xmin, xmax, tmin, tmax, f, g, V, Nx, Nt):
    """Solve
    u_t + a(x) * u_x = 0
    """

    # Case I: Analytical solution for a(x)=1, u(x,0)=V(x)    (V,V' periodic)
    x = np.linspace(xmin, xmax, Nx)
    t = np.linspace(tmin, tmax, Nt)
    u = V((x[:, None] - t) % 1)

    # Case II: Wendroff for a(x)=1, u(x,0)=f(x), u(0,t)=g(t)    (f(0)=g(0))
    """
    x = np.linspace(xmin, xmax, Nx)
    t = np.linspace(tmin, tmax, Nt)
    h = x[1] - x[0]
    dt = t[1] - t[0]
    lam = dt / h
    u = np.zeros([Nx, Nt])
    u[:, 0] = f(x)
    u[0, :] = g(t)

    r = (1 - lam) / (1 + lam)
    K = np.eye(Nx - 1, k=0)
    K_temp = np.eye(Nx - 1, k=0)
    Trans = np.eye(Nx - 1, k=-1)
    for _ in range(Nx - 2):
        K_temp = (-r) * (Trans @ K_temp)
        K += K_temp
    D = r * np.eye(Nx - 1, k=0) + np.eye(Nx - 1, k=-1)

    for n in range(Nt - 1):
        b = np.zeros(Nx - 1)
        b[0] = g(n * dt) - r * g((n + 1) * dt)
        u[1:, n + 1] = K @ (D @ u[1:, n] + b)
    """

    # Case III: Wendroff for a(x)=1+0.1*V(x), u(x,0)=f(x), u(0,t)=g(t)    (f(0)=g(0))
    """
    x = np.linspace(xmin, xmax, Nx)
    t = np.linspace(tmin, tmax, Nt)
    h = x[1] - x[0]
    dt = t[1] - t[0]
    lam = dt / h
    v = 1 + 0.1 * V(x)
    u = np.zeros([Nx, Nt])
    u[:, 0] = f(x)
    u[0, :] = g(t)
    a = (v[:-1] + v[1:]) / 2
    k = (1 - a * lam) / (1 + a * lam)
    K = np.eye(Nx - 1, k=0)
    K_temp = np.eye(Nx - 1, k=0)
    Trans = np.eye(Nx - 1, k=-1)
    for _ in range(Nx - 2):
        K_temp = (-k[:, None]) * (Trans @ K_temp)
        K += K_temp
    D = np.diag(k) + np.eye(Nx - 1, k=-1)

    for n in range(Nt - 1):
        b = np.zeros(Nx - 1)
        b[0] = g(n * dt) - k[0] * g((n + 1) * dt)
        u[1:, n + 1] = K @ (D @ u[1:, n] + b)
    """

    # Case IV: Wendroff for a(x)=1+0.1*(V(x)+V(1-x))/2, u(x,0)=f(x)   (f,f' periodic)
    """
    x = np.linspace(xmin, xmax, Nx)
    t = np.linspace(tmin, tmax, Nt)
    h = x[1] - x[0]
    dt = t[1] - t[0]
    lam = dt / h
    v = 1 + 0.1 * (V(x) + V(x)[::-1]) / 2
    u = np.zeros([Nx, Nt])
    u[:, 0] = f(x)

    a = (v[:-1] + v[1:]) / 2
    I = np.eye(Nx - 1)
    Ir = np.roll(I, 1, axis=0)
    D = lam * a[:, None] * (I - Ir)
    A = I + Ir + D
    B = I + Ir - D

    for n in range(Nt - 1):
        u[1:, n + 1] = np.linalg.solve(A, B @ u[1:, n])
    u[0, :] = u[-1, :]
    """

    return x, t, u


def main():
    # Case I: Analytical solution for a(x)=1, u(x,0)=V(x)    (V,V' periodic)
    xmin, xmax = 0, 1
    tmin, tmax = 0, 1
    V = lambda x: np.sin(2 * np.pi * x)
    f = None
    g = None

    u_true = lambda x, t: V(x - t)

    Nx, Nt = 100, 100
    x, t, u = solve_CVC(xmin, xmax, tmin, tmax, f, g, V, Nx, Nt)

    print(np.max(abs(u - u_true(x[:, None], t))))
    print(np.average(abs(u - u_true(x[:, None], t))))

    # Case II: Wendroff for a(x)=1, u(x,0)=V(x), u(0,t)=0    (V(0)=0)
    """
    xmin, xmax = 0, 1
    tmin, tmax = 0, 1
    V = None
    f = lambda x: (2 * np.pi * x) ** 5
    g = lambda t: (2 * np.pi * (-t)) ** 5

    u_true = lambda x, t: (2 * np.pi * (x - t)) ** 5

    Nx, Nt = 100, 100
    x, t, u = solve_CVC(xmin, xmax, tmin, tmax, f, g, V, Nx, Nt)

    print(np.max(abs(u - u_true(x[:, None], t))))
    print(np.average(abs(u - u_true(x[:, None], t))))
    """

    # Case III: Wendroff for a(x)=1+0.1*V(x), u(x,0)=f(x), u(0,t)=g(t)    (f(0)=g(0))
    """
    vel = 1
    xmin, xmax = 0, 1
    tmin, tmax = 0, 1
    V = lambda x: np.ones_like(x) * vel
    f = lambda x: np.sin(2 * np.pi * x)
    g = lambda t: np.sin(2 * np.pi * (-(1 + 0.1 * vel) * t))

    u_true = lambda x, t: np.sin(2 * np.pi * (x - (1 + 0.1 * vel) * t))

    Nx, Nt = 100, 100
    x, t, u = solve_CVC(xmin, xmax, tmin, tmax, f, g, V, Nx, Nt)

    print(np.max(abs(u - u_true(x[:, None], t))))
    print(np.average(abs(u - u_true(x[:, None], t))))
    """

    # Case IV: Wendroff for a(x)=1+0.1*(V(x)+V(1-x))/2, u(x,0)=f(x)    (f,f' periodic)
    """
    vel = 1
    xmin, xmax = 0, 1
    tmin, tmax = 0, 1
    V = lambda x: np.ones_like(x) * vel
    f = lambda x: np.sin(2 * np.pi * x)
    g = lambda t: np.sin(2 * np.pi * (-(1 + 0.1 * vel) * t))

    u_true = lambda x, t: np.sin(2 * np.pi * (x - (1 + 0.1 * vel) * t))

    Nx, Nt = 100, 100
    x, t, u = solve_CVC(xmin, xmax, tmin, tmax, f, g, V, Nx, Nt)

    print(np.max(abs(u - u_true(x[:, None], t))))
    print(np.average(abs(u - u_true(x[:, None], t))))
    """

    # plot
    u_true = u_true(x[:, None], t)
    error = abs(u - u_true)
    axis = plt.subplot(111)
    plt.imshow(error, cmap="rainbow", vmin=0)
    plt.colorbar()
    xlabel = [format(i, ".1f") for i in np.linspace(0, 1, num=11)]
    ylabel = [format(i, ".1f") for i in np.linspace(0, 1, num=11)]
    axis.set_xticks(range(0, 101, 10))
    axis.set_xticklabels(xlabel)
    axis.set_yticks(range(0, 101, 10))
    axis.set_yticklabels(ylabel)
    axis.set_xlabel("t")
    axis.set_ylabel("x")
    axis.set_title(r"Error", fontdict={"fontsize": 30}, loc="left")

    return error


if __name__ == "__main__":
    error = main()
