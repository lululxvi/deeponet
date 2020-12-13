from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
from pathos.pools import ProcessPool
from scipy import interpolate
from scipy.integrate import solve_ivp
from scipy.special import legendre

import config
from ADR_solver import solve_ADR
from ADVD_solver import solve_ADVD
from CVC_solver import solve_CVC
from utils import timing


class LTSystem(object):
    def __init__(self, npoints_output):
        """Legendre transform J_n{f(x)}.

        Args:
            npoints_output: For a input function, choose n=0,1,2,...,`npoints_output`-1 as data.
        """
        self.npoints_output = npoints_output

    @timing
    def gen_operator_data(self, space, m, num):
        """For each input function, generate `npoints_output` data, so the total number N = num x npoints_output.
        """
        print("Generating operator data...", flush=True)
        features = space.random(num)
        sensors = np.linspace(0, 2, num=m)[:, None]
        sensor_values = space.eval_u(features, sensors)

        sensor_values_tile = np.tile(sensor_values, (1, self.npoints_output)).reshape(
            [-1, m]
        )
        ns = np.tile(np.arange(self.npoints_output)[:, None], (num, 1))
        s_values = np.vstack(list(map(self.eval_s, sensor_values)))
        return [sensor_values_tile, ns], s_values

    def eval_s(self, sensor_value):
        """Compute J_n{f(x)} for a `sensor_value` of `f` with n=0,1,...,'npoints_output'-1.
        """
        x = np.linspace(-1, 1, num=10000)
        samplings = interpolate.interp1d(
            np.linspace(-1, 1, len(sensor_value)), sensor_value, kind="cubic"
        )(x)
        ns = np.arange(self.npoints_output)
        ys = np.vstack(list(map(lambda n: legendre(n)(x), ns)))

        return np.sum((samplings * ys)[:, 1:], axis=1, keepdims=True) * (x[1] - x[0])


class ODESystem(object):
    def __init__(self, g, s0, T):
        self.g = g
        self.s0 = s0
        self.T = T

    @timing
    def gen_operator_data(self, space, m, num):
        print("Generating operator data...", flush=True)
        features = space.random(num)
        sensors = np.linspace(0, self.T, num=m)[:, None]
        sensor_values = space.eval_u(features, sensors)
        x = self.T * np.random.rand(num)[:, None]
        y = self.eval_s_space(space, features, x)
        return [sensor_values, x], y

    def eval_s_space(self, space, features, x):
        """For a list of functions in `space` represented by `features`
        and a list `x`, compute the corresponding list of outputs.
        """

        def f(feature, xi):
            return self.eval_s(lambda t: space.eval_u_one(feature, t), xi[0])

        p = ProcessPool(nodes=config.processes)
        res = p.map(f, features, x)
        return np.array(list(res))

    def eval_s_func(self, u, x):
        """For an input function `u` and a list `x`, compute the corresponding list of outputs.
        """
        res = map(lambda xi: self.eval_s(u, xi[0]), x)
        return np.array(list(res))

    def eval_s(self, u, tf):
        """Compute `s`(`tf`) for an input function `u`.
        """

        def f(t, y):
            return self.g(y, u(t), t)

        sol = solve_ivp(f, [0, tf], self.s0, method="RK45")
        return sol.y[0, -1:]


class DRSystem(object):
    def __init__(self, D, k, T, Nt, npoints_output):
        """Diffusion-reaction on the domain [0, 1] x [0, T].

        Args:
            T: Time [0, T].
            Nt: Nt in FDM
            npoints_output: For a input function, randomly choose these many points from the solver output as data
        """
        self.D = D
        self.k = k
        self.T = T
        self.Nt = Nt
        self.npoints_output = npoints_output

    @timing
    def gen_operator_data(self, space, m, num):
        """For each input function, generate `npoints_output` data, so the total number N = num x npoints_output.
        """
        print("Generating operator data...", flush=True)
        features = space.random(num)
        sensors = np.linspace(0, 1, num=m)[:, None]
        sensor_values = space.eval_u(features, sensors)
        # p = ProcessPool(nodes=config.processes)
        # s_values = p.map(self.eval_s, sensor_values)
        s_values = map(self.eval_s, sensor_values)
        res = np.vstack(list(map(self.eval_s_sampling, sensor_values, s_values)))
        return [res[:, :m], res[:, m:-1]], res[:, -1:]

    def eval_s_sampling(self, sensor_value, s):
        """Given a `sensor_value` of `u` and the corresponding solution `s`, generate the 
        sampling outputs.
        """
        m = sensor_value.shape[0]
        x = np.random.randint(m, size=self.npoints_output)
        t = np.random.randint(self.Nt, size=self.npoints_output)
        xt = np.hstack([x[:, None], t[:, None]]) * [1 / (m - 1), self.T / (self.Nt - 1)]
        y = s[x][range(self.npoints_output), t][:, None]
        return np.hstack([np.tile(sensor_value, (self.npoints_output, 1)), xt, y])

    def eval_s(self, sensor_value):
        """Compute s(x, t) over m * Nt points for a `sensor_value` of `u`.
        """
        return solve_ADR(
            0,
            1,
            0,
            self.T,
            lambda x: self.D * np.ones_like(x),
            lambda x: np.zeros_like(x),
            lambda u: self.k * u ** 2,
            lambda u: 2 * self.k * u,
            lambda x, t: np.tile(sensor_value[:, None], (1, len(t))),
            lambda x: np.zeros_like(x),
            len(sensor_value),
            self.Nt,
        )[2]


class CVCSystem(object):
    def __init__(self, f, g, T, Nt, npoints_output):
        """Advection on the domain [0, 1] x [0, T].

        Args:
            T: Time [0, T].
            Nt: Nt in FDM
            npoints_output: For a input function, randomly choose these many points from the solver output as data
        """
        self.f = f
        self.g = g
        self.T = T
        self.Nt = Nt
        self.npoints_output = npoints_output

    @timing
    def gen_operator_data(self, space, m, num):
        """For each input function, generate `npoints_output` data, so the total number N = num x npoints_output.
        """
        print("Generating operator data...", flush=True)
        features = space.random(num)
        sensors = np.linspace(0, 1, num=m)[:, None]
        # Case I Input: V(sin^2(pi*x))
        sensor_values = space.eval_u(features, np.sin(np.pi * sensors) ** 2)
        # Case II Input: x*V(x)
        # sensor_values = sensors.T * space.eval_u(features, sensors)
        # Case III/IV Input: V(x)
        # sensor_values = space.eval_u(features, sensors)
        # p = ProcessPool(nodes=config.processes)
        # s_values = np.array(p.map(self.eval_s, sensor_values))
        s_values = np.array(list(map(self.eval_s, sensor_values)))
        res = np.vstack(list(map(self.eval_s_sampling, sensor_values, s_values)))
        return [res[:, :m], res[:, m:-1]], res[:, -1:]

    def eval_s_sampling(self, sensor_value, s):
        """Given a `sensor_value` of `u` and the corresponding solution `s`, generate the 
        sampling outputs.
        """
        m = sensor_value.shape[0]
        x = np.random.randint(m, size=self.npoints_output)
        t = np.random.randint(self.Nt, size=self.npoints_output)
        xt = np.hstack([x[:, None], t[:, None]]) * [1 / (m - 1), self.T / (self.Nt - 1)]
        y = s[x][range(self.npoints_output), t][:, None]
        return np.hstack([np.tile(sensor_value, (self.npoints_output, 1)), xt, y])

    def eval_s(self, sensor_value):
        """Compute s(x, t) over m * Nt points for a `sensor_value` of `u`.
        """
        # Case I: Analytical solution for a(x)=1, u(x,0)=V(x)    (V,V' periodic)
        return solve_CVC(
            0,
            1,
            0,
            self.T,
            self.f,
            self.g,
            interpolate.interp1d(
                np.linspace(0, 1, len(sensor_value)), sensor_value, kind="cubic"
            ),
            len(sensor_value),
            self.Nt,
        )[2]
        # Case II: Wendroff for a(x)=1, u(x,0)=f(x), u(0,t)=g(t)    (f(0)=g(0))
        """
        return solve_CVC(
            0,
            1,
            0,
            self.T,
            lambda x: sensor_value,
            lambda t: np.zeros_like(t),
            None,
            len(sensor_value),
            self.Nt,
        )[2]
        """
        # Case III: Wendroff for a(x)=1+0.1*V(x), u(x,0)=f(x), u(0,t)=g(t)    (f(0)=g(0))
        """
        return solve_CVC(
            0,
            1,
            0,
            self.T,
            lambda x: x ** 2,
            lambda t: np.sin(np.pi * t),
            lambda x: sensor_value,
            len(sensor_value),
            self.Nt,
        )[2]
        """
        # Case IV: Wendroff for a(x)=1+0.1*(V(x)+V(1-x))/2, u(x,0)=f(x)    (f,f' periodic)
        """
        return solve_CVC(
            0,
            1,
            0,
            self.T,
            lambda x: np.sin(2 * np.pi * x),
            None,
            lambda x: sensor_value,
            len(sensor_value),
            self.Nt,
        )[2]
        """


class ADVDSystem(object):
    def __init__(self, f, g, T, Nt, npoints_output):
        """Advection-diffusion on the domain [0, 1] x [0, T].

        Args:
            T: Time [0, T].
            Nt: Nt in FDM
            npoints_output: For a input function, randomly choose these many points from the solver output as data
        """
        self.f = f
        self.g = g
        self.T = T
        self.Nt = Nt
        self.npoints_output = npoints_output

    @timing
    def gen_operator_data(self, space, m, num):
        """For each input function, generate `npoints_output` data, so the total number N = num x npoints_output.
        """
        print("Generating operator data...", flush=True)
        features = space.random(num)
        sensors = np.linspace(0, 1, num=m)[:, None]
        # Input: V(sin^2(pi*x))
        sensor_values = space.eval_u(features, np.sin(np.pi * sensors) ** 2)
        # p = ProcessPool(nodes=config.processes)
        # s_values = np.array(p.map(self.eval_s, sensor_values))
        s_values = np.array(list(map(self.eval_s, sensor_values)))
        res = np.vstack(list(map(self.eval_s_sampling, sensor_values, s_values)))
        return [res[:, :m], res[:, m:-1]], res[:, -1:]

    def eval_s_sampling(self, sensor_value, s):
        """Given a `sensor_value` of `u` and the corresponding solution `s`, generate the 
        sampling outputs.
        """
        m = sensor_value.shape[0]
        x = np.random.randint(m, size=self.npoints_output)
        t = np.random.randint(self.Nt, size=self.npoints_output)
        xt = np.hstack([x[:, None], t[:, None]]) * [1 / (m - 1), self.T / (self.Nt - 1)]
        y = s[x][range(self.npoints_output), t][:, None]
        return np.hstack([np.tile(sensor_value, (self.npoints_output, 1)), xt, y])

    def eval_s(self, sensor_value):
        """Compute s(x, t) over m * Nt points for a `sensor_value` of `u`.
        """
        Nt_pc = (self.Nt - 1) * 10 + 1
        return solve_ADVD(
            0,
            1,
            0,
            self.T,
            self.f,
            self.g,
            lambda x: sensor_value,
            len(sensor_value),
            Nt_pc,
        )[2][:, 0:Nt_pc:10]
