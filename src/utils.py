from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import sys
import time
from functools import wraps

import numpy as np


def timing(f):
    """Decorator for measuring the execution time of methods."""

    @wraps(f)
    def wrapper(*args, **kwargs):
        ts = time.time()
        result = f(*args, **kwargs)
        te = time.time()
        print("%r took %f s\n" % (f.__name__, te - ts))
        sys.stdout.flush()
        return result

    return wrapper


def merge_values(values):
    return np.hstack(values) if isinstance(values, (list, tuple)) else values


def trim_to_65535(x):
    """Incorrect output when batch size > 65535.
    https://github.com/tensorflow/tensorflow/issues/9870
    https://github.com/tensorflow/tensorflow/issues/13869
    """
    N = 65535
    if isinstance(x, (list, tuple)):
        return (x[0][:N], x[1][:N]), (x[0][N:], x[1][N:])
    return x[:N], x[N:]


def mean_squared_error_outlier(y_true, y_pred):
    error = np.ravel((y_true - y_pred) ** 2)
    error = np.sort(error)[: -len(error) // 1000]
    return np.mean(error)


def safe_test(model, data, X_test, y_test, fname=None):
    def is_nonempty(X):
        return len(X[0]) > 0 if isinstance(X, (list, tuple)) else len(X) > 0

    y_pred = []
    X = X_test
    while is_nonempty(X):
        X_add, X = trim_to_65535(X)
        y_pred.append(model.predict(data.transform_inputs(X_add)))
    y_pred = np.vstack(y_pred)
    error = np.mean((y_test - y_pred) ** 2)
    print("Test MSE: {}".format(error))
    error = mean_squared_error_outlier(y_test, y_pred)
    print("Test MSE w/o outliers: {}\n".format(error))

    if fname is not None:
        np.savetxt(fname, np.hstack((X_test[1], y_test, y_pred)))


def eig(kernel, num, Nx, eigenfunction=True):
    """Compute the eigenvalues and eigenfunctions of a kernel on [0, 1].
    """
    h = 1 / (Nx - 1)
    c = kernel(np.linspace(0, 1, num=Nx)[:, None])[0] * h
    A = np.empty((Nx, Nx))
    for i in range(Nx):
        A[i, i:] = c[: Nx - i]
        A[i, i::-1] = c[: i + 1]
    A[:, 0] *= 0.5
    A[:, -1] *= 0.5

    if not eigenfunction:
        return np.flipud(np.sort(np.real(np.linalg.eigvals(A))))[:num]

    eigval, eigvec = np.linalg.eig(A)
    eigval, eigvec = np.real(eigval), np.real(eigvec)
    idx = np.flipud(np.argsort(eigval))[:num]
    eigval, eigvec = eigval[idx], eigvec[:, idx]
    for i in range(num):
        eigvec[:, i] /= np.trapz(eigvec[:, i] ** 2, dx=h) ** 0.5
    return eigval, eigvec


def trapz(y, dx):
    """Integrate [y(x1), y(x2), ...] or [[y1(x1), y1(x2), ...], [y2(x1), y2(x2), ...], ...]
    using the composite trapezoidal rule.

    Return: [I1(x1)=0, I1(x2), ...] or [[I1(x1)=0, I1(x2), ...], [I2(x1)=0, I2(x2), ...], ...]
    """
    if len(y.shape) == 1:
        left = np.cumsum(y)[:-1]
        right = np.cumsum(y[1:])
        return np.hstack(([0], (left + right) / 2 * dx))
    left = np.cumsum(y, axis=1)[:, :-1]
    right = np.cumsum(y[:, 1:], axis=1)
    return np.hstack((np.zeros((len(y), 1)), (left + right) / 2 * dx))


def make_triple(sensor_value, x, y, num):
    """For a `sensor_value` of `u`, a list of locations `x` and the corresponding solution `y`,
    generate a dataset of `num` triples.

    sensor_value: 1d array
    x: 2d array, N x d
    y: 1d array
    """
    idx = np.random.choice(len(x), size=num, replace=False)
    x = x[idx]
    y = y[idx][:, None]
    return np.hstack([np.tile(sensor_value, (num, 1)), x, y])
