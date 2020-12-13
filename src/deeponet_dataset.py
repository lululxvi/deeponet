from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np

import deepxde as dde
from utils import mean_squared_error_outlier, safe_test, trim_to_65535


def run(m, net, lr, epochs):
    d = np.load("train.npz")
    X_train, y_train = (d["X_train0"], d["X_train1"]), d["y_train"]
    d = np.load("test.npz")
    X_test, y_test = (d["X_test0"], d["X_test1"]), d["y_test"]

    X_test_trim = trim_to_65535(X_test)[0]
    y_test_trim = trim_to_65535(y_test)[0]
    data = dde.data.OpDataSet(
        X_train=X_train, y_train=y_train, X_test=X_test_trim, y_test=y_test_trim
    )

    model = dde.Model(data, net)
    model.compile("adam", lr=lr, metrics=[mean_squared_error_outlier])
    checker = dde.callbacks.ModelCheckpoint(
        "model/model.ckpt", save_better_only=True, period=1000
    )
    losshistory, train_state = model.train(epochs=epochs, callbacks=[checker])
    dde.saveplot(losshistory, train_state, issave=False, isplot=True)

    model.restore("model/model.ckpt-" + str(train_state.best_step), verbose=1)
    safe_test(model, data, X_test, y_test)

    for i in range(10):
        d = np.load("example{}.npz".format(i))
        X_test, y_test = (d["X_test0"], d["X_test1"]), d["y_test"]
        safe_test(model, data, X_test, y_test, fname="example{}.dat".format(i))


def main():
    # Pathwise solution
    # m = 100
    # epochs = 50000
    # dim_x = 6
    # Statistical solution
    m = 240
    epochs = 20000
    dim_x = 1
    lr = 0.001
    net = dde.maps.OpNN(
        [m, 100, 100],
        [dim_x, 100, 100],
        "relu",
        "Glorot normal",
        use_bias=True,
        stacked=False,
    )

    run(m, net, lr, epochs)


if __name__ == "__main__":
    main()
