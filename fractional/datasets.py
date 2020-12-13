import numpy as np


training_Lu = np.float32(np.loadtxt("training_Lu.txt"))
training_u = np.float32(np.loadtxt("training_u.txt"))
training_y = np.float32(np.loadtxt("training_y.txt"))
training_alpha = np.float32(np.loadtxt("training_alpha.txt"))

test_Lu = np.float32(np.loadtxt("test_Lu.txt"))
test_u = np.float32(np.loadtxt("test_u.txt"))
test_y = np.float32(np.loadtxt("test_y.txt"))
test_alpha = np.float32(np.loadtxt("test_alpha.txt"))

test_Lu0 = np.float32(np.loadtxt("test_Lu0.txt"))
test_u0 = np.float32(np.loadtxt("test_u0.txt"))
test_y0 = np.float32(np.loadtxt("test_y0.txt"))
test_alpha0 = np.float32(np.loadtxt("test_alpha0.txt"))

if len(training_y.shape) == 1:
    training_y = training_y.reshape((-1, 1))
if len(test_y.shape) == 1:
    test_y = test_y.reshape((-1, 1))
if len(training_alpha.shape) == 1:
    training_alpha = training_alpha.reshape((-1, 1))
if len(test_alpha.shape) == 1:
    test_alpha = test_alpha.reshape((-1, 1))
if len(test_y0.shape) == 1:
    test_y0 = test_y0.reshape((-1, 1))
if len(test_alpha0.shape) == 1:
    test_alpha0 = test_alpha0.reshape((-1, 1))
if len(test_u0.shape) == 1:
    test_u0 = test_u0.reshape((-1, 1))
if len(test_alpha0.shape) == 0:
    test_alpha0 = np.array([[test_alpha0]], dtype=np.float32)
if len(test_Lu0.shape) == 1:
    test_Lu0 = test_Lu0.reshape((-1, 1))
# test_frac_Lap = np.float32(np.loadtxt('test_frac_Lap0.txt'))
# test_u = np.float32(np.loadtxt('test_u0.txt').reshape((-1,1)))
# print(test_u.shape)
# test_r_y = np.float32(np.loadtxt('test_r_y0.txt')).T
# test_t_y = np.float32(np.loadtxt('test_t_y0.txt')).T
# test_alpha = np.float32(np.loadtxt('test_alpha0.txt'))

# training_y = np.concatenate((training_r_y.reshape((-1,1)), training_t_y.reshape((-1,1))),axis=1)
# test_y = np.concatenate((test_r_y.reshape((-1,1)), test_t_y.reshape((-1,1))),axis=1)

N_X = training_u.shape[0]
N_U = training_u.shape[1]
N_Y = training_y.shape[0]
N_A = training_alpha.shape[0]

n_x = test_u.shape[0]
n_u = test_u.shape[1]
n_y = test_y.shape[0]
n_a = test_alpha.shape[0]
# print(n_a, n_u)
n_x0 = test_u0.shape[0]
n_u0 = test_u0.shape[1]
n_y0 = test_y0.shape[0]
n_a0 = test_alpha0.shape[0]

d = training_y.shape[1] + training_alpha.shape[1]

counter = 0
X_u_train = np.zeros((N_U * N_A * N_Y, N_X), dtype=np.float32)
X_y_train = np.zeros((N_U * N_A * N_Y, d), dtype=np.float32)
Y_train = np.zeros((N_U * N_A * N_Y, 1), dtype=np.float32)

for i in range(N_A):
    for j in range(N_U):
        u_sample = training_u[:, j : (j + 1)].T
        alpha_sample = np.array([[training_alpha[i, 0]]], dtype=np.float32)
        U = np.tile(u_sample, (N_Y, 1))
        A = np.tile(alpha_sample, (N_Y, 1))
        index = i * N_U + j
        X_u_train[(counter * N_Y) : ((counter + 1) * N_Y), :] = U
        X_y_train[(counter * N_Y) : ((counter + 1) * N_Y), :] = np.concatenate(
            (training_y, A), axis=1
        )
        Y_train[(counter * N_Y) : ((counter + 1) * N_Y), 0:1] = training_Lu[
            :, index : (index + 1)
        ]
        counter = counter + 1

counter = 0
X_u_test = np.zeros((n_u * n_a * n_y, n_x), dtype=np.float32)
X_y_test = np.zeros((n_u * n_a * n_y, d), dtype=np.float32)
Y_test = np.zeros((n_u * n_a * n_y, 1), dtype=np.float32)

for i in range(n_a):
    for j in range(n_u):
        u_sample = test_u[:, j : (j + 1)].T
        alpha_sample = np.array([[test_alpha[i, 0]]], dtype=np.float32)
        U = np.tile(u_sample, (n_y, 1))
        A = np.tile(alpha_sample, (n_y, 1))
        index = i * n_u + j
        X_u_test[(counter * n_y) : ((counter + 1) * n_y), :] = U
        X_y_test[(counter * n_y) : ((counter + 1) * n_y), :] = np.concatenate(
            (test_y, A), axis=1
        )
        Y_test[(counter * n_y) : ((counter + 1) * n_y), 0:1] = test_Lu[
            :, index : (index + 1)
        ]
        counter = counter + 1

counter = 0
X_u_test0 = np.zeros((n_u0 * n_a0 * n_y0, n_x), dtype=np.float32)
X_y_test0 = np.zeros((n_u0 * n_a0 * n_y0, d), dtype=np.float32)
Y_test0 = np.zeros((n_u0 * n_a0 * n_y0, 1), dtype=np.float32)

for i in range(n_a0):
    for j in range(n_u0):
        u_sample0 = test_u0[:, j : (j + 1)].T
        alpha_sample0 = np.array([[test_alpha0[i, 0]]], dtype=np.float32)
        U0 = np.tile(u_sample0, (n_y0, 1))
        A0 = np.tile(alpha_sample0, (n_y0, 1))
        index = i * n_u0 + j
        X_u_test0[(counter * n_y0) : ((counter + 1) * n_y0), :] = U0
        X_y_test0[(counter * n_y0) : ((counter + 1) * n_y0), :] = np.concatenate(
            (test_y0, A0), axis=1
        )
        Y_test0[(counter * n_y0) : ((counter + 1) * n_y0), 0:1] = test_Lu0[
            :, index : (index + 1)
        ]
        counter = counter + 1

# X_u_train = np.loadtxt('X_u_train.txt')
# X_y_train = np.loadtxt('X_y_train.txt')
# Y_train = np.loadtxt('Y_train.txt')
data_path = "data/"
np.savez_compressed(
    data_path + "train.npz", X_u_train=X_u_train, X_y_train=X_y_train, Y_train=Y_train
)
np.savez_compressed(
    data_path + "test.npz", X_u_test=X_u_test, X_y_test=X_y_test, Y_test=Y_test
)
np.savez_compressed(
    data_path + "test0.npz", X_u_test=X_u_test0, X_y_test=X_y_test0, Y_test=Y_test0
)
