import tensorflow as tf
import matplotlib.pyplot as plt
import scipy.special as scisp
import numpy as np
from SALib.sample import sobol_sequence
import time
import sys

# import datasets as ds

random_seed = 12345


def xavier_init(size):
    in_dim = size[0]
    out_dim = size[1]
    xavier_stddev = np.sqrt(2.0 / (in_dim + out_dim))
    return tf.Variable(
        tf.truncated_normal(
            [in_dim, out_dim], stddev=xavier_stddev, seed=random_seed, dtype=tf.float32
        ),
        dtype=tf.float32,
    )


# def neural_net(X, weights, biases):
#    num_layers = len(weights) + 1
#    H = X
#    for l in range(0,num_layers-1):
#        W = weights[l]
#        b = biases[l]
#        H = tf.nn.tanh(tf.add(tf.matmul(H, W), b))
#    W = weights[-1]
#    b = biases[-1]
#    Y = tf.add(tf.matmul(H, W), b)
#    Y = H
#    return Y


def neural_net2(X, weights, biases):
    num_layers = len(weights) + 1
    H = X
    for l in range(0, num_layers - 1):
        W = weights[l]
        b = biases[l]
        H = tf.nn.tanh(tf.add(tf.matmul(H, W), b))

    Y = H
    return Y


def neural_net1(X, weights, biases):
    num_layers = len(weights) + 1
    H = X
    for l in range(0, num_layers - 2):
        W = weights[l]
        b = biases[l]
        H = tf.nn.tanh(tf.add(tf.matmul(H, W), b))
    W = weights[-1]
    b = biases[-1]
    Y = tf.add(tf.matmul(H, W), b)
    return Y


################  Specify parameters and hyperparameters
### learning 1D Caputo derivative
m = 15  # length of u vector
d = 2  # dim of (y,alpha)

### learning 2D fractional Laplacian
# m = 225  # length of u vector
# d = 3 # dim of (x,y,alpha)

batch_size = 100000
num_epoch = 1000001
print_skip = 100
is_test = False
# is_test = True

### 1D Caputo
layers_u = [m] + [40] * 3
layers_y = [d] + [40] * 3

### 2D fractional Laplacian
# layers_u = [m] + [60]*3
# layers_y = [d] + [60]*3

store_path = "./saved_model/"

################################# buidling ONet

L_u = len(layers_u)
L_y = len(layers_y)

b0 = tf.Variable(0.0, name="b0", dtype=tf.float32)

weights_u = [
    tf.Variable(
        xavier_init([layers_u[l], layers_u[l + 1]]),
        name="weights_u" + str(l),
        dtype=tf.float32,
    )
    for l in range(0, L_u - 1)
]
biases_u = [
    tf.Variable(
        tf.zeros((1, layers_u[l + 1]), dtype=tf.float32, name="biases_u" + str(l)),
        dtype=tf.float32,
    )
    for l in range(0, L_u - 1)
]

weights_y = [
    tf.Variable(
        xavier_init([layers_y[l], layers_y[l + 1]]),
        name="weights_y" + str(l),
        dtype=tf.float32,
    )
    for l in range(0, L_y - 1)
]
biases_y = [
    tf.Variable(
        tf.zeros((1, layers_y[l + 1]), dtype=tf.float32, name="biases_y" + str(l)),
        dtype=tf.float32,
    )
    for l in range(0, L_y - 1)
]

x_u = tf.placeholder(tf.float32, shape=(None, m))
x_y = tf.placeholder(tf.float32, shape=(None, d))
y = tf.placeholder(tf.float32, shape=(None, 1))

net_u = neural_net1(x_u, weights_u, biases_u)
net_y = neural_net2(x_y, weights_y, biases_y)

net_o = tf.reduce_sum(net_u * net_y, axis=1, keepdims=True) + b0

saver = tf.train.Saver(
    var_list=[weights_u[l] for l in range(L_u - 1)]
    + [biases_u[l] for l in range(L_u - 1)]
    + [weights_y[l] for l in range(L_y - 1)]
    + [biases_y[l] for l in range(L_y - 1)]
    + [b0]
)

############ defining loss and optimizer

loss = tf.reduce_mean(tf.square(net_o - y)) / tf.reduce_mean(tf.square(y))
optimizer_Adam = tf.train.AdamOptimizer(1.0e-3)
# tt0 = time.time()
train_op_Adam = optimizer_Adam.minimize(loss)
# tt1 = time.time()
# print ('loss_graph CPU time: ', tt1-tt0)

############  generating and loading training, validation, and test sets

# if is_test == False:
#    tt0 = time.time()
#    ds.training_set(m, d, n_u, n_y)
#    tt1 = time.time()
#    print ('Generate training set CPU time: ', tt1-tt0)
#
# ds.test_set(m, d, n_y)
data_path = "data/"

data = np.load(data_path + "train.npz")
X_u_train, X_y_train, Y_train = data["X_u_train"], data["X_y_train"], data["Y_train"]

data = np.load(data_path + "test.npz")
X_u_test, X_y_test, Y_test = data["X_u_test"], data["X_y_test"], data["Y_test"]

data = np.load(data_path + "test0.npz")
X_u_test0, X_y_test0, Y_test0 = data["X_u_test"], data["X_y_test"], data["Y_test"]

# data = np.load("test_fabricated.npz")
# X_u_test, X_y_test, Y_test =  data["X_u_test"], data["X_y_test"], data["Y_test"]

# X_u_train = (X_u_train0 - np.mean(X_u_train0,axis=0,keepdims=True))/np.std(X_u_train0,axis=0, keepdims=True)
# X_y_train = (X_y_train0 - np.mean(X_y_train0,axis=0,keepdims=True))/np.std(X_y_train0,axis=0, keepdims=True)
#
#
# X_u_test = (X_u_test0- np.mean(X_u_train0,axis=0,keepdims=True))/np.std(X_u_train0,axis=0, keepdims=True)
# X_y_test = (X_y_test0 - np.mean(X_y_train0,axis=0,keepdims=True))/np.std(X_y_train0,axis=0, keepdims=True)

################## Training, validating or test
loss_train_h = []
loss_test_h = []
loss_test0_h = []

i_h = []

if is_test == False:
    tt0 = time.time()
    min_loss = 1e16
    num_batch = X_u_train.shape[0] // batch_size

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        #        feed_train = {x_u: X_u_train, x_y: X_y_train, y: Y_train}
        feed_test = {x_u: X_u_test, x_y: X_y_test, y: Y_test}
        feed_test0 = {x_u: X_u_test0, x_y: X_y_test0, y: Y_test0}

        ind = np.arange(X_u_train.shape[0])
        for i in range(num_epoch):
            np.random.shuffle(ind)
            for j in range(num_batch):
                feed_train_batch = {
                    x_u: X_u_train[ind[(j * batch_size) : ((j + 1) * batch_size)], 0:],
                    x_y: X_y_train[ind[(j * batch_size) : ((j + 1) * batch_size)], 0:],
                    y: Y_train[ind[(j * batch_size) : ((j + 1) * batch_size)], 0:],
                }
                if i % print_skip == 0 and j == num_batch - 1:
                    temp_loss = sess.run(loss, feed_train_batch)

                    if temp_loss < min_loss:
                        save_path = saver.save(sess, store_path + "paras_NN.ckpt")
                        min_loss = temp_loss
                        loss_train = temp_loss
                        loss_test, Y_pred = sess.run([loss, net_o], feed_test)
                        loss_test0, Y_pred0 = sess.run([loss, net_o], feed_test0)
                        error = np.linalg.norm(Y_pred - Y_test) / np.linalg.norm(Y_test)
                        error0 = np.linalg.norm(Y_pred0 - Y_test0) / np.linalg.norm(
                            Y_test0
                        )

                        loss_train_h.append(loss_train)
                        loss_test_h.append(loss_test)
                        loss_test0_h.append(loss_test0)

                        i_h.append(np.float64(i))

                        fig = plt.figure()
                        losst = np.stack(loss_train_h)
                        lossv = np.stack(loss_test_h)
                        lossv0 = np.stack(loss_test0_h)

                        ii = np.stack(i_h)
                        plt.semilogy(ii, losst, "r", label="Training loss")
                        plt.semilogy(ii, lossv, "b", label="Test loss")
                        plt.semilogy(ii, lossv0, "b", label="Test loss0")

                        plt.xlabel("Number of epochs")
                        plt.ylabel("Loss")
                        plt.title("Training and test")
                        plt.legend()
                        plt.savefig(store_path + "Training_test0.png", dpi=300)
                        plt.tight_layout()
                        plt.close(fig)

                        fig = plt.figure()
                        losst = np.stack(loss_train_h)
                        lossv = np.stack(loss_test_h)
                        lossv0 = np.stack(loss_test0_h)

                        ii = np.stack(i_h)
                        plt.semilogy(ii, losst, "r", label="Training loss")
                        plt.semilogy(ii, lossv, "b", label="Test loss")

                        plt.xlabel("Number of epochs")
                        plt.ylabel("Loss")
                        plt.title("Training and test")
                        plt.legend()
                        plt.savefig(store_path + "Training_test.png", dpi=300)
                        plt.tight_layout()
                        plt.close(fig)

                        with open(store_path + "training_validation.txt", "a") as f:
                            f.write(
                                "Epoch: "
                                + str(i + 1)
                                + " Training loss: "
                                + str(loss_train)
                                + "  Test loss: "
                                + str(loss_test)
                                + "  Test loss0: "
                                + str(loss_test0)
                                + " RelErr: "
                                + str(error)
                                + "\n\n"
                            )
                        print(
                            "\n",
                            "Epoch: ",
                            i + 1,
                            "Training loss: ",
                            loss_train,
                            "Test loss: ",
                            loss_test,
                            "Test loss0: ",
                            loss_test0,
                            "Rel_Err: ",
                            error,
                        )

                        np.savetxt(store_path + "loss_train.txt", losst)
                        np.savetxt(store_path + "loss_test.txt", lossv)
                        np.savetxt(store_path + "loss-test0.txt", lossv0)
                        np.savetxt(store_path + "ii.txt", ii)

                sess.run(train_op_Adam, feed_train_batch)

    tt1 = time.time()

    print("Training and validation CPU time: ", tt1 - tt0)

else:
    tt0 = time.time()

    with tf.Session() as sess:
        saver.restore(sess, store_path + "paras_NN.ckpt")
        feed_test = {x_u: X_u_test, x_y: X_y_test, y: Y_test}
        feed_test0 = {x_u: X_u_test0, x_y: X_y_test0, y: Y_test0}

        #       feed_train = {x_u: X_u_train, x_y: X_y_train, y: Y_train}
        feed_valid = {x_u: X_u_test, x_y: X_y_test, y: Y_test}
        #       train_loss = sess.run(loss, feed_train)
        valid_loss = sess.run(loss, feed_valid)
        test_loss, Y_pred = sess.run([loss, net_o], feed_test)
        test_loss0, Y_pred0 = sess.run([loss, net_o], feed_test0)

        test_err = np.linalg.norm(Y_pred - Y_test) / np.linalg.norm(Y_test)
        test_err0 = np.linalg.norm(Y_pred0 - Y_test0) / np.linalg.norm(Y_test0)

        with open(store_path + "test.txt", "a") as f:
            f.write(
                "  Validation loss: "
                + str(valid_loss)
                + " Test loss: "
                + str(test_loss)
                + " Test loss0: "
                + str(test_loss0)
                + " RelErr: "
                + str(test_err)
                + "\n\n"
            )

        print(
            "Valid_loss: ",
            valid_loss,
            "Test_loss: ",
            test_loss,
            "test rel_Err: ",
            test_err,
            "Test_loss0: ",
            test_loss0,
            "test rel_Err0: ",
            test_err0,
        )

        #       np.savetxt('Y_pred.txt', Y_pred)
        fig = plt.figure()
        plt.plot(Y_pred, Y_test, "r.", Y_test, Y_test, "b:")
        plt.savefig(store_path + "prediction.png", dpi=300)
        plt.close(fig)
        # rr = X_y_test[:100,0].reshape((10,10))
        # tt = X_y_test[:100,1].reshape((10,10))

        # fig = plt.figure()
        # plt.subplot(121)
        # plt.contourf(rr*np.cos(tt), rr*np.sin(tt), Y_pred[:100].reshape(rr.shape),100,cmap='jet')
        # plt.colorbar()
        # plt.subplot(122)
        # plt.contourf(rr*np.cos(tt), rr*np.sin(tt), Y_test[:100].reshape(rr.shape),100,cmap='jet')
        # plt.colorbar()
        # plt.title(r'$\alpha= $'+str(X_y_test[0,-1]))
        # plt.tight_layout()
        # plt.savefig(store_path+'prediction1_fabricated.png',dpi=300)
        # plt.close(fig)

    #       fig = plt.figure()
    #       plt.plot(X_y_test[0:9,0:1].flatten(), Y_pred[0:9,0:1].flatten(),'r',label='pred:  '+r'$G\{u\}(y,0.01)$')
    #       plt.plot(X_y_test[0:9,0:1].flatten(), Y_test[0:9,0:1].flatten(),'b',label='test:   '+r'$\frac{d^{0.01}u}{dy^0.01}(y)$')
    #       plt.title('Prediction ' +r' $G\{u\}(y,\alpha=0.01)\approx \frac{d^{0.01}u}{dy^0.01}(y)$')
    #       plt.xlabel('y')
    #       plt.ylabel(r'$G\{u\}(y,\alpha)$')
    #       plt.tight_layout()
    #       plt.legend()
    #       plt.savefig(store_path+'prediction1.png',dpi=500)
    #
    #       fig = plt.figure()
    #       plt.plot(X_y_test[81:,0:1].flatten(), Y_pred[81:,0:1].flatten(),'r',label='pred:  '+r'$G\{u\}(y,0.99)$')
    #       plt.plot(X_y_test[81:,0:1].flatten(), Y_test[81:,0:1].flatten(),'b',label='test:   '+r'$\frac{d^{0.99}u}{dy^0.99}(y)$')
    #       plt.title('Prediction ' +r' $G\{u\}(y,\alpha=0.99) \approx \frac{d^{0.99}u}{dy^0.99}(y)$')
    #       plt.xlabel('y')
    #       plt.ylabel(r'$G\{u\}(y,\alpha)$')
    #       plt.tight_layout()
    #       plt.legend()
    #       plt.savefig(store_path+'prediction2.png',dpi=500)
    #
    #       plt.show()
    #       plt.close(fig)
    tt1 = time.time()
    print("Test CPU time: ", tt1 - tt0)
