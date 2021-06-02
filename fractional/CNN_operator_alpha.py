### Variational autoencoder

import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np

# from keras.datasets import mnist


class CNN_model:
    def __init__(self):
        pass

    def conv2D_layer(
        self,
        in_features,
        out_channel,
        padding_size=0,
        filter_size=3,
        activation="tanh",
        output_paras=False,
    ):
        batch_size, height, width, channel = in_features.shape
        in_height = height.value
        in_width = width.value
        in_channel = channel.value
        if height != width:
            raise AssertionError("The input image must be square like")
        S = 1  # unit-stride
        out_height = (in_height - filter_size + 2 * padding_size) / S + 1
        in_dim = in_height ** 2 * in_channel
        xavier_stddev = np.sqrt(2.0 / (in_dim))
        weights = tf.Variable(
            tf.truncated_normal(
                [filter_size, filter_size, in_channel, out_channel],
                stddev=xavier_stddev,
                dtype=tf.float32,
                seed=None,
            ),
            dtype=tf.float32,
        )
        biases = tf.Variable(tf.zeros((1, out_channel), dtype=tf.float32))
        padded_features = tf.pad(
            in_features,
            [
                [0, 0],
                [padding_size, padding_size],
                [padding_size, padding_size],
                [0, 0],
            ],
            "CONSTANT",
        )
        feature_maps = tf.nn.conv2d(
            padded_features,
            weights,
            padding="VALID",
            strides=[1, 1, 1, 1],
            data_format="NHWC",
        )
        feature_maps = feature_maps + biases
        if activation == "relu":
            out_features = tf.nn.relu(feature_maps)
        elif activation == "sigmoid":
            out_features = tf.nn.sigmoid(feature_maps)
        elif activation == "identity":
            out_features = feature_maps
        elif activation == "tanh":
            out_features = tf.nn.tanh(feature_maps)

        if output_paras == True:
            return out_features, weights, biases
        else:
            return out_features

    def pooling2D(self, in_features, size):
        return tf.nn.max_pool(
            in_features,
            ksize=[1] + size + [1],
            strides=[1] + size + [1],
            padding="VALID",
        )

    def dense(self, in_features, out_channel, activation="tanh", output_paras=False):
        flattened_features = tf.layers.flatten(in_features)
        in_dim = flattened_features.shape[1]
        in_dim = in_dim.value
        out_dim = out_channel
        xavier_stddev = np.sqrt(2.0 / (in_dim + out_dim))
        weights = tf.Variable(
            tf.truncated_normal(
                [in_dim, out_dim], stddev=xavier_stddev, dtype=tf.float32, seed=None
            ),
            dtype=tf.float32,
        )
        biases = tf.Variable(tf.zeros((1, out_dim), dtype=tf.float32))
        feature_maps = tf.matmul(flattened_features, weights)
        feature_maps = feature_maps + biases
        if activation == "relu":
            out_features = tf.nn.relu(feature_maps)
        elif activation == "softmax":
            exp0 = tf.exp(-1.0 * feature_maps)
            out_features = exp0 / tf.reduce_sum(exp0, axis=1, keepdims=True)
        elif activation == "sigmoid":
            out_features = tf.nn.sigmoid(feature_maps)
        elif activation == "softplus":
            out_features = tf.nn.softplus(feature_maps)
        elif activation == "identity":
            out_features = feature_maps
        elif activation == "tanh":
            out_features = tf.nn.tanh(feature_maps)

        if output_paras == True:
            return out_features, weights, biases
        else:
            return out_features

    def cross_entropy(self, out_features, labels):
        return tf.reduce_sum(
            -1.0 * tf.reduce_sum(tf.log(out_features) * labels, axis=1)
        )

    def latent_space(self, in_features, latent_dim, batch_size):
        z_mean = self.dense(in_features, latent_dim, activation="identity")
        z_log_var = self.dense(
            in_features, latent_dim, activation="identity"
        )  # log std
        out_features = z_mean + tf.sqrt(tf.exp(z_log_var)) * tf.random_normal(
            (batch_size, latent_dim)
        )
        return out_features, z_mean, z_log_var

    def dense_reshape(self, in_features, shape):
        return tf.reshape(in_features, [-1] + shape)

    def upsampling(self, in_features, shape):
        return tf.image.resize_images(in_features, size=shape)

    def VAE_loss(self, in_features, batch_labels, z_mean, z_log_var):
        dim = in_features.shape
        dim = dim[1].value * dim[2].value * dim[3].value
        predicted = tf.reshape(in_features, (-1, dim))
        ground_truth = tf.reshape(batch_labels, (-1, dim))
        CE_loss = tf.reduce_mean(
            tf.reduce_sum(tf.square(predicted - ground_truth), axis=1)
        ) / tf.reduce_mean(tf.reduce_sum(tf.square(ground_truth), axis=1))
        epsilon = 1e-7
        # CE_loss = -1.0*tf.reduce_mean(tf.reduce_sum(tf.log(epsilon+predicted)*ground_truth+tf.log(1.0+epsilon-predicted)*(1-ground_truth),axis=1))
        REF_loss = -1.0 * tf.reduce_mean(
            tf.reduce_sum(
                tf.log(epsilon + ground_truth) * ground_truth
                + tf.log(1.0 + epsilon - ground_truth) * (1 - ground_truth),
                axis=1,
            )
        )
        KL_loss = -0.0 * tf.reduce_mean(
            tf.reduce_sum(1 + z_log_var - tf.square(z_mean) - tf.exp(z_log_var), axis=1)
        )
        # loss1 = -1.0* tf.reduce_sum(tf.log(epsilon+predicted)*ground_truth+tf.log(1.0+epsilon-predicted)*(1-ground_truth),axis=1)
        loss1 = CE_loss
        loss2 = KL_loss
        loss = loss1 + loss2
        relative_err = tf.linalg.norm(predicted - ground_truth, 2) / tf.linalg.norm(
            ground_truth, 2
        )
        return loss, CE_loss, KL_loss, REF_loss, relative_err


# (train_images, train_labels), (test_images, test_labels) = mnist.load_data()

# train_images = train_images.astype('float32')/255
# test_images = test_images.astype('float32')/255

# train_images = np.expand_dims(train_images, axis=3) # add channel axis
# test_images = np.expand_dims(test_images, axis=3)


# train_labels0 = np.zeros((train_images.shape[0],10),dtype=np.float32)
# train_labels0[np.arange(train_images.shape[0]),train_labels]=1.0

# test_labels0 = np.zeros((test_images.shape[0],10),dtype=np.float32)
# test_labels0[np.arange(test_images.shape[0]),test_labels]=1.0

image_size = 15
num_u = 10000

# images = np.loadtxt('training_u.txt',dtype = np.float32).T
# labels0 = np.loadtxt('training_Lu.txt', dtype=np.float32).T
# cart_co = np.loadtxt('training_y.txt', dtype=np.float32)
# alpha = np.loadtxt('training_alpha.txt', dtype=np.float32)


# images_test = np.loadtxt('test_u.txt',dtype = np.float32).T
# labels0_test = np.loadtxt('test_Lu.txt', dtype=np.float32).T
# cart_co_test = np.loadtxt('test_y.txt', dtype=np.float32)
# alpha_test = np.loadtxt('test_alpha.txt', dtype=np.float32)
train_data = np.load("train.npz")
test_data = np.load("test.npz")

images = train_data["images"]
labels0 = train_data["labels0"]
cart_co = train_data["cart_co"]
alpha = train_data["alpha"]

images_test = test_data["images"]
labels0_test = test_data["labels0"]
cart_co_test = test_data["cart_co"]
alpha_test = test_data["alpha"]

batch_images = tf.placeholder(tf.float32, shape=(None, image_size, image_size, 1))
batch_labels = tf.placeholder(
    tf.float32, shape=(None, image_size, image_size, alpha.shape[0])
)

x_grid = cart_co[:, 0].reshape((image_size, image_size)).T
y_grid = cart_co[:, 1].reshape((image_size, image_size)).T

x_grid_test = cart_co_test[:, 0].reshape((image_size, image_size)).T
y_grid_test = cart_co_test[:, 1].reshape((image_size, image_size)).T

labels = np.zeros((images.shape[0], image_size ** 2, alpha.shape[0]))
for i in range(alpha.shape[0]):
    labels[:, :, i] = labels0[(i * num_u) : ((i + 1) * num_u), 0:]

labels_test = np.zeros((images.shape[0], image_size ** 2, alpha_test.shape[0]))
for i in range(alpha_test.shape[0]):
    labels_test[:, :, i] = labels0_test[(i * num_u) : ((i + 1) * num_u), 0:]

np.random.seed(1234)
# index = np.arange(images.shape[0])
# np.random.shuffle(index)
# idd = int(0.7*images.shape[0])
#
# train_images = images[:idd,0:]
# train_labels = labels[:idd,0:,0:]
# test_images = images[idd:,0:]
# test_labels = labels[idd:,0:,0:]
train_images = images
train_labels = labels
test_images = images_test
test_labels = labels_test

# max_image = np.amax(train_images0)
# min_image = np.amin(train_images0)
# max_label = np.amax(train_labels0)
# min_label = np.amin(train_labels0)

# max_image = np.amax(images)
# min_image = np.amin(images)
# max_label = np.amax(labels)
# min_label = np.amin(labels)

# train_images = (train_images0 - min_image) / (max_image - min_image)
# train_labels = (train_labels0 - min_label) / (max_label - min_label)
# test_images = (test_images0 - min_image) / (max_image - min_image)
# test_labels = (test_labels0 - min_label) / (max_label - min_label)

train_images = train_images.reshape((-1, image_size, image_size, 1))
train_labels = train_labels.reshape((-1, image_size, image_size, alpha.shape[0]))
test_images = test_images.reshape((-1, image_size, image_size, 1))
test_labels = test_labels.reshape((-1, image_size, image_size, alpha_test.shape[0]))

# plt.imshow(train_images[1021,:].reshape((image_size,image_size)), cmap='jet')
# plt.colorbar()
# plt.show()
# plt.imshow(train_labels[1021,:].reshape((image_size,image_size)), cmap='jet')
# plt.colorbar()
# plt.show()

# aaa = 1

num_epoch = 10001
batch_size0 = 100
batch_size = tf.placeholder(tf.int32, shape=())
latent_dim = 20

x = CNN_model()

# feature = x.conv2D_layer(batch_images, 32)
# feature = x.pooling2D(feature, [2,2])
# feature = x.conv2D_layer(feature, 64)
# feature = x.pooling2D(feature, [2,2])
# feature = x.conv2D_layer(feature, 64)
# feature = x.dense(feature, 64)
# feature, z_mean, z_log_var = x.latent_space(feature, latent_dim, batch_size)
# feature0 = feature
# feature = x.dense(feature, 64)
# feature = x.dense(feature, 3*3*64)
# feature = x.dense_reshape(feature, [3,3,64])
# feature = x.conv2D_layer(feature, 64, padding_size=2)
# feature = x.upsampling(feature, shape=[11,11])
# feature = x.conv2D_layer(feature, 32, padding_size=2)
# feature = x.upsampling(feature, shape=[26,26])
# feature = x.conv2D_layer(feature, alpha.shape[0], padding_size=2, activation = 'identity')
# loss, loss1, loss2, ref_loss, predicted0, ground_truth0 = x.VAE_loss(feature, batch_labels, z_mean, z_log_var)
feature = x.conv2D_layer(batch_images, 32)
feature = x.pooling2D(feature, [2, 2])
feature = x.conv2D_layer(feature, 64)
feature = x.dense(feature, 64)
feature, z_mean, z_log_var = x.latent_space(feature, latent_dim, batch_size)
feature0 = feature
feature = x.dense(feature, 64)
feature = x.dense(feature, 4 * 4 * 64)
feature = x.dense_reshape(feature, [4, 4, 64])
feature = x.conv2D_layer(feature, 32, padding_size=2)
feature = x.upsampling(feature, shape=[13, 13])
feature = x.conv2D_layer(feature, alpha.shape[0], padding_size=2, activation="identity")
loss, loss1, loss2, ref_loss, relative_err = x.VAE_loss(
    feature, batch_labels, z_mean, z_log_var
)

# feature = x.dense(batch_images, 500)
# feature = x.dense(feature, 500)

# feature, z_mean, z_log_var = x.latent_space(feature, latent_dim, batch_size)
# feature0 = feature
# feature = x.dense(feature, 500)
# feature = x.dense(feature, 500)
# feature = x.dense(feature, 30*30*1, activation = 'identity')
# feature = x.dense_reshape(feature, [30,30,1])
# loss, loss1, loss2, ref_loss = x.VAE_loss(feature, batch_labels, z_mean, z_log_var)

index = np.arange(train_images.shape[0])
optimizer_Adam = tf.train.AdamOptimizer(1.0e-3)  # Adam as SGD algorithm
# optimizer_RMS = tf.train.RMSPropOptimizer(1.0e-3)
train_op_Adam = optimizer_Adam.minimize(loss)
# train_op_RMS = optimizer_RMS.minimize(loss)


def latent_variables(feature, images, image0, labels, batch_size0, sess):
    feed_dict = {batch_images: images, batch_size: batch_size0}
    feed_dict0 = {batch_images: image0, batch_size: 1}

    lv = sess.run(feature0, feed_dict)
    lv0 = sess.run(feature0, feed_dict0)
    label0 = np.argmax(labels, axis=1)
    fig = plt.figure()
    index0 = np.argwhere(label0 == 0)
    plt.plot(lv[index0, 0], lv[index0, 1], "c.")
    index1 = np.argwhere(label0 == 1)
    plt.plot(lv[index1, 0], lv[index1, 1], "b.")
    index2 = np.argwhere(label0 == 2)
    plt.plot(lv[index2, 0], lv[index2, 1], "y.")
    index3 = np.argwhere(label0 == 3)
    plt.plot(lv[index3, 0], lv[index3, 1], "k.")
    index7 = np.argwhere(label0 == 7)
    plt.plot(lv[index7, 0], lv[index7, 1], "r.")
    index9 = np.argwhere(label0 == 9)
    plt.plot(lv[index9, 0], lv[index9, 1], "g.")
    plt.plot(lv0[0, 0], lv0[0, 1], "go")
    plt.savefig("VAE4.png", dpi=300)
    plt.show()
    plt.close(fig)


init = tf.global_variables_initializer()
num_batch = int(train_images.shape[0] / batch_size0)
# feed_dict_train = {batch_images: train_images, batch_labels: train_labels, batch_size: train_images.shape[0]}
feed_dict_test = {
    batch_images: test_images,
    batch_labels: test_labels,
    batch_size: test_images.shape[0],
}

batch_loss_record = []
test_loss_record = []
epoch_record = []

with tf.Session() as sess:
    sess.run(init)
    for epoch in range(num_epoch):
        np.random.shuffle(index)
        for i in range(num_batch):

            batch_input = train_images[
                index[(i * batch_size0) : ((i + 1) * batch_size0)], :, :, :
            ]
            batch_output = train_labels[
                index[(i * batch_size0) : ((i + 1) * batch_size0)], :
            ]
            feed_dict = {
                batch_images: batch_input,
                batch_labels: batch_output,
                batch_size: batch_size0,
            }
            sess.run(train_op_Adam, feed_dict=feed_dict)
            # sess.run(train_op_RMS, feed_dict = feed_dict )
        if epoch % 10 == 0:

            loss_test_val, loss_test_1_val, loss_test_2_val, ref_loss_val = sess.run(
                [loss, loss1, loss2, ref_loss], feed_dict_test
            )
            loss_batch_val, loss_batch_1_val, loss_batch_2_val = sess.run(
                [loss, loss1, loss2], feed_dict
            )
            batch_loss_record.append(loss_batch_val)
            test_loss_record.append(loss_test_val)
            epoch_record.append(epoch)

            epoch_vec = np.reshape(np.stack(epoch_record), (-1, 1))
            batch_loss_vec = np.reshape(np.stack(batch_loss_record), (-1, 1))
            test_loss_vec = np.reshape(np.stack(test_loss_record), (-1, 1))

            fig = plt.figure()
            plt.semilogy(
                epoch_vec, batch_loss_vec, "ro-", label="Training loss per batch"
            )
            plt.semilogy(epoch_vec, test_loss_vec, "bo-", label="Test loss")
            plt.legend()
            plt.savefig("loss_curve.png", dpi=300)
            # plt.show()
            plt.close(fig)

            record = np.concatenate((epoch_vec, batch_loss_vec, test_loss_vec), axis=1)
            np.savetxt("records.txt", record)

            print(
                "\n Epoch. ",
                epoch,
                " Batch. ",
                i,
                "  Batch_loss.  ",
                [loss_batch_val, loss_batch_1_val, loss_batch_2_val],
                "  Test_loss.   ",
                [loss_test_val, loss_test_1_val, loss_test_2_val],
            )
            print("Ref_loss: ", ref_loss_val)
            # if np.isnan(loss_test_val):
            #     aaa = pred_val
            print("\n")
            # predicted_val, ground_truth_val = sess.run([predicted0, ground_truth0],feed_dict_test)
            predicted_all, rel_err = sess.run([feature, relative_err], feed_dict_test)
            predicted = predicted_all[2]
            ground_truth = test_labels[2]
            fig = plt.figure()
            plt.subplot(121)
            plt.imshow(predicted[:, :, 0].T, cmap="jet")
            plt.colorbar()
            plt.subplot(122)
            plt.imshow(ground_truth[:, :, 0].T, cmap="jet")
            plt.colorbar()
            plt.tight_layout()
            plt.savefig("VAE3.png", dpi=300)
            # plt.show()
            plt.close(fig)

            fig = plt.figure()
            plt.subplot(121)
            plt.contourf(x_grid, y_grid, predicted[:, :, 0].T, cmap="jet")
            plt.colorbar()
            plt.subplot(122)
            plt.contourf(x_grid, y_grid, ground_truth[:, :, 0].T, cmap="jet")
            plt.colorbar()
            plt.tight_layout()
            plt.savefig("VAE33.png", dpi=300)
            # plt.show()
            plt.close(fig)

            print("Average relative error: ", rel_err, "\n")

            print(
                "Err: ",
                np.linalg.norm(predicted[:, :, 0] - ground_truth[:, :, 0])
                / np.linalg.norm(ground_truth[:, :, 0]),
            )
            # latent_variables(feature0, test_images[:1000,0:], test_images[0:1,0:], test_labels0[:1000,0:], 1000, sess)
