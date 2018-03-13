import tensorflow as tf
import numpy as np
import time
import datetime
import random


def weight_variables(shape, name):
    initial = tf.truncated_normal(shape=shape, stddev=0.1)
    return tf.Variable(initial, name=name+'_weight')


def bias_variables(shape, name):
    initial = tf.constant(value=0.1, dtype=tf.float32, shape=shape)
    return tf.Variable(initial, name=name+'_bias')


def conv2d(x, W):
    return tf.nn.conv2d(input=x, filter=W, strides=[1, 1, 1, 1],
                        padding='SAME')


def max_pooling_2x2(x):
    return tf.nn.max_pool(value=x, ksize=[1, 2, 2, 1],
                          strides=[1, 2, 2, 1], padding='SAME')


def load_data():
    mnist = tf.contrib.learn.datasets.load_dataset("mnist")
    train_data = mnist.train.images
    train_labels = np.asarray(mnist.train.labels, dtype=np.int32)
    eval_data = mnist.test.images
    eval_labels = np.asarray(mnist.test.labels, dtype=np.int32)
    return train_data, train_labels, eval_data, eval_labels


def quantize_weight(weight, K, Cs_):
    m = weight.shape[0]/Cs_
    weight_m = tf.split(weight, m, axis=0)
    for w_m in weight_m:
        indices = random.sample(range(weight.shape[0]), K)
        centroids = []
        for i in range(K):
            centroids.append(w_m[:, indices[i]])
        classes = np.zeros(w_m.shape[1])
        while True:
            flag = False

            pass


# define computation graph
x = tf.placeholder(dtype=tf.float32, shape=[None, 784], name='input')
y_ = tf.placeholder(dtype=tf.int32, shape=[None, 10], name='labels')
# first conv layer
x_reshape = tf.reshape(x, shape=[-1, 28, 28, 1])
W_conv1 = weight_variables([5, 5, 1, 32],name='conv1')
b_conv1 = bias_variables([32], name='conv1')
h_conv1 = tf.nn.relu(conv2d(x_reshape, W_conv1) + b_conv1)
# first pooling

h_pool1 = max_pooling_2x2(h_conv1)

# second conv layer

W_conv2 = weight_variables([5, 5, 32, 64], name='conv2')
b_conv2 = bias_variables([64], name='conv2')
h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)
# second pooling layer

h_pool2 = max_pooling_2x2(h_conv2)

# flatten the pooling result for the dense layer

flatten_pooling = tf.reshape(h_pool2, [-1, 7*7*64])

# fully-connected layer 1
W_fc1 = weight_variables([7*7*64, 1024], name='fc1')
b_fc1 = bias_variables([1024], name='fc1')

h_fc1 = tf.nn.relu(tf.matmul(flatten_pooling, W_fc1) + b_fc1)

# dropout

keep_rate = tf.placeholder(tf.float32, shape=None, name='drop_rate')
h_fc1_after_drop = tf.nn.dropout(h_fc1, keep_rate)

# fully-connected layer 2 (output layer)

W_fc2 = weight_variables([1024, 10], name='fc2')
b_fc2 = bias_variables([10], name='fc2')
h_fc2 = tf.matmul(h_fc1_after_drop, W_fc2) + b_fc2
# use soft-max to get probability
softmax_out = tf.nn.softmax(h_fc2)

# get loss(cross_entropy)
cross_entropy = -tf.reduce_sum(tf.cast(y_, tf.float32)*tf.log(softmax_out))

saver = tf.train.Saver()
with tf.Session() as sess:
    saver.restore(sess, 'model/baseline/baseline.ckpt')
    weights = [v for v in tf.trainable_variables() if v.name.find('weight') != -1]
    biases = [v for v in tf.trainable_variables() if v.name.find('bias') != -1]
    print(sess.run(tf.reshape(weights[0], [5*5*1, 32])))

