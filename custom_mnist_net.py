from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


# import tf
import numpy as np
import tensorflow as tf


def load_data():
    mnist = tf.contrib.learn.datasets.load_dataset("mnist")
    train_data = mnist.train.images
    train_labels = np.asarray(mnist.train.labels, dtype=np.int32)
    eval_data = mnist.test.images
    eval_labels = np.asarray(mnist.test.labels, dtype=np.int32)
    return train_data, train_labels, eval_data, eval_labels


sess = tf.InteractiveSession()
x = tf.placeholder("float", shape=[None, 784])
_y = tf.placeholder("float", shape=[None, 10])

W = tf.Variable(tf.zeros([784, 10]))
b = tf.Variable(tf.zeros([10]))

y = tf.nn.softmax(tf.matmul(x, W)+b)
cross_entropy = -tf.reduce_sum(_y*tf.log(y))
train_step = tf.train.GradientDescentOptimizer(learning_rate=0.01).minimize(cross_entropy)
train_data, train_labels, eval_data, eval_labels = load_data()
# bulid dataset
dataset = tf.data.Dataset.from_tensor_slices((train_data, train_labels))
dataset = dataset.repeat().batch(100)
iterator = dataset.make_initializable_iterator()
next_train_data = iterator.get_next()
sess.run(tf.global_variables_initializer())
sess.run(iterator.initializer)
sess.run(tf.local_variables_initializer())

for i in range(1000):
    batch = sess.run(next_train_data)
    labels = batch[1]
    one_hot_labels = tf.one_hot(indices=labels, depth=10)
    one_hot_labels = one_hot_labels.eval()
    train_step.run(feed_dict={x: batch[0], _y: one_hot_labels})
    correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(_y, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    print(accuracy.eval(feed_dict={x: batch[0], _y: one_hot_labels}))

correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(_y, 1))
accuracy = tf.reduce_mean(correct_prediction, tf.cast(correct_prediction, tf.float32))
labels = tf.one_hot(indices=eval_labels, depth=10)
print(accuracy.eval(feed_dict={x: eval_data, _y: np.array(labels)}))
