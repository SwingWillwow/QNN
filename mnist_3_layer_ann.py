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


def dense_layer(inputs, in_size, out_size):
    tmpW = tf.Variable(tf.truncated_normal([in_size, out_size], stddev=0.1, dtype=tf.float32))
    tmpW = tf.div(tmpW, tf.sqrt(tf.cast(in_size,dtype=tf.float32)))
    W = tf.Variable(tmpW, name="Weight", dtype=tf.float32)
    b = tf.Variable(tf.constant(0.1, shape=[out_size]), name="bias")
    return tf.nn.relu(tf.matmul(inputs, W) + b)


def main(unused_argv):
    train_data, train_labels, eval_data, eval_labels = load_data()
    # bulid dataset
    dataset = tf.data.Dataset.from_tensor_slices((train_data, train_labels))
    dataset = dataset.repeat().batch(100)
    iterator = dataset.make_initializable_iterator()
    next_train_data = iterator.get_next()
    # bulid testset
    testset = tf.data.Dataset.from_tensor_slices((eval_data, eval_labels))
    iterator_test = testset.make_initializable_iterator()
    next_test_data = iterator_test.get_next()
    x = tf.placeholder(tf.float32, shape=[None, train_data.shape[1]], name='input')
    # y = tf.placeholder(tf.int32, shape=[None, 10], name='labels')
    y = tf.placeholder(tf.int32, shape=[None], name='labels')
    # defined layers
    with tf.name_scope('hidden_1'):
        h1 = dense_layer(x, train_data.shape[1], 100)
    # with tf.name_scope('hidden_2'):
    #     h2 = dense_layer(h1, 1000, 300)
    with tf.name_scope('output'):
        z = dense_layer(h1, 100, 10)
    with tf.name_scope('loss'):
        # classes = tf.argmax(y, axis=1)
        # loss = tf.losses.sparse_softmax_cross_entropy(labels=classes, logits=z)
        loss = tf.losses.sparse_softmax_cross_entropy(labels=y, logits=z)
    loss_summary = tf.summary.scalar('loss', loss)
    with tf.name_scope('accuracy'):
        # accuracy = tf.metrics.accuracy(labels=classes, predictions=tf.argmax(input=z, axis=1))
        accuracy = tf.metrics.accuracy(labels=y, predictions=tf.argmax(input=z, axis=1))
    with tf.name_scope('train'):
        train_step = tf.train.AdamOptimizer(learning_rate=1e-4).minimize(loss=loss)
    with tf.Session() as sess:

        # sess = tf_debug.LocalCLIDebugWrapperSession(sess)
        sess.run(tf.global_variables_initializer())
        sess.run(iterator.initializer)
        sess.run(iterator_test.initializer)
        sess.run(tf.local_variables_initializer())
        # train_writer = tf.summary.FileWriter('./summary', graph=tf.get_default_graph())
        for i in range(550*280):
            batch = sess.run(next_train_data)
            labels = batch[1]
            # one_hot_labels = tf.one_hot(labels, depth=10)
            # one_hot_labels = sess.run(one_hot_labels)
            summary, _ = sess.run([loss_summary, train_step],
                                  feed_dict={x: batch[0],
                                             y: labels})

            if i % 100 == 1:
                ac, los = sess.run([accuracy, loss], feed_dict={x: batch[0], y: labels})
                print("accuracy:", ac[0], " loss: ", los, " step: ", i)
        # print('Final loss on test set:', sess.run([loss],
        #                                           feed_dict={x: np.array(sess.run(next_test_data)[0]),
        #                                                      y: np.array(sess.run(next_test_data)[1])}))


if __name__ == "__main__":
    tf.app.run()
