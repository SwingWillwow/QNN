import tensorflow as tf
import numpy as np
# from tensorflow.python import debug as tf_debug
import random

y = [[3, 7, 4, 1],
     [2, 4, 3, 5],
     [0, 0, 1, 0],
     [1, 0, 0, 0],
     [4, 3, 2, 1],
     [2, 4, 5, 3]]
x = [1,7,6,6,9,4,5,4,9,4,2,1,8,1,2,2,5,7,9,8,3,7,3,8,3,6,5]
y_3 = [1,0,1,0,1,0,0,1,1,1,0,0]
y_4 = [1,0,0,1,1,1,0,1,1,1,0,1,0,0,1,0,1,1,1,1,0,1,0,0]
# print(np.reshape(y_2, y_2.size))
# y_2 = np.reshape(y_2, [3, 3, 2, 2])
# print(y_2)
x = np.array(x, dtype=np.float32)
x = x.reshape([1, 3, 3, 3])
# y_3 = np.array(y_3, dtype=np.float32)
# y_3 = y_3.reshape([2, 2, 3, 1])
y_4 = np.array(y_4, dtype=np.float32)
y_4 = y_4.reshape([2, 2, 3, 2])
y = np.array(y,dtype=float)
filters = tf.split(y_4, 2, axis=3)
for i in range(len(filters)):
    filters[i] = tf.reshape(filters[i], [2, 2, 3])
subspace = []

# result = tf.nn.conv2d(input=x, filter=y_4, strides=[1, 1, 1, 1], padding='VALID')
# y = tf.Variable(y)

# result2 = tf.split(y_2, 2, axis=0)
with tf.Session() as sess:
    # sess = tf_debug.LocalCLIDebugWrapperSession(sess)
    sess.run(tf.global_variables_initializer())
    sess.run(tf.local_variables_initializer())
    # print(sess.run(result))
    print(sess.run(tf.reduce_mean(y,1)))
    # print(sess.run(result2[0]))
    # print('------------------------')
    # print(sess.run(tf.transpose(result2[0])))
    # result2[0] = tf.reshape(tf.transpose(result2[0]), [9, 2])
    # print(sess.run(result2[0]))
    # K = 2
    #
    # for w in result:
    #     w = tf.transpose(w)
    #     indices = random.sample(range(4), K)
    #     centroid = []
    #     for i in range(K):
    #         centroid.append(w[indices[i]])
    #     # for c in centroid:
    #     #     print(c.eval())
    #     # print(w.eval())
    #     classes = np.zeros(w.shape[0])
    #     while True:
    #         flag = False
    #         for i in range(w.shape[0]):
    #             min_dis = 1000000
    #             for j in range(len(centroid)):
    #                 dis = tf.sqrt(tf.reduce_sum(tf.square(centroid[j]-w[i])))
    #                 if dis < min_dis:
    #                     classes[i] = j
    #                     flag = True
    #
    #         if not flag:
    #             break
    #     break




