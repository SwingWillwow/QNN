import tensorflow as tf
import numpy as np
import random
# y = [1,0,0,1,1,1,0,1,1,1,0,1,0,0,1,0,1,1,1,1,0,1,0,0]
# x = [1, 7, 6, 6, 9, 4, 5, 4, 9, 4, 2, 1, 8, 1, 2, 2, 5, 7, 9, 8, 3, 7, 3, 8, 3, 6, 5]
# y = [1,0,0,1,0,1,1,0,1,1,0,1,0,1,1,1,1,1,0,0,0,1,0,0,0,0,0,1,1,0,0,1,1,1,0,0,1,1,0,1,0,1,0,1,0,0,1,1]
y = [1, 0, 0, 1, 1, 1, 0, 0, 1, 0,1,1,0,1,1,1,0,1,1,1,0, 1, 0, 0]
x = [1,7,6,4,2,1,9,8,3,6,9,4,8,1,2,7,3,8,5,4,9,2,5,7,3,6,5]
# c = [1, 2, 3, 4, 5, 6, 7, 8]
# c = tf.Variable(c)
# c = tf.reshape(c,[2,2,2])
x = np.array(x, dtype=np.float32)
x = x.reshape([1, 3, 3, 3])
y = np.array(y, dtype=np.float32)
y = y.reshape([2, 2, 3, 2])
conv1 = tf.nn.conv2d(x, y, [1, 1, 1, 1], padding='VALID')
sub_spaces = tf.split(y, 3, 2)
dictionary = [[] for i in range(len(sub_spaces))]
centroid = [[] for i in range(len(sub_spaces))]
classes = np.zeros([y.shape[0], y.shape[1], y.shape[3]], dtype=np.int32)
min_dis = np.full([y.shape[0], y.shape[1], y.shape[3]],float("inf"),dtype=np.float32)
for i in range(len(sub_spaces)):
    shape = np.asarray(sub_spaces[i][0, 0, :, 0]).shape
    centroid[i] = [np.empty(shape) for j in range(4)]

    while True:
        flag = False
        for j in range(y.shape[0]):
            for k in range(y.shape[1]):
                for l in range(y.shape[3]):
                    for c in range(len(centroid[i])):
                        # def true_fn():
                        #     global min_dis, classes, flag, j, k, l, c, dis
                        #
                        #     return tf.constant(0)
                        #
                        # def false_fn():
                        #     return tf.constant(0)
                        # dis = np.sqrt(np.sum(np.square(c-sub_spaces[i][j, k, :, l]), axis=np.ndim(c)))
                        dis = tf.sqrt(tf.reduce_sum(np.square(centroid[i][c]-sub_spaces[i][j, k, :, l]),
                                                    axis=np.ndim(centroid[i][c])))
                        # tf.cond(tf.less(dis, min_dis[j, k, l]),true_fn=true_fn, false_fn=false_fn)
                        dis = tf.Session().run(dis)
                        print(dis)
                        if dis < min_dis[j, k, l]:
                            min_dis[j, k, l] = dis
                            classes[j, k, l] = c
                            flag = True
        if not flag:
            break
        sum_ = [np.zeros(shape) for j in range(4)]
        count = [0 for j in range(4)]
        for j in range(y.shape[0]):
            for k in range(y.shape[1]):
                for l in range(y.shape[3]):
                    sum_[classes[j, k, l]] += sub_spaces[i][j, k, :, l]
                    count[classes[j, k, l]] += 1
        for j in range(4):
            if count[j] != 0:
                centroid[i][j] = sum_[j]/count[j]
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    for c in centroid[0]:
        if isinstance(c, tf.Tensor):
            print(sess.run(c))
        else:
            print(c)

