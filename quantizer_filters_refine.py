import random
import math

import numpy as np
import tensorflow as tf


def quantize_conv_paramters(W, Cs_, K):
    M = int(W.shape[2]) / Cs_
    M = int(M)
    sub_spaces = tf.split(W, M, 2)
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        sess.run(tf.local_variables_initializer())
        sub_spaces = sess.run(sub_spaces)
    centroid_shape = np.asarray(sub_spaces[0][0, 0, :, 0]).shape
    codebooks = [[np.empty(centroid_shape) for j in range(K)] for i in range(M)]
    classes = [np.zeros([W.shape[0], W.shape[1], W.shape[3]], dtype=np.int32) for i in range(M)]
    for i in range(M):
        indices = random.sample(range(W.shape[0]*W.shape[1]*W.shape[3]), K)
        for t in range(K):
            T = indices[t]
            ti = math.floor(T / int(W.shape[1] * W.shape[3]))
            tj = math.floor((T-int(W.shape[1]*W.shape[3])*ti)/int(W.shape[1]))
            tk = T-(int(W.shape[1]*W.shape[3])*ti)-(int(W.shape[1])*tj)
            codebooks[i][t] = np.array(sub_spaces[i][ti, tj, :, tk])
        while True:
            min_dis = np.full([W.shape[0], W.shape[1], W.shape[3]], float("inf"), dtype=np.float32)
            flag = False
            for j in range(W.shape[0]):
                for k in range(W.shape[1]):
                    for l in range(W.shape[3]):
                        for c in range(K):
                            dis = np.sqrt(np.sum(np.square(np.subtract(codebooks[i][c], sub_spaces[i][j, k, :, l])),
                                                 axis=np.ndim(codebooks[i][c])-1))
                            if dis < min_dis[j, k, l]:
                                min_dis[j, k, l] = dis
                                classes[i][j, k, l] = c

            sum_ = [np.zeros(centroid_shape) for j in range(K)]
            count = [0 for j in range(K)]
            for j in range(W.shape[0]):
                for k in range(W.shape[1]):
                    for l in range(W.shape[3]):
                        sum_[classes[i][j, k, l]] += sub_spaces[i][j, k, :, l]
                        count[classes[i][j, k, l]] += 1
            for j in range(K):
                if count[j] != 0:
                    tmp = sum_[j]/count[j]
                    if abs(np.sum(codebooks[i][j] - tmp)) > 1e-5:
                        codebooks[i][j] = tmp
                        flag = True
            if not flag:
                break
    return codebooks, classes


def get_search_list(input, codebooks, M):
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        sess.run(tf.local_variables_initializer())
        sub_inputs = tf.split(input, M, axis=3)
        sub_inputs = sess.run(sub_inputs)
    search_list = [np.zeros([input.shape[0], input.shape[1], input.shape[2], len(codebooks[0])],
                            dtype=np.float32) for i in range(M)]
    for i in range(M):
        for n in range(input.shape[0]):
            for h in range(input.shape[1]):
                for w in range(input.shape[2]):
                    for k in range(len(codebooks[i])):
                        search_list[i][n, h, w, k] = np.inner(codebooks[i][k], sub_inputs[i][n, h, w, :])
    return search_list


def get_ans(search_list, classes):
    kernel_height = classes[0].shape[0]
    kernel_width = classes[0].shape[1]
    out_chanel = classes[0].shape[2]
    batch_number = search_list[0].shape[0]
    input_height = search_list[0].shape[1]
    input_width = search_list[0].shape[2]
    out_height = input_height - kernel_height + 1
    out_width = input_width - kernel_width + 1
    output = np.zeros([batch_number, out_height, out_width, out_chanel])
    for n in range(batch_number):
        for h in range(out_height):
            for w in range(out_width):
                for c in range(out_chanel):
                    for i in range(kernel_height):
                        for j in range(kernel_width):
                            for m in range(len(search_list)):
                                output[n, h, w, c] += search_list[m][n, h+i, w+j, classes[m][i, j, c]]
    return output


# test data
y = [1, 0, 0, 1, 1, 1, 0, 0, 1, 0, 1, 1, 0, 1, 1, 1, 0, 1, 1, 1, 0, 1, 0, 0]
x = [1, 7, 6, 4, 2, 1, 9, 8, 3, 6, 9, 4, 8, 1, 2, 7, 3, 8, 5, 4, 9, 2, 5, 7, 3, 6, 5]
# change into Tensor
x = tf.Variable(x, dtype=tf.float32)
x = tf.reshape(x, [1, 3, 3, 3])
y = tf.Variable(y, dtype=tf.float32)
y = tf.reshape(y, [2, 2, 3, 2])
# convolution of x and y
conv1 = tf.nn.conv2d(x, y, [1, 1, 1, 1], padding='VALID')

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    sess.run(tf.local_variables_initializer())
    # print(sess.run(conv1))
    codebooks, classes = quantize_conv_paramters(y, 1, 2)
    search_list = get_search_list(x, codebooks, 3)
    ans = get_ans(search_list, classes)
    print(ans)