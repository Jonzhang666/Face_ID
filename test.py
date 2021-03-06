import tensorflow as tf
import numpy as np
from tensorflow.examples.tutorials.mnist import input_data

from adjust_image_size import load_dataset, resize_image, IMAGE_SIZE
# number 0 to 1000 data

images, labels = load_dataset("D:\pythonworkplace\Face_ID\data")

# Parameters
learning_rate = 0.001
training_epochs = 29

# Network Parameters
n_input = IMAGE_SIZE * IMAGE_SIZE
n_classes = 3


def compute_accuracy(v_xs, v_ys):
    global prediction
    y_pre = sess.run(prediction, feed_dict={xs: v_xs, keep_prob: 1})
    correct_prediction = tf.equal(tf.argmax(y_pre, 1), tf.argmax(v_ys, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    result = sess.run(accuracy, feed_dict={xs: v_xs, ys: v_ys, keep_prob: 1})
    return result


def weight_variable(shape, name):
    initial = tf.truncated_normal(shape, stddev=0.1) #标准差0.1
    return tf.Variable(initial, dtype=tf.float32, name=name)


def bias_variable(shape, name):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial, dtype=tf.float32, name=name)


def conv2d(x, W):
    # stride [1, x_movement, y_movement, 1]
    # Must have strides[0] = strides[3] = 1
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')


def max_pool_2x2(x):
    # stride [1, x_movement, y_movement, 1]
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')


# define placeholder for inputs to network
xs = tf.placeholder(tf.float32, [None, IMAGE_SIZE, IMAGE_SIZE])/255  # IMAGE_SIZExIMAGE_SIZE/255.
ys = tf.placeholder(tf.float32, [None, n_classes])
keep_prob = tf.placeholder(tf.float32)

x_image = tf.reshape(xs, [-1, IMAGE_SIZE, IMAGE_SIZE, 1])
# print(x_image.shape)  # [n_samples, 28,28,1]

## conv1 layer ##
W_conv1 = weight_variable([10, 10, 1, 32], "W_conv1")  # patch 5x5, in size 1, out size 32
b_conv1 = bias_variable([32], "b_conv1")
h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)  # output size 28x28x32
h_pool1 = max_pool_2x2(h_conv1)  # output size 14x14x32

## conv2 layer ##
W_conv2 = weight_variable([10, 10, 32, 64], "W_conv2")  # patch 5x5, in size 32, out size 64
b_conv2 = bias_variable([64], "b_conv2")
h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)  # output size 14x14x64
h_pool2 = max_pool_2x2(h_conv2)  # output size 7x7x64

## conv3 layer ##
W_conv3 = weight_variable([10, 10, 64, 128], "W_conv3")  # patch 5x5, in size 32, out size 64
b_conv3 = bias_variable([128], "b_conv3")
h_conv3 = tf.nn.relu(conv2d(h_pool2, W_conv3) + b_conv3)  # output size 14x14x64
h_pool3 = max_pool_2x2(h_conv3)  # output size 7x7x64

## fc1 layer ##
W_fc1 = weight_variable([7 * 7 * 128, 2048], "W_fc1")
b_fc1 = bias_variable([2048], "b_fc1")
# [n_samples, 7, 7, 64] ->> [n_samples, 7*7*64]
h_pool2_flat = tf.reshape(h_pool3, [-1, 7 * 7 * 128])
h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)
h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

## fc2 layer ##
W_fc2 = weight_variable([2048, 3], "W_fc2")
b_fc2 = bias_variable([3], "b_fc2")
prediction = tf.nn.softmax(tf.matmul(h_fc1_drop, W_fc2) + b_fc2)

# saver model
saver = tf.train.Saver()

# Launch the gtrph
with tf.Session() as sess:
    saver.restore(sess, "my_net/save_net.ckpt")
    # print("weights:", sess.run(W))
    # print("biases:", sess.run(b))

    img = images[:1000].reshape([-1, 56, 56, 1])
    img_label = labels[:1000]
    real_label = sess.run(tf.argmax(img_label, 1))
    h_c1 = sess.run(h_conv1, feed_dict={x_image: img})
    h_p1 = sess.run(h_pool1, feed_dict={h_conv1: h_c1})
    h_c2 = sess.run(h_conv2, feed_dict={h_pool1: h_p1})
    h_p2 = sess.run(h_pool2, feed_dict={h_conv2: h_c2})

    h_p2_f = sess.run(h_pool2_flat, feed_dict={h_pool2: h_p2})
    h_f1 = sess.run(h_fc1, feed_dict={h_pool2_flat: h_p2_f})
    h_f1_d = sess.run(h_fc1_drop, feed_dict={h_fc1: h_f1, keep_prob: 0.5})
    ret = sess.run(prediction, feed_dict={h_fc1_drop: h_f1_d})

    num_pred = sess.run(tf.argmax(ret, 1))

    correct_prediction = tf.equal(tf.argmax(ret, 1), tf.argmax(img_label, 1))
    accuracy = sess.run(tf.reduce_mean(tf.cast(correct_prediction, tf.float32)))

    print("预测值:\n", num_pred)
    print("真实值:\n", real_label)
    print("准确率：", accuracy)
    print("模型恢复成功")


