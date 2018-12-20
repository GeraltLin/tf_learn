# encoding: utf-8
'''
@author: linwenxing
@contact: linwx.mail@gmail.com
'''

from CNNs.download_cifar import cifar10_input
import tensorflow as tf
import numpy as np
import time

batch_size = 128
data_dir = './cifar-10-batches-bin'
print("begin")
images_train, labels_train = cifar10_input.inputs(eval_data=False, data_dir=data_dir, batch_size=batch_size)
images_test, labels_test = cifar10_input.inputs(eval_data=True, data_dir=data_dir, batch_size=batch_size)
print("begin data")

# tf Graph Input
x = tf.placeholder(tf.float32, [None, 24, 24, 3])  # cifar data image of shape 24*24*3
y = tf.placeholder(tf.float32, [None, 10])  # 0-9 数字=> 10 classes
training = tf.placeholder(tf.bool)
x_image = tf.reshape(x, [-1, 24, 24, 3])


h_conv1 = tf.layers.conv2d(x_image, 64, [5, 5], 1, 'SAME')
# h_conv1 = tf.layers.batch_normalization(inputs=h_conv1, training=training)
h_conv1 = tf.nn.relu(h_conv1)
h_pool1 = tf.layers.max_pooling2d(h_conv1, [2, 2], strides=2, padding='SAME')

h_conv2 = tf.layers.conv2d(h_pool1, 64, [5, 5], 1, 'SAME')
# h_conv2 = tf.layers.batch_normalization(inputs=h_conv2, training=training)
h_conv2 = tf.nn.relu(h_conv2)

h_pool2 = tf.layers.max_pooling2d(h_conv2, [2, 2], strides=2, padding='SAME')

nt_hpool2 = tf.layers.average_pooling2d(h_pool2, [6, 6], strides=6, padding='SAME')
nt_hpool2_flat = tf.reshape(nt_hpool2, [-1, 64])

y_conv = tf.layers.dense(nt_hpool2_flat, 10, activation=tf.nn.softmax)

cross_entropy = tf.losses.softmax_cross_entropy(onehot_labels=y,logits=y_conv)

cross_entropy = tf.reduce_mean(cross_entropy)
train_step = tf.train.AdamOptimizer(1e-3).minimize(cross_entropy)

correct_prediction = tf.equal(tf.argmax(y_conv, 1), tf.argmax(y, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))

sess = tf.Session()
sess.run(tf.global_variables_initializer())
tf.train.start_queue_runners(sess=sess)
time1 = time.time()

for i in range(15000):  # 20000
    image_batch, label_batch = sess.run([images_train, labels_train])
    label_b = np.eye(10, dtype=float)[label_batch]  # one hot

    train_step.run(feed_dict={x: image_batch, y: label_b,training:True}, session=sess)
    if i % 200 == 0:
        time2 = time.time()
        print('time {}'.format(time2 - time1))
        train_accuracy = accuracy.eval(feed_dict={
            x: image_batch, y: label_b,training:False}, session=sess)
        print("step %d, training accuracy %g" % (i, train_accuracy))

image_batch, label_batch = sess.run([images_test, labels_test])
label_b = np.eye(10, dtype=float)[label_batch]  # one hot
print("finished！ test accuracy %g" % accuracy.eval(feed_dict={
    x: image_batch, y: label_b}, session=sess))