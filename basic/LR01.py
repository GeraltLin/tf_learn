# encoding: utf-8
'''
@author: linwenxing
@contact: linwx.mail@gmail.com
'''
from collections import OrderedDict

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

train_X = np.linspace(-1, 1, 100)
train_Y = 2 * train_X + np.random.randn(train_X.shape[0]) * 0.3 # y=2x，但是加入了噪声

X = tf.placeholder(dtype=tf.float32)
Y = tf.placeholder(dtype=tf.float32)

W = tf.Variable(tf.random_normal([1]),name='w')
b = tf.Variable(tf.random_normal([1]),name='b')

Z = tf.multiply(X,W)+b

cost = tf.losses.mean_squared_error(Y,Z)

optimizer = tf.train.AdamOptimizer(learning_rate = 0.01).minimize(cost)

init = tf.global_variables_initializer()
training_epochs = 200

display_step = 2

# 启动session
with tf.Session() as sess:
    sess.run(init)

    # Fit all training data
    for epoch in range(training_epochs):
        for (x, y) in zip(train_X, train_Y):
            sess.run(optimizer, feed_dict={X: x, Y: y})

        #显示训练中的详细信息
        if epoch % display_step == 0:
            loss = sess.run(cost, feed_dict={X: train_X, Y:train_Y})
            print ("Epoch:", epoch+1, "cost=", loss,"W=", sess.run(W), "b=", sess.run(b))


    print (" Finished!")
    print ("cost=", sess.run(cost, feed_dict={X: train_X, Y: train_Y}), "W=", sess.run(W), "b=", sess.run(b))
    #print ("cost:",cost.eval({X: train_X, Y: train_Y}))

    #图形显示
    plt.plot(train_X, train_Y, 'ro', label='Original data')
    plt.plot(train_X, sess.run(W) * train_X + sess.run(b), label='Fitted line')
    plt.legend()
    plt.show()