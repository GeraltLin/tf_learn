# encoding: utf-8
'''
@author: linwenxing
@contact: linwx.mail@gmail.com
'''

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

train_X = np.linspace(-1, 1, 100)
train_Y = 2 * train_X + np.random.randn(train_X.shape[0]) * 0.3 # y=2x，但是加入了噪声
data_ph ={
    'x':tf.placeholder(dtype=tf.float32,name='x'),
    'y':tf.placeholder(dtype=tf.float32,name='y')
}

with tf.variable_scope("fc"):
    W = tf.Variable(tf.random_normal([1]),name='w')
    b = tf.Variable(tf.random_normal([1]),name='b')

    Z = tf.add((tf.multiply(data_ph['x'],W)),b,name='z')

cost = tf.losses.mean_squared_error(data_ph['y'],Z)

optimizer = tf.train.AdamOptimizer(learning_rate = 0.01).minimize(cost)

init = tf.global_variables_initializer()
training_epochs = 200
display_step = 2
saver = tf.train.Saver()

config = tf.ConfigProto(log_device_placement=False,allow_soft_placement
=True)
# 启动session
config.gpu_options.allow_growth = True
with tf.Session(config=config) as sess:
    sess.run(init)

    # Fit all training data
    for epoch in range(training_epochs):
        if epoch%50 ==0:
            saver.save(sess,'../LR_model/LR.ckpt',global_step=epoch+1)
        for (x, y) in zip(train_X, train_Y):
            sess.run(optimizer, feed_dict={data_ph['x']: x, data_ph['y']: y})

        #显示训练中的详细信息
        if epoch % display_step == 0:
            loss = sess.run(cost, feed_dict={data_ph['x']: x, data_ph['y']: y})
            print ("Epoch:", epoch+1, "cost=", loss,"W=", sess.run(W), "b=", sess.run(b))

