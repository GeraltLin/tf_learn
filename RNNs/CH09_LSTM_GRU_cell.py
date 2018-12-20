# -*- coding: utf-8 -*-
"""
@author: linwenxing
@contact: linwx.mail@gmail.com
"""

import tensorflow as tf
# 导入 MINST 数据集
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("../basic/MNIST_data", one_hot=True)

n_input = 28 # MNIST data 输入 (img shape: 28*28)
n_steps = 28 # timesteps
n_hidden = 128 # hidden layer num of features
n_classes = 10  # MNIST 列别 (0-9 ，一共10类)

tf.reset_default_graph()

# tf Graph input
x = tf.placeholder("float", [None, n_steps, n_input])
y = tf.placeholder("float", [None, n_classes])


# x1 = tf.unstack(x, n_steps, 1) static_rnn 传的是个listx

#1 BasicLSTMCell
lstm_cell = tf.nn.rnn_cell.LSTMCell(num_units=n_hidden)
# outputs, states = tf.nn.static_rnn(lstm_cell, x1, dtype=tf.float32)
outputs, states = tf.nn.dynamic_rnn(lstm_cell,x,dtype=tf.float32)
outputs = tf.transpose(outputs, [1, 0, 2]) #dynamic_rnn 需要转置





# gru
# lstm_cell = tf.nn.rnn_cell.GRUCell(num_units=n_hidden)
#
# outputs = tf.nn.static_rnn(lstm_cell, x1, dtype=tf.float32)

# muti cell
# num_units = [128, 64]
# cells = [tf.nn.rnn_cell.LSTMCell(num_units=n) for n in num_units]
# stacked_rnn_cell = tf.nn.rnn_cell.MultiRNNCell(cells)
# outputs, states = tf.nn.static_rnn(stacked_rnn_cell, x1, dtype=tf.float32)

pred = tf.layers.dense(outputs[-1],n_classes)



learning_rate = 0.001
training_iters = 100000
batch_size = 128
display_step = 10

# Define loss and optimizer
cost = tf.reduce_mean(tf.losses.softmax_cross_entropy(onehot_labels=y,logits=pred))
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)

# Evaluate model
correct_pred = tf.equal(tf.argmax(pred,1), tf.argmax(y,1))
accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

# 启动session
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    step = 1
    # Keep training until reach max iterations
    while step * batch_size < training_iters:
        batch_x, batch_y = mnist.train.next_batch(batch_size)
        # Reshape data to get 28 seq of 28 elements
        batch_x = batch_x.reshape((batch_size, n_steps, n_input))
        # Run optimization op (backprop)
        sess.run(optimizer, feed_dict={x: batch_x, y: batch_y})
        if step % display_step == 0:
            # 计算批次数据的准确率
            acc = sess.run(accuracy, feed_dict={x: batch_x, y: batch_y})
            # Calculate batch loss
            loss = sess.run(cost, feed_dict={x: batch_x, y: batch_y})
            print ("Iter " + str(step*batch_size) + ", Minibatch Loss= " + \
                  "{:.6f}".format(loss) + ", Training Accuracy= " + \
                  "{:.5f}".format(acc))
        step += 1
    print (" Finished!")

    # 计算准确率 for 128 mnist test images
    test_len = 128
    test_data = mnist.test.images[:test_len].reshape((-1, n_steps, n_input))
    test_label = mnist.test.labels[:test_len]
    print ("Testing Accuracy:", \
        sess.run(accuracy, feed_dict={x: test_data, y: test_label}))

    
    

