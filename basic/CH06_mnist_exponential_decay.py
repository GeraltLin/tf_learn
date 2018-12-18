# -*- coding: utf-8 -*-
"""
@author: linwenxing
@contact: linwx.mail@gmail.com
"""

import tensorflow as tf #导入tensorflow库
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)
import pylab 

tf.reset_default_graph()
# tf Graph Input
x = tf.placeholder(tf.float32, [None, 784]) # mnist data维度 28*28=784
y = tf.placeholder(tf.float32, [None, 10]) # 0-9 数字=> 10 classes

# Set model weights
W = tf.Variable(tf.random_normal([784, 10]))
b = tf.Variable(tf.zeros([10]))

# 构建模型
pred = tf.nn.softmax(tf.matmul(x, W) + b) # Softmax分类

# Minimize error using cross entropy
cost = tf.reduce_mean(-tf.reduce_sum(y*tf.log(pred), reduction_indices=1))


tf.summary.scalar(name='loss_value',tensor=cost)
global_step = tf.Variable(0, trainable=False)
learning_rate = 0.1
learning_rate = tf.train.exponential_decay(learning_rate,
                                           global_step=global_step,
                                           decay_steps=10000, decay_rate=0.9)
optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(cost,global_step=global_step)

training_epochs = 100
batch_size = 100
display_step = 1
saver = tf.train.Saver()

# 启动session
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())# Initializing OP
    merge_op = tf.summary.merge_all()
    summary_writer = tf.summary.FileWriter(logdir='./MNIST_log', graph=sess.graph)

    # 启动循环开始训练
    for epoch in range(training_epochs):
        avg_cost = 0.
        total_batch = int(mnist.train.num_examples/batch_size)

        # 遍历全部数据集
        for i in range(total_batch):
            batch_xs, batch_ys = mnist.train.next_batch(batch_size)
            # Run optimization op (backprop) and cost op (to get loss value)
            _, c = sess.run([optimizer, cost], feed_dict={x: batch_xs,
                                                          y: batch_ys})

            curent_step = sess.run(global_step)
            # print(curent_step)
            # Compute average loss
            avg_cost += c / total_batch

            # #写merge
            merge_op_write = sess.run(merge_op, feed_dict={x: batch_xs,y: batch_ys})
            summary_writer.add_summary(merge_op_write, global_step=curent_step)

        # 显示训练中的详细信息
        if (epoch+1) % display_step == 0:
            print ("Epoch:", '%04d' % (epoch+1), "cost=", "{:.9f}".format(avg_cost))
            saver.save(sess,'./MNIST_model/mnist.ckpt',global_step=curent_step)
            # Save model weights to disk

            # print( " Finished!")

            # 测试 model
            correct_prediction = tf.equal(tf.argmax(pred, 1), tf.argmax(y, 1))
            # 计算准确率
            accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
            print ("Accuracy:", accuracy.eval({x: mnist.test.images, y: mnist.test.labels}))





