# -*- coding: utf-8 -*-
"""
@author: linwenxing
@contact: linwx.mail@gmail.com
"""


import tensorflow as tf
# 导入 MINST 数据集
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

#参数设置
learning_rate = 0.001
training_epochs = 25
batch_size = 100
display_step = 1

# Network Parameters
n_hidden_1 = 256 # 1st layer number of features
n_hidden_2 = 256 # 2nd layer number of features
n_input = 784 # MNIST data 输入 (img shape: 28*28)
n_classes = 10  # MNIST 列别 (0-9 ，一共10类)

# tf Graph input
x = tf.placeholder("float", [None, n_input])
y = tf.placeholder("float", [None, n_classes])

# Create model
def multilayer_perceptron(x, weights, biases):
    # Hidden layer with RELU activation
    layer_1 = tf.add(tf.matmul(x, weights['h1']), biases['b1'])
    layer_1 = tf.nn.relu(layer_1)
    # Hidden layer with RELU activation
    layer_2 = tf.add(tf.matmul(layer_1, weights['h2']), biases['b2'])
    layer_2 = tf.nn.relu(layer_2)
    # Output layer with linear activation
    out_layer = tf.matmul(layer_2, weights['out']) + biases['out']
    return out_layer
    
# Store layers weight & bias
weights = {
    'h1': tf.Variable(tf.random_normal([n_input, n_hidden_1])),
    'h2': tf.Variable(tf.random_normal([n_hidden_1, n_hidden_2])),
    'out': tf.Variable(tf.random_normal([n_hidden_2, n_classes]))
}
biases = {
    'b1': tf.Variable(tf.random_normal([n_hidden_1])),
    'b2': tf.Variable(tf.random_normal([n_hidden_2])),
    'out': tf.Variable(tf.random_normal([n_classes]))
}

# 构建模型
pred = multilayer_perceptron(x, weights, biases)

# Define loss and optimizer
cost = tf.reduce_mean(tf.losses.softmax_cross_entropy(onehot_labels=y,logits=pred))

tf.summary.scalar(name='loss_value_2',tensor=cost)
global_step = tf.Variable(0, trainable=False)
learning_rate = 0.001

learning_rate = tf.train.exponential_decay(learning_rate,
                                           global_step=global_step,
                                           decay_steps=10000, decay_rate=0.9)
optimizer = tf.train.AdamOptimizer(learning_rate).minimize(cost,global_step=global_step)

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






