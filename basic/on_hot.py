#coding: utf-8
"""
@author: linwenxing
@contact: linwx.mail@gmail.com
"""
import tensorflow as tf
Y = tf.placeholder(dtype=tf.floa32)# print(Y)
y_one_hot = tf.one_hot(Y, 4)

with tf.Session() as sess:
    tt =sess.run(y_one_hot,feed_dict={Y:[0,1,2,3,3]})
    print(tt)