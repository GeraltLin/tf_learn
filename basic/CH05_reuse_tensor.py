# encoding: utf-8
'''
@author: linwenxing
@contact: linwx.mail@gmail.com
'''

"""
tf.get_variable在创建变量时， 会去检查图（一个计算任务） 中是否已经创
建过该变量。 如果创建过并且本次调用时没有被设为共享方式， 则会报错。
"""
import tensorflow as tf

tf.reset_default_graph()

# var1 = tf.get_variable("firstvar",shape=[2],dtype=tf.float32)
# var2 = tf.get_variable("firstvar",shape=[2],dtype=tf.float32)

with tf.variable_scope("test1", ):
    var1 = tf.get_variable("firstvar", shape=[2], dtype=tf.float32)

    with tf.variable_scope("test2"):
        var2 = tf.get_variable("firstvar", shape=[2], dtype=tf.float32)

print("var1:", var1.name)
print("var2:", var2.name)

with tf.variable_scope("test1", reuse=True):
    var3 = tf.get_variable("firstvar", shape=[2], dtype=tf.float32)
    with tf.variable_scope("test2"):
        var4 = tf.get_variable("firstvar", shape=[2], dtype=tf.float32)

print("var3:", var3.name)
print("var4:", var4.name)