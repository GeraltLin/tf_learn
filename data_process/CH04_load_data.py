# encoding: utf-8
'''
@author: linwenxing
@contact: linwx.mail@gmail.com0
'''
"""
消耗 NumPy 数组
如果您的所有输入数据都适合存储在内存中，则根据输入数据创建 Dataset 的最简单方法是将它们转换为 tf.Tensor 对象，
并使用 Dataset.from_tensor_slices()。

"""
import numpy as np
import tensorflow as tf
# Load the training data into two NumPy arrays, for example using `np.load()`.
with np.load("boston_housing.npz") as data:
    x = data['x']
    y = data['y']

features_placeholder = tf.placeholder(x.dtype, x.shape)
labels_placeholder = tf.placeholder(y.dtype, y.shape)

dataset = tf.data.Dataset.from_tensor_slices((features_placeholder, labels_placeholder))
# [Other transformations on `dataset`...]
iterator = dataset.make_initializable_iterator()
sess = tf.Session()
sess.run(iterator.initializer, feed_dict={features_placeholder: x,
                                          labels_placeholder: y})
print(sess.run(iterator.get_next()))


# # Assume that each row of `features` corresponds to the same row as `labels`.
# assert features.shape[0] == labels.shape[0]
#
# dataset = tf.data.Dataset.from_tensor_slices((features, labels))
import keras.datasets.boston_housing
