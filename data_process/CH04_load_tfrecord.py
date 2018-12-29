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
# # Load the training data into two NumPy arrays, for example using `np.load()`.
# with np.load("boston_housing.npz") as data:
#     x = data['x']
#     y = data['y']
#
# features_placeholder = tf.placeholder(x.dtype, x.shape)
# labels_placeholder = tf.placeholder(y.dtype, y.shape)
#
# dataset = tf.data.Dataset.from_tensor_slices((features_placeholder, labels_placeholder))
# # [Other transformations on `dataset`...]
# iterator = dataset.make_initializable_iterator()
# sess = tf.Session()
# sess.run(iterator.initializer, feed_dict={features_placeholder: x,
#                                           labels_placeholder: y})
# print(sess.run(iterator.get_next()))

"""
消耗 TFRecord 数据
tf.data API 支持多种文件格式，因此您可以处理那些不适合存储在内存中的大型数据集。
例如，TFRecord 文件格式是一种面向记录的简单二进制格式，很多 TensorFlow 应用采用此格式来训练数据。
通过 tf.data.TFRecordDataset 类，您可以将一个或多个 TFRecord 文件的内容作为输入管道的一部分进行流式传输。
"""
def _parse_function(example_proto):
  features = {"data": tf.FixedLenFeature((), tf.string),
              "label": tf.FixedLenFeature((), tf.float32)}
  parsed_features = tf.parse_single_example(example_proto, features)
  data = tf.decode_raw(parsed_features['data'], tf.float64)
  return data, parsed_features["label"]

filenames = ["boston_housing.tfrecords"]
dataset = tf.data.TFRecordDataset(filenames)
dataset = dataset.map(_parse_function)

# dataset = dataset.repeat()  # Repeat the input indefinitely.
# dataset = dataset.batch(32)
iterator = dataset.make_initializable_iterator()
sess = tf.Session()
sess.run(iterator.initializer)

x,y = sess.run(iterator.get_next())
print(x)
print(x,y)