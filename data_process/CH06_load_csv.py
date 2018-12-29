# encoding: utf-8
'''
@author: linwenxing
@contact: linwx.mail@gmail.com
'''


"""
CSV 文件格式是用于以纯文本格式存储表格数据的常用格式。
tf.contrib.data.CsvDataset 类提供了一种从符合 RFC 4180 的一个或多个 CSV 文件中提取记录的方法。
给定一个或多个文件名以及默认值列表后，CsvDataset 将生成一个元素元组，元素类型对应于为每个 CSV 记录提供的默认元素类型。
像 TFRecordDataset 和 TextLineDataset 一样，CsvDataset 将接受 filenames（作为 tf.Tensor），因此您可以通过传递 tf.placeholder(tf.string) 进行参数化

"""
import tensorflow as tf
filenames = ["file1.csv", "file2.csv"]

record_defaults = [tf.string] * 2   # Eight required float columns
dataset = tf.contrib.data.CsvDataset(filenames, record_defaults)


iterator = dataset.make_initializable_iterator()
next_element = iterator.get_next()
sess = tf.Session()

# Initialize an iterator over a dataset with 10 elements.
sess.run(iterator.initializer)
for i in range(20):
  value = sess.run(next_element)
  print(value)