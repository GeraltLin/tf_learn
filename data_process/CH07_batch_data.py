# encoding: utf-8
'''
@author: linwenxing
@contact: linwx.mail@gmail.com
'''

"""
最简单的批处理形式是将数据集中的 n 个连续元素堆叠为一个元素。
Dataset.batch() 转换正是这么做的，它与 tf.stack() 运算符具有相同的限制
（被应用于元素的每个组件）：即对于每个组件 i，所有元素的张量形状都必须完全相同。

"""
import tensorflow as tf
inc_dataset = tf.data.Dataset.range(100)
dec_dataset = tf.data.Dataset.range(0, -100, -1)
dataset = tf.data.Dataset.zip((inc_dataset, dec_dataset))
batched_dataset = dataset.batch(4)

iterator = batched_dataset.make_one_shot_iterator()
next_element = iterator.get_next()

sess = tf.Session()

print(sess.run(next_element))  # ==> ([0, 1, 2,   3],   [ 0, -1,  -2,  -3])
print(sess.run(next_element))  # ==> ([4, 5, 6,   7],   [-4, -5,  -6,  -7])
print(sess.run(next_element))  # ==> ([8, 9, 10, 11],   [-8, -9, -10, -11])


"""
使用填充批处理张量
上述方法适用于具有相同大小的张量。
不过，很多模型（例如序列模型）处理的输入数据可能具有不同的大小（例如序列的长度不同）。
为了解决这种情况，可以通过 Dataset.padded_batch() 转换来指定一个或多个会被填充的维度，
从而批处理不同形状的张量。
"""
dataset = tf.data.Dataset.range(100)
dataset = dataset.map(lambda x: tf.fill([tf.cast(x, tf.int32)], x))
dataset = dataset.padded_batch(4, padded_shapes=[None])

iterator = dataset.make_one_shot_iterator()
next_element = iterator.get_next()

print(sess.run(next_element))  # ==> [[0, 0, 0], [1, 0, 0], [2, 2, 0], [3, 3, 3]]
print(sess.run(next_element))  # ==> [[4, 4, 4, 4, 0, 0, 0],
                               #      [5, 5, 5, 5, 5, 0, 0],
                               #      [6, 6, 6, 6, 6, 6, 0],
                               #      [7, 7, 7, 7, 7, 7, 7]]
