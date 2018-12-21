# encoding: utf-8
'''
@author: linwenxing
@contact: linwx.mail@gmail.com
'''

import tensorflow as tf
"""
一个数据集包含多个元素，每个元素的结构都相同。一个元素包含一个或多个 tf.Tensor 对象，这些对象称为组件。
每个组件都有一个 tf.DType，表示张量中元素的类型；以及一个 tf.TensorShape，表示每个元素（可能部分指定）的静态形状。
您可以通过 Dataset.output_types 和 Dataset.output_shapes 属性检查数据集元素各个组件的推理类型和形状。
这些属性的嵌套结构映射到元素的结构，此元素可以是单个张量、张量元组，也可以是张量的嵌套元组。
"""


dataset1 = tf.data.Dataset.from_tensor_slices(tf.random_uniform([4, 10]))
print(dataset1.output_types)  # ==> "tf.float32"
print(dataset1.output_shapes)  # ==> "(10,)"

dataset2 = tf.data.Dataset.from_tensor_slices(
   (tf.random_uniform([4]),
    tf.random_uniform([4, 100], maxval=100, dtype=tf.int32)))
print(dataset2.output_types)  # ==> "(tf.float32, tf.int32)"
print(dataset2.output_shapes)  # ==> "((), (100,))"

dataset3 = tf.data.Dataset.zip((dataset1, dataset2))
print(dataset3.output_types)  # ==> (tf.float32, (tf.float32, tf.int32))
print(dataset3.output_shapes)  # ==> "(10, ((), (100,)))"


"""
为元素的每个组件命名通常会带来便利性，例如，如果它们表示训练样本的不同特征。
除了元组之外，还可以使用 collections.namedtuple 或将字符串映射到张量的字典来表示 Dataset 的单个元素。
"""

dataset = tf.data.Dataset.from_tensor_slices(
   {"a": tf.random_uniform([4]),
    "b": tf.random_uniform([4, 100], maxval=100, dtype=tf.int32)})
print(dataset.output_types)  # ==> "{'a': tf.float32, 'b': tf.int32}"
print(dataset.output_shapes)  # ==> "{'a': (), 'b': (100,)}"


"""
Dataset 转换支持任何结构的数据集。
在使用 Dataset.map()、Dataset.flat_map() 和 Dataset.filter() 转换时
（这些转换会对每个元素应用一个函数），元素结构决定了函数的参数：

"""

# dataset1 = dataset1.map(lambda x: x+1)
#
# dataset2 = dataset2.flat_map(lambda x, y: ...)
#
# # Note: Argument destructuring is not available in Python 3.
# dataset3 = dataset3.filter(lambda x, (y, z): ...)
dataset1.make_initializable_iterator()

iterator = dataset.make_initializable_iterator()


config = tf.ConfigProto(log_device_placement=True,allow_soft_placement
=True)
# 启动session
config.gpu_options.allow_growth = True
with tf.Session(config=config) as sess:
    sess.run(iterator.initializer)
    print(iterator.get_next())
