# encoding: utf-8
'''
@author: linwenxing
@contact: linwx.mail@gmail.com
'''


"""
构建了表示输入数据的 Dataset 后，下一步就是创建 Iterator 来访问该数据集中的元素。
tf.data API 目前支持下列迭代器，复杂程度逐渐增大：
    单次，
    可初始化，
    可重新初始化，以及
    可馈送。
单次迭代器是最简单的迭代器形式，仅支持对数据集进行一次迭代，不需要显式初始化。
单次迭代器可以处理基于队列的现有输入管道支持的几乎所有情况，但它们不支持参数化。
以 Dataset.range() 为例：

"""
import tensorflow as tf
sess = tf.Session()
dataset = tf.data.Dataset.range(100)
iterator = dataset.make_one_shot_iterator()
next_element = iterator.get_next()
for i in range(100):
    values = sess.run(next_element)
    print(values)
    assert i == values


"""
您需要先运行显式 iterator.initializer 操作，然后才能使用可初始化迭代器。虽然有些不便，
但它允许您使用一个或多个 tf.placeholder() 张量（可在初始化迭代器时馈送）参数化数据集的定义。继续以 Dataset.range() 为例：
"""

max_value = tf.placeholder(tf.int64,shape=[])
dataset = tf.data.Dataset.range(max_value)
iterator = dataset.make_initializable_iterator()
next_element = iterator.get_next()

# Initialize an iterator over a dataset with 10 elements.
sess.run(iterator.initializer, feed_dict={max_value: 10})
for i in range(10):
  value = sess.run(next_element)
  print(value)
  assert i == value


"""
可重新初始化迭代器可以通过多个不同的 Dataset 对象进行初始化。
例如，您可能有一个训练输入管道，它会对输入图片进行随机扰动来改善泛化；
还有一个验证输入管道，它会评估对未修改数据的预测。
这些管道通常会使用不同的 Dataset 对象，这些对象具有相同的结构（即每个组件具有相同类型和兼容形状）。
"""
training_dataset = tf.data.Dataset.range(100).map(
    lambda x: x + tf.random_uniform([], -10, 10, tf.int64))
validation_dataset = tf.data.Dataset.range(50)

iterator = tf.data.Iterator.from_structure(training_dataset.output_types,                                           training_dataset.output_shapes)
next_element = iterator.get_next()

training_init_op = iterator.make_initializer(training_dataset)
validation_init_op = iterator.make_initializer(validation_dataset)

for _ in range(20):
  # Initialize an iterator over the training dataset.
  sess.run(training_init_op)
  for _ in range(100):
    sess.run(next_element)

  # Initialize an iterator over the validation dataset.
  sess.run(validation_init_op)
  for _ in range(50):
    sess.run(next_element)


"""
可馈送迭代器可以与 tf.placeholder 一起使用，以选择所使用的 Iterator（在每次调用 tf.Session.run 时）
（通过熟悉的 feed_dict 机制）。它提供的功能与可重新初始化迭代器的相同，但在迭代器之间切换时不需要从数据集的开头初始化迭代器。
例如，以上面的同一训练和验证数据集为例，您可以使用 tf.data.Iterator.from_string_handle 定义一个可让您在两个数据集之间切换的可馈送迭代器：
"""# Define training and validation datasets with the same structure.
training_dataset = tf.data.Dataset.range(100).map(
    lambda x: x + tf.random_uniform([], -10, 10, tf.int64)).repeat()
validation_dataset = tf.data.Dataset.range(50)


handle = tf.placeholder(tf.string, shape=[])
iterator = tf.data.Iterator.from_string_handle(
    handle, training_dataset.output_types, training_dataset.output_shapes)
next_element = iterator.get_next()

training_iterator = training_dataset.make_one_shot_iterator()
validation_iterator = validation_dataset.make_initializable_iterator()


training_handle = sess.run(training_iterator.string_handle())
validation_handle = sess.run(validation_iterator.string_handle())

# Loop forever, alternating between training and validation.
while True:

  for _ in range(200):
    sess.run(next_element, feed_dict={handle: training_handle})

  sess.run(validation_iterator.initializer)
  for _ in range(50):
    sess.run(next_element, feed_dict={handle: validation_handle})

"""
消耗迭代器中的值
Iterator.get_next() 方法返回一个或多个 tf.Tensor 对象，这些对象对应于迭代器有符号的下一个元素。
每次评估这些张量时，它们都会获取底层数据集中下一个元素的值。（请注意，与 TensorFlow 中的其他有状态对象一样，
调用 Iterator.get_next() 并不会立即使迭代器进入下个状态。您必须在 TensorFlow 表达式中使用此函数返回的 tf.Tensor 对象，
并将该表达式的结果传递到 tf.Session.run()，以获取下一个元素并使迭代器进入下个状态。）
如果迭代器到达数据集的末尾，则执行 Iterator.get_next() 操作会产生 tf.errors.OutOfRangeError。
在此之后，迭代器将处于不可用状态；如果需要继续使用，则必须对其重新初始化。
"""

dataset = tf.data.Dataset.range(5)
iterator = dataset.make_initializable_iterator()
next_element = iterator.get_next()

# Typically `result` will be the output of a model, or an optimizer's
# training operation.
result = tf.add(next_element, next_element)

sess.run(iterator.initializer)
print(sess.run(result))  # ==> "0"
print(sess.run(result))  # ==> "2"
print(sess.run(result))  # ==> "4"
print(sess.run(result))  # ==> "6"
print(sess.run(result))  # ==> "8"
try:
  sess.run(result)
except tf.errors.OutOfRangeError:
  print("End of dataset")  # ==> "End of dataset"


"""
如果数据集的每个元素都具有嵌套结构，
则 Iterator.get_next() 的返回值将是一个或多个 tf.Tensor 对象，这些对象具有相同的嵌套结构：
"""

dataset1 = tf.data.Dataset.from_tensor_slices(tf.random_uniform([4, 10]))
dataset2 = tf.data.Dataset.from_tensor_slices((tf.random_uniform([4]), tf.random_uniform([4, 100])))
dataset3 = tf.data.Dataset.zip((dataset1, dataset2))

iterator = dataset3.make_initializable_iterator()

sess.run(iterator.initializer)
next1, (next2, next3) = iterator.get_next()