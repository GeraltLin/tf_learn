# encoding: utf-8
'''
@author: linwenxing
@contact: linwx.mail@gmail.com
'''
import tensorflow as tf
import numpy as np

def _float_feature(value):
    return tf.train.Feature(float_list=tf.train.FloatList(value=[value]))

def _bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=value))



def _float_feature_list(value):
    return tf.train.Feature(float_list=tf.train.FloatList(value=value))

def save_tfrecords(data, label, desfile):
    with tf.python_io.TFRecordWriter(desfile) as writer:

        for i in range(len(data)):
            feature = {'x': _float_feature_list(list(data[i])),
                       'y': _float_feature(label[i])}
            example = tf.train.Example(features=tf.train.Features(feature=feature))
            serialized = example.SerializeToString()
            writer.write(serialized)




with np.load("boston_housing.npz") as data:
    x = data['x']
    y = data['y']
print(list(x[0]))
save_tfrecords(x,y,'boston_housing.tfrecords')



def save_tfrecords(data, label, desfile):
    with tf.python_io.TFRecordWriter(desfile) as writer:
        for i in range(len(data)):
            features = tf.train.Features(
                feature = {
                    "x":tf.train.Feature(bytes_list = tf.train.BytesList(value = [data[i].astype(np.float64).tostring()])),
                    "y":tf.train.Feature(int64_list = tf.train.Int64List(value = [label[i]]))
                }
            )
            example = tf.train.Example(features = features)
            serialized = example.SerializeToString()
            writer.write(serialized)


# 将不定长样本padding补0成定长
def padding(data, maxlen=10):
    for i in range(len(data)):
        data[i] = np.hstack([data[i], np.zeros((maxlen - len(data[i])))])


lens = np.random.randint(low=3, high=10, size=(10,))
data = [np.arange(l) for l in lens]
padding(data)
label = [0, 0, 0, 0, 0, 1, 1, 1, 1, 1]

save_tfrecords(data, label, "data.tfrecords")
print(data)
print()

tf_data = data[0]
print(tf_data)
tf_data = bytes(tf_data)
print(tf_data)

print(list(tf_data))
# b2 = bytes(tf_data,encoding='utf8')  #必须制定编码格式
print(type(tf_data))

#
num = np.array(1)
num_byte = bytes(num,encoding='utf8')
print(list(num_byte))