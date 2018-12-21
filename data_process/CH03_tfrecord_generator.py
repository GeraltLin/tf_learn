# encoding: utf-8
'''
@author: linwenxing
@contact: linwx.mail@gmail.com
'''
import tensorflow as tf
import numpy as np

def save_tfrecords(data, label, desfile):
    with tf.python_io.TFRecordWriter(desfile) as writer:
        for i in range(len(data)):
            features = tf.train.Features(
                feature = {
                    "data":tf.train.Feature(bytes_list = tf.train.BytesList(value = [data[i].astype(np.float64).tostring()])),
                    "label":tf.train.Feature(float_list = tf.train.FloatList(value = [label[i]]))
                }
            )
            example = tf.train.Example(features = features)
            serialized = example.SerializeToString()
            writer.write(serialized)


with np.load("boston_housing.npz") as data:
    x = data['x']
    y = data['y']
#print([x[0]])
save_tfrecords(list(x),list(y),'boston_housing.tfrecords')
print(list(y)[0])