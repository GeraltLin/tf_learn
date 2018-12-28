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
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

def save_tfrecords(data, label, desfile):
    with tf.python_io.TFRecordWriter(desfile) as writer:

        for i in range(len(data)):
            feature = {"data": _bytes_feature(data[i].astype(np.float64).tostring()),
                       "label": _float_feature(label[i])}
            example = tf.train.Example(features=tf.train.Features(feature=feature))
            serialized = example.SerializeToString()
            writer.write(serialized)

with np.load("boston_housing.npz") as data:
    x = data['x']
    y = data['y']
save_tfrecords(x,y,'boston_housing.tfrecords')

