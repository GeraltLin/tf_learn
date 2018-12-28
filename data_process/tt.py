#coding: utf-8
"""
@author: linwenxing
@contact: linwx.mail@gmail.com
"""
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

def padding(data, maxlen=10):
    for i in range(len(data)):
        data[i] = np.hstack([data[i], np.zeros((maxlen-len(data[i])))])



with np.load("boston_housing.npz") as data:
    x = data['x']
    y = data['y']
# save_tfrecords(x,y,'boston_housing.tfrecords')



def _parse_function(example_proto):
  features = {"data": tf.FixedLenFeature((), tf.string),
              "label": tf.FixedLenFeature((), tf.float32)}
  parsed_features = tf.parse_single_example(example_proto, features)
  data = tf.decode_raw(parsed_features['data'], tf.float64)
  return data, parsed_features["label"]

def load_tfrecords(srcfile):
    sess = tf.Session()

    dataset = tf.data.TFRecordDataset(srcfile) # load tfrecord file
    dataset = dataset.map(_parse_function) # parse data into tensor

    iterator = dataset.make_one_shot_iterator()
    next_data = iterator.get_next()

    while True:
        try:
            data, label = sess.run(next_data)
            print (data)
            print (label)
        except tf.errors.OutOfRangeError:
            break
load_tfrecords(srcfile="boston_housing.tfrecords")
