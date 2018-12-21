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
                    "data":tf.train.Feature(bytes_list = tf.train.BytesList(value = [data[i].astype(np.float64).tostring()])),
                    "label":tf.train.Feature(int64_list = tf.train.Int64List(value = [label[i]]))
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


def _parse_function(example_proto):
    features = {"data": tf.FixedLenFeature((), tf.string),
                "label": tf.FixedLenFeature((), tf.int64)}
    parsed_features = tf.parse_single_example(example_proto, features)
    data = tf.decode_raw(parsed_features['data'], tf.float32)
    return data, parsed_features["label"]


def load_tfrecords(srcfile):
    sess = tf.Session()

    dataset = tf.data.TFRecordDataset(srcfile)  # load tfrecord file
    dataset = dataset.map(_parse_function)  # parse data into tensor
    # dataset = dataset.repeat(2)  # repeat for 2 epoches
    # dataset = dataset.batch(5)  # set batch_size = 5

    iterator = dataset.make_one_shot_iterator()
    next_data = iterator.get_next()

    while True:
        try:
            data, label = sess.run(next_data)
            print (data)
            print(len(data))
            print (label)
        except tf.errors.OutOfRangeError:
            break

load_tfrecords('data.tfrecords')