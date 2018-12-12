# coding: utf-8

# In[1]:

import pandas as pd
import numpy as np
import tensorflow as tf
from tfrecorder import TFrecorder


tfr = TFrecorder()
def input_fn_maker(path, data_info_path, shuffle=False, batch_size=1, epoch=1, padding=None):
    def input_fn():
        filenames = tfr.get_filenames(path=path, shuffle=shuffle)
        dataset = tfr.get_dataset(paths=filenames, data_info=data_info_path, shuffle=shuffle,
                                  batch_size=batch_size, epoch=epoch, padding=padding)
        iterator = dataset.make_one_shot_iterator()
        return iterator.get_next()

    return input_fn



flags = tf.app.flags
flags.DEFINE_string('saved_model_dir', 'mnist_model_cnn',
                    'Output directory for model and training stats.')
FLAGS = flags.FLAGS



padding_info = ({'image': [784, ], 'label': []})
test_input_fn = input_fn_maker('mnist_tfrecord/test/', 'mnist_tfrecord/data_info.csv', batch_size=512,
                               padding=padding_info)

def model_fn(features, mode):
    # reshape 784维的图片到28x28的平面表达，1为channel数
    features['image'] = tf.reshape(features['image'], [-1, 28, 28, 1])
    # shape: [None,28,28,1]
    conv1 = tf.layers.conv2d(
        inputs=features['image'],
        filters=32,
        kernel_size=[5, 5],
        padding="same",
        activation=tf.nn.relu,
        name='conv1')
    # shape: [None,28,28,32]
    pool1 = tf.layers.max_pooling2d(inputs=conv1, pool_size=[2, 2], strides=2, name='pool1')
    # shape: [None,14,14,32]
    conv2 = tf.layers.conv2d(
        inputs=pool1,
        filters=64,
        kernel_size=[5, 5],
        padding="same",
        activation=tf.nn.relu,
        name='conv2')
    # shape: [None,14,14,64]
    pool2 = tf.layers.max_pooling2d(inputs=conv2, pool_size=[2, 2], strides=2, name='pool2')
    # shape: [None,7,7,64]
    pool2_flat = tf.reshape(pool2, [-1, 7 * 7 * 64], name='pool2_flat')
    # shape: [None,3136]
    dense1 = tf.layers.dense(inputs=pool2_flat, units=1024, activation=tf.nn.relu, name='dense1')
    # shape: [None,1024]
    dropout = tf.layers.dropout(inputs=dense1, rate=0.4, training=mode == tf.estimator.ModeKeys.TRAIN)
    # shape: [None,1024]
    logits = tf.layers.dense(inputs=dropout, units=10, name='output')
    # shape: [None,10]
    predictions = {
        "image": features['image'],
        "conv1_out": conv1,
        "pool1_out": pool1,
        "conv2_out": conv2,
        "pool2_out": pool2,
        "pool2_flat_out": pool2_flat,
        "dense1_out": dense1,
        "logits": logits,
        "classes": tf.argmax(input=logits, axis=1),
        "labels": features['label'],
        "probabilities": tf.nn.softmax(logits, name="softmax_tensor")
    }
    if mode == tf.estimator.ModeKeys.PREDICT:
        return tf.estimator.EstimatorSpec(mode=mode, predictions=predictions)

    loss = tf.losses.sparse_softmax_cross_entropy(labels=features['label'], logits=logits)

    if mode == tf.estimator.ModeKeys.TRAIN:
        optimizer = tf.train.AdamOptimizer(learning_rate=1e-3)
        train_op = optimizer.minimize(
            loss=loss,
            global_step=tf.train.get_global_step())
        return tf.estimator.EstimatorSpec(mode=mode, loss=loss, train_op=train_op)
    eval_metric_ops = {"accuracy": tf.metrics.accuracy(labels=features['label'], predictions=predictions["classes"])}
    return tf.estimator.EstimatorSpec(mode=mode, loss=loss, eval_metric_ops=eval_metric_ops)



def get_estimator(config):
    '''Return the model as a Tensorflow Estimator object.'''
    return tf.estimator.Estimator(model_fn=model_fn, config=config)





# tensors_to_log = {"probabilities": "softmax_tensor",
#                   "probabilities": "softmax_tensor", }
#
#
# mnist_classifier = tf.estimator.Estimator(
#     model_fn=model_fn, model_dir="mnist_model_cnn")
#
# print("======================================================")
# predicts = list(mnist_classifier.predict(input_fn=test_input_fn))
#
# print("======================================================")

# predicts = list(mnist_classifier.predict(input_fn=test_input_fn))

config = tf.estimator.RunConfig()
config = config.replace(model_dir=FLAGS.saved_model_dir)
estimator = get_estimator(config)
predict_input_fn = test_input_fn
result = estimator.predict(input_fn=predict_input_fn)
for r in result:
    print(r)
print('==========================================')
result = estimator.predict(input_fn=predict_input_fn)
for r in result:
    print(r)