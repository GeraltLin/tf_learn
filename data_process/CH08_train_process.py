# encoding: utf-8
'''
@author: linwenxing
@contact: linwx.mail@gmail.com
'''

# filenames = ["/var/data/file1.tfrecord", "/var/data/file2.tfrecord"]
# dataset = tf.data.TFRecordDataset(filenames)
# dataset = dataset.map(...)
# dataset = dataset.shuffle(buffer_size=10000)
# dataset = dataset.batch(32)
# dataset = dataset.repeat(num_epochs)
# iterator = dataset.make_one_shot_iterator()
#
# next_example, next_label = iterator.get_next()
# loss = model_function(next_example, next_label)
#
# training_op = tf.train.AdagradOptimizer(...).minimize(loss)
#
# with tf.train.MonitoredTrainingSession(...) as sess:
#   while not sess.should_stop():
#     sess.run(training_op)

import tensorflow as tf
tf.estimator.DNNClassifier