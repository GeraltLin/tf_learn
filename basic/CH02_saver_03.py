# encoding: utf-8
'''
@author: linwenxing
@contact: linwx.mail@gmail.com
'''

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt



saver = tf.train.import_meta_graph("../LR_model/LR.ckpt-151.meta")
config = tf.ConfigProto(log_device_placement=True,allow_soft_placement
=True)
# 启动session
config.gpu_options.allow_growth = True
with tf.Session(config=config) as sess:
    kpt = tf.train.latest_checkpoint('../LR_model')
    if kpt != None:
        saver.restore(sess, kpt)
        graph = tf.get_default_graph()
        tensor_list = [tensor.name for tensor in tf.get_default_graph().as_graph_def().node if tensor.op=="Placeholder"]
    z = graph.get_tensor_by_name("fc/z:0")
    x = graph.get_tensor_by_name("x:0")
    feed_dict = {x:3}
    print(sess.run(z,feed_dict=feed_dict))
    print(sess.run(z,feed_dict=feed_dict))

    # for tensor in tensor_list:
    #     print(tensor)

#
# import os
# from tensorflow.python import pywrap_tensorflow
#
# checkpoint_path = os.path.join('../LR_model', "LR.ckpt-151")
# reader = pywrap_tensorflow.NewCheckpointReader(checkpoint_path)
# var_to_shape_map = reader.get_variable_to_shape_map()
# for key in var_to_shape_map:
#     print("tensor_name: ", key)