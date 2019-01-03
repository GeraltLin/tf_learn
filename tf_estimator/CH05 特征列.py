# encoding: utf-8
'''
@author: linwenxing
@contact: linwx.mail@gmail.com
'''

import tensorflow as tf
from tf_estimator import iris_data

predict_x = {
        'SepalLength': [[5.1]],
        'SepalWidth': [[3.3]],
        'PetalLength': [[1.7]],
        'PetalWidth': [[0.5]],
    }


(train_x, train_y), (test_x, test_y) = iris_data.load_data()

my_feature_columns = []
    # key ：'SepalLength', 'SepalWidth','PetalLength', 'PetalWidth',
for key in train_x.keys():
    my_feature_columns.append(tf.feature_column.numeric_column(key=key))



inputs = tf.feature_column.input_layer(predict_x,my_feature_columns)


var_init = tf.global_variables_initializer()
table_init = tf.tables_initializer()
sess = tf.Session()
sess.run((var_init, table_init))

print(sess.run(inputs))
#[[1.7 0.5 5.1 3.3]]