# encoding: utf-8
'''
@author: linwenxing
@contact: linwx.mail@gmail.com
'''

import tensorflow as tf

features = {
    'sales' : [[5], [10], [8], [9]],
    'department': ['a', 'b', 'c', 'a']}

department_column = tf.feature_column.categorical_column_with_vocabulary_list(
        'department', ['a', 'b','c','d','e'])

department_column = tf.feature_column.indicator_column(department_column)

columns = [
    tf.feature_column.numeric_column('sales'),
    department_column
]

inputs = tf.feature_column.input_layer(features, columns)


var_init = tf.global_variables_initializer()
table_init = tf.tables_initializer()
sess = tf.Session()
sess.run((var_init, table_init))

print(sess.run(inputs))

"""

a b c d e sales
[[ 1.  0.  0.  0.  0.  5.]
 [ 0.  1.  0.  0.  0. 10.]
 [ 0.  0.  1.  0.  0.  8.]
 [ 1.  0.  0.  0.  0.  9.]]

"""
