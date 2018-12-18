
# encoding: utf-8
'''
@author: linwenxing
@contact: linwx.mail@gmail.com
'''

from sklearn import datasets
import tensorflow as tf
import numpy as np
from sklearn.model_selection import train_test_split

def to_categorical(y, num_classes=None, dtype='float32'):
    """Converts a class vector (integers) to binary class matrix.

    E.g. for use with categorical_crossentropy.

    # Arguments
        y: class vector to be converted into a matrix
            (integers from 0 to num_classes).
        num_classes: total number of classes.
        dtype: The data type expected by the input, as a string
            (`float32`, `float64`, `int32`...)

    # Returns
        A binary matrix representation of the input. The classes axis
        is placed last.
    """
    y = np.array(y, dtype='int')
    input_shape = y.shape
    if input_shape and input_shape[-1] == 1 and len(input_shape) > 1:
        input_shape = tuple(input_shape[:-1])
    y = y.ravel()
    if not num_classes:
        num_classes = np.max(y) + 1
    n = y.shape[0]
    categorical = np.zeros((n, num_classes), dtype=dtype)
    categorical[np.arange(n), y] = 1
    output_shape = input_shape + (num_classes,)
    categorical = np.reshape(categorical, output_shape)
    return categorical


digits = datasets.load_digits()
label = digits.target
one_label = []
seven_label = []
one_data = []
seven_data = []
for index,label in enumerate(label):
    if label==1:
        one_label.append(index)
    elif label==7:
        seven_label.append(index)
print(one_label)
print(seven_label)

for index,data in enumerate(digits.images):
    if index in one_label:
        one_data.append(data)
    elif index in seven_label:
        seven_data.append(data)

# print(len(one_data))
# print(len(seven_data))
seven_data = seven_data[:50]
one_data.extend(seven_data)
label = np.zeros(232,dtype=np.uint8)
label[-1:-50:-1]=1
label = to_categorical(label,num_classes=2)
data = one_data
data = list(map(lambda x:x.reshape(-1,),data))


# print(len(data))
x_train, x_test, y_train, y_test = train_test_split(data, label, test_size=0.3, random_state=0)
y_train = y_train.reshape(-1,2)
y_test = y_test.reshape(-1,2)
X = tf.placeholder(dtype=tf.float32,shape=[None,64])
Y = tf.placeholder(dtype=tf.float32,shape=[None,2])

h1 = tf.layers.dense(inputs=X,units=10,activation='sigmoid')
p = tf.layers.dense(inputs=h1,units=2)
loss = tf.losses.sigmoid_cross_entropy(multi_class_labels=Y,logits=p)
tf.summary.scalar(name='loss_value',tensor=loss)


optimizer = tf.train.AdamOptimizer(learning_rate = 0.01).minimize(loss)


correct_prediction = tf.equal(tf.argmax(Y,1),tf.argmax(p,1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction,tf.float32))
tf.summary.scalar(name='acc_value',tensor=accuracy)


init = tf.global_variables_initializer()
training_epochs = 100
display_step = 1



# 启动session
with tf.Session() as sess:
    sess.run(init)
    merge_op  =  tf.summary.merge_all()
    summary_writer = tf.summary.FileWriter(logdir='../tf_log',graph=sess.graph)
    # Fit all training data
    for epoch in range(training_epochs):
        # for (x, y) in zip(x_train, y_train):
        sess.run(optimizer, feed_dict={X: x_train, Y: y_train})

        merge_op_write = sess.run(merge_op,feed_dict={X: x_train, Y: y_train})
        summary_writer.add_summary(merge_op_write,global_step=epoch)

        #显示训练中的详细信息
        if epoch % display_step == 0:
            acc = sess.run(accuracy, feed_dict={X: x_train, Y:y_train})

            print('epoch{}:{}'.format(epoch,acc))




