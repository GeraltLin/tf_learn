# encoding: utf-8
'''
@author: linwenxing
@contact: linwx.mail@gmail.com
'''

from sklearn import datasets
import tensorflow as tf
import numpy as np
from sklearn.model_selection import train_test_split

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
label = np.zeros(232)
label[-1:-50:-1]=1
# label = to_categorical(label,num_classes=2)

data = one_data
data = list(map(lambda x:x.reshape(-1,),data))


# print(len(data))
x_train, x_test, y_train, y_test = train_test_split(data, label, test_size=0.3, random_state=0)
y_train = y_train.reshape(1,-1)[0]
y_test = y_test.reshape(1,-1)[0]
from collections import Counter
print(Counter(y_train))
from sklearn.utils.class_weight import compute_class_weight
print(np.unique(y_train))
class_weights = compute_class_weight('balanced', np.unique(y_train), y_train)

print(class_weights)
X = tf.placeholder(dtype=tf.float32,shape=[None,64])
Y = tf.placeholder(dtype=tf.int32)# print(Y)
y_one_hot = tf.one_hot(Y, 2)
y_one_hot = tf.cast(y_one_hot,tf.float32)


h1 = tf.layers.dense(inputs=X,units=10,activation='sigmoid')
p = tf.layers.dense(inputs=h1,units=2)






cost = tf.losses.sigmoid_cross_entropy(multi_class_labels=y_one_hot,logits=p)
w = np.array([1,1],dtype='float32').reshape([2,1])
w_ls=tf.Variable(w,name="w_ls",trainable=False)
w_temp = tf.matmul(y_one_hot, w_ls) #代价敏感因子，w是权重项链表
loss=tf.reduce_mean(tf.multiply(cost,w_temp))  #代价敏感下的交叉熵损失

optimizer = tf.train.AdamOptimizer(learning_rate = 0.01).minimize(loss)


correct_prediction = tf.equal(tf.argmax(y_one_hot,1),tf.argmax(p,1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction,tf.float32))


init = tf.global_variables_initializer()
training_epochs = 100
display_step = 1



# 启动session
with tf.Session() as sess:
    sess.run(init)


    # Fit all training data
    for epoch in range(training_epochs):
        # for (x, y) in zip(x_train, y_train):
        sess.run(optimizer, feed_dict={X: x_train, Y: y_train})
        # print(y_)
        #显示训练中的详细信息
        if epoch % display_step == 0:
            acc = sess.run(accuracy, feed_dict={X: x_test, Y:y_test})

            print('epoch{}:{}'.format(epoch,acc))




