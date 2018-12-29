# encoding: utf-8
'''
@author: linwenxing
@contact: linwenxing@zbj.com
'''

"""
本文档介绍了 Estimator - 一种可极大地简化机器学习编程的高阶 TensorFlow API。Estimator 会封装下列操作：

训练
评估
预测
导出以供使用
您可以使用我们提供的预创建的 Estimator，也可以编写自定义 Estimator。
所有 Estimator（无论是预创建的还是自定义）都是基于 tf.estimator.Estimator 类的类


借助预创建的 Estimator，您能够在比基本 TensorFlow API 高级很多的概念层面上进行操作。
由于 Estimator 会为您处理所有“管道工作”，因此您不必再为创建计算图或会话而操心。
也就是说，预创建的 Estimator 会为您创建和管理 Graph 和 Session 对象。
此外，借助预创建的 Estimator，您只需稍微更改下代码，就可以尝试不同的模型架构。
例如，DNNClassifier 是一个预创建的 Estimator 类，它根据密集的前馈神经网络训练分类模型。

"""
###########################################################

"""

预创建的 Estimator 程序的结构
依赖预创建的 Estimator 的 TensorFlow 程序通常包含下列四个步骤：

编写一个或多个数据集导入函数。 例如，您可以创建一个函数来导入训练集，并创建另一个函数来导入测试集。每个数据集导入函数都必须返回两个对象：

一个字典，其中键是特征名称，值是包含相应特征数据的张量（或 SparseTensor）
一个包含一个或多个标签的张量
例如，以下代码展示了输入函数的基本框架：

def input_fn(dataset):
   ...  # manipulate dataset, extracting the feature dict and the label
   return feature_dict, label
"""
###########################################################

"""
定义特征列。 每个 tf.feature_column 都标识了特征名称、特征类型和任何输入预处理操作。
例如，以下代码段创建了三个存储整数或浮点数据的特征列。前两个特征列仅标识了特征的名称和类型。第三个特征列还指定了一个 lambda，该程序将调用此 lambda 来调节原始数据：

# Define three numeric feature columns.
population = tf.feature_column.numeric_column('population')
crime_rate = tf.feature_column.numeric_column('crime_rate')
median_education = tf.feature_column.numeric_column('median_education',
                    normalizer_fn=lambda x: x - global_education_mean)
"""
###########################################################
"""
实例化相关的预创建的 Estimator。 例如，下面是对名为 LinearClassifier 的预创建 Estimator 进行实例化的示例代码：

# Instantiate an estimator, passing the feature columns.
estimator = tf.estimator.LinearClassifier(
    feature_columns=[population, crime_rate, median_education],
    )
调用训练、评估或推理方法。例如，所有 Estimator 都提供训练模型的 train 方法。

# my_training_set is the function created in Step 1
estimator.train(input_fn=my_training_set, steps=2000)

"""