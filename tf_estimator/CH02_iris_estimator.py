# encoding: utf-8
'''
@author: linwenxing
@contact: linwx.mail@gmail.com
'''

import argparse
import tensorflow as tf

from  tf_estimator import iris_data


parser = argparse.ArgumentParser()
parser.add_argument('--batch_size', default=100, type=int, help='batch size')
parser.add_argument('--train_steps', default=1000, type=int,
                    help='number of training steps')

def main(argv):
    args = parser.parse_args(argv[1:])

    # Fetch the data
    (train_x, train_y), (test_x, test_y) = iris_data.load_data()

    # 定义特征列
    my_feature_columns = []
    #key ：'SepalLength', 'SepalWidth','PetalLength', 'PetalWidth',
    for key in train_x.keys():
        my_feature_columns.append(tf.feature_column.numeric_column(key=key))

    my_checkpointing_config = tf.estimator.RunConfig(
        save_checkpoints_secs=30,  # Save checkpoints 每30秒
        keep_checkpoint_max=10,  # 保存10个
    )

    # Build 2 hidden layer DNN with 10, 10 units respectively.
    classifier = tf.estimator.DNNClassifier(
        feature_columns=my_feature_columns,
        # Two hidden layers of 10 nodes each.
        hidden_units=[10, 10],
        # The model must choose between 3 classes.
        n_classes=3,
        # checkpoint 存储路径
        model_dir='models/iris',
        # checkpointing相关配置
        config=my_checkpointing_config)

    # 第一次调用 Estimator 的 train 方法时，TensorFlow 会将一个检查点保存到 model_dir 中。
    # 随后每次调用 Estimator 的 train、evaluate 或 predict 方法时：
    # 一旦存在检查点，TensorFlow 就会在您每次调用 train()、evaluate() 或 predict() 时重建模型。
    classifier.train(
        input_fn=lambda:iris_data.train_input_fn(train_x, train_y,
                                                 args.batch_size),
        steps=args.train_steps)

    # Evaluate the model.
    eval_result = classifier.evaluate(
        input_fn=lambda:iris_data.eval_input_fn(test_x, test_y,
                                                args.batch_size))

    print('\nTest set accuracy: {accuracy:0.3f}\n'.format(**eval_result))

    # Generate predictions from the model
    expected = ['Setosa', 'Versicolor', 'Virginica']
    predict_x = {
        'SepalLength': [5.1, 5.9, 6.9],
        'SepalWidth': [3.3, 3.0, 3.1],
        'PetalLength': [1.7, 4.2, 5.4],
        'PetalWidth': [0.5, 1.5, 2.1],
    }

    predictions = classifier.predict(
        input_fn=lambda:iris_data.eval_input_fn(predict_x,
                                                labels=None,
                                                batch_size=args.batch_size))

    template = ('\nPrediction is "{}" ({:.1f}%), expected "{}"')

    for pred_dict, expec in zip(predictions, expected):
        class_id = pred_dict['class_ids'][0]
        probability = pred_dict['probabilities'][class_id]

        print(template.format(iris_data.SPECIES[class_id],
                              100 * probability, expec))


if __name__ == '__main__':
    tf.logging.set_verbosity(tf.logging.INFO)
    tf.app.run(main)