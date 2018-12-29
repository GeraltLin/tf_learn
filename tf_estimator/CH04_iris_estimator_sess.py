# encoding: utf-8
'''
@author: linwenxing
@contact: linwx.mail@gmail.com
'''

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import tensorflow as tf

from tf_estimator import iris_data

parser = argparse.ArgumentParser()
parser.add_argument('--batch_size', default=100, type=int, help='batch size')
parser.add_argument('--train_steps', default=1000, type=int,
                    help='number of training steps')
init_checkpoint = 'models/iris'

def create_model(features, params):
    net = tf.feature_column.input_layer(features, params['feature_columns'])
    for units in params['hidden_units']:
        net = tf.layers.dense(net, units=units, activation=tf.nn.relu)

    # Compute logits (1 per class).
    logits = tf.layers.dense(net, params['n_classes'], activation=None)

    # Compute predictions.
    predicted_classes = tf.argmax(logits, 1)


    return logits,predicted_classes

def create_predict_model(features, params):
    net = features
    for units in params['hidden_units']:
        net = tf.layers.dense(net, units=units, activation=tf.nn.relu)

    # Compute logits (1 per class).
    logits = tf.layers.dense(net, params['n_classes'], activation=None)

    # Compute predictions.
    predicted_classes = tf.argmax(logits, 1)


    return logits,predicted_classes


def my_model(features, labels, mode, params):
    """DNN with three hidden layers and learning_rate=0.1."""

    logits, predicted_classes = create_model(features, params)




    if mode == tf.estimator.ModeKeys.PREDICT:
        predictions = {
            'class_ids': predicted_classes[:, tf.newaxis],
            'probabilities': tf.nn.softmax(logits),
            'logits': logits,
        }
        return tf.estimator.EstimatorSpec(mode, predictions=predictions)

    # Compute evaluation metrics.
    loss = tf.losses.sparse_softmax_cross_entropy(labels=labels, logits=logits)
    accuracy = tf.metrics.accuracy(labels=labels,
                                   predictions=predicted_classes,
                                   name='acc_op')
    metrics = {'accuracy': accuracy}
    tf.summary.scalar('accuracy', accuracy[1])

    if mode == tf.estimator.ModeKeys.EVAL:
        return tf.estimator.EstimatorSpec(
            mode, loss=loss, eval_metric_ops=metrics)

    # Create training op.
    assert mode == tf.estimator.ModeKeys.TRAIN

    optimizer = tf.train.AdagradOptimizer(learning_rate=0.1)
    train_op = optimizer.minimize(loss, global_step=tf.train.get_global_step())
    return tf.estimator.EstimatorSpec(mode, loss=loss, train_op=train_op)




def main(argv):
    args = parser.parse_args(argv[1:])

    # Fetch the data
    (train_x, train_y), (test_x, test_y) = iris_data.load_data()

    # 定义特征列
    my_feature_columns = []
    # key ：'SepalLength', 'SepalWidth','PetalLength', 'PetalWidth',
    for key in train_x.keys():
        my_feature_columns.append(tf.feature_column.numeric_column(key=key))

    my_checkpointing_config = tf.estimator.RunConfig(
        save_checkpoints_secs=30,  # Save checkpoints 每30秒
        keep_checkpoint_max=10,  # 保存10个
    )

    # Build 2 hidden layer DNN with 10, 10 units respectively.
    classifier = tf.estimator.Estimator(
        model_fn=my_model,
        params={
            'feature_columns': my_feature_columns,
            # Two hidden layers of 10 nodes each.
            'hidden_units': [10, 10],
            # The model must choose between 3 classes.
            'n_classes': 3,
        },
        model_dir='models/iris',
        config=my_checkpointing_config)

    # Train the Model.
    classifier.train(
        input_fn=lambda:iris_data.train_input_fn(train_x, train_y, args.batch_size),
        steps=args.train_steps)

    # Evaluate the model.
    eval_result = classifier.evaluate(
        input_fn=lambda:iris_data.eval_input_fn(test_x, test_y, args.batch_size))

    print('\nTest set accuracy: {accuracy:0.3f}\n'.format(**eval_result))

    # Generate predictions from the model
    expected = ['Setosa', 'Versicolor', 'Virginica']
    predict_x = {
        'SepalLength': [[5.1]],
        'SepalWidth': [[3.3]],
        'PetalLength': [[1.7]],
        'PetalWidth': [[0.5]],
    }

    gpu_config = tf.ConfigProto()
    gpu_config.gpu_options.allow_growth = True
    sess = tf.Session(config=gpu_config)

    graph = tf.get_default_graph()
    with graph.as_default():
        print("going to restore checkpoint")
        # sess.run(tf.global_variables_initializer())
        SepalLength = tf.placeholder(shape=[None,1],dtype=tf.float32)
        SepalWidth = tf.placeholder(shape=[None,1],dtype=tf.float32)
        PetalLength = tf.placeholder(shape=[None,1],dtype=tf.float32)
        PetalWidth = tf.placeholder(shape=[None,1],dtype=tf.float32)
        features = tf.concat([SepalLength, SepalWidth, PetalLength, PetalWidth], axis=-1)

        params = {
            # Two hidden layers of 10 nodes each.
            'hidden_units': [10, 10],
            # The model must choose between 3 classes.
            'n_classes': 3,
        }

        logits, predicted_classes = create_predict_model(features, params)
        saver = tf.train.Saver()
        saver.restore(sess, tf.train.latest_checkpoint(init_checkpoint))

    with graph.as_default():
        feed_dict = {SepalLength: predict_x['SepalLength'], SepalWidth: predict_x['SepalWidth'], PetalLength: predict_x['PetalLength'],
                     PetalWidth: predict_x['PetalWidth']}
        predicted_classes = sess.run([predicted_classes], feed_dict)

        print(predicted_classes)





    predictions = classifier.predict(
        input_fn=lambda:iris_data.eval_input_fn(predict_x,
                                                labels=None,
                                                batch_size=args.batch_size))

    for pred_dict, expec in zip(predictions, expected):
        template = ('\nPrediction is "{}" ({:.1f}%), expected "{}"')

        class_id = pred_dict['class_ids'][0]
        probability = pred_dict['probabilities'][class_id]
        print(class_id)
        print(template.format(iris_data.SPECIES[class_id],
                              100 * probability, expec))


if __name__ == '__main__':
    tf.logging.set_verbosity(tf.logging.INFO)
    tf.app.run(main)