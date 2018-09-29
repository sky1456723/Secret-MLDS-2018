from __future__ import print_function

import ssl
ssl._create_default_https_context = ssl._create_unverified_context

import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("/tmp/data/", one_hot=False)

import matplotlib.pyplot as plt

### PARAMETERS ###

learning_rate = 0.001
num_steps = 10 # 1 step = 1 update
num_watch = 30 # program stops after 30 train sessions (300 updates)
batch_size = 64

num_classes = 10 # MNIST total classes (0-9 digits)
dropout = 0.25

### METHODS ###

def cnn1(input_dict, n_classes, dropout, is_training):
    # Reuse should not be determined by user.
    with tf.variable_scope('ConvNet', reuse=tf.AUTO_REUSE):
        x = tf.reshape(input_dict['images'], \
            shape = [-1, 28, 28, 1])

        ### GEOMETRY 1
        ### 28x28x32 + 14x14x32 + 14x14x64 + 7x7x64 + 1024 + 10 = 48074

        conv1 = tf.layers.max_pooling2d( \
                tf.layers.conv2d(x, 32, 5, activation = tf.nn.relu), \
            2, 2)
        conv2 = tf.layers.max_pooling2d( \
                tf.layers.conv2d(conv1, 64, 3, activation = tf.nn.relu), \
            2, 2)
        fc = tf.layers.dropout( \
                tf.layers.dense(tf.contrib.layers.flatten(conv2), 1024), \
            rate = dropout, training = is_training)
        return tf.layers.dense(fc, n_classes)

def cnn2(input_dict, n_classes, dropout, is_training):
    with tf.variable_scope('ConvNet', reuse=tf.AUTO_REUSE):
        x = tf.reshape(input_dict['images'], \
            shape = [-1, 28, 28, 1])

        ### GEOMETRY 2
        ### 28x28x28 + 14x14x28 + 14x14x56 + 7x7x56 + 7x7x84 + 4x4x84 + 1464 + 10 = 48074

        conv1 = tf.layers.max_pooling2d( \
                tf.layers.conv2d(x, 28, 5, activation = tf.nn.relu), \
            2, 2)
        conv2 = tf.layers.max_pooling2d( \
                tf.layers.conv2d(x, 56, 3, activation = tf.nn.relu), \
            2, 2)
        conv3 = tf.layers.max_pooling2d( \
                tf.layers.conv2d(x, 84, 3, activation = tf.nn.relu), \
            2, 2, padding = 'same')
        fc = tf.layers.dropout( \
                tf.layers.dense(tf.contrib.layers.flatten(conv2), 1464), \
            rate = dropout, training = is_training)
        return tf.layers.dense(fc, n_classes)

def cnn3(input_dict, n_classes, dropout, is_training):
    with tf.variable_scope('ConvNet', reuse=tf.AUTO_REUSE):
        x = tf.reshape(input_dict['images'], \
            shape = [-1, 28, 28, 1])

        ### GEOMETRY 3
        ### 28*28*61 + 1024 + 10 = 48858

        conv = tf.layers.max_pooling2d( \
                tf.layers.conv2d(x, 61, 5, activation = tf.nn.relu), \
            2, 2)
        fc = tf.layers.dropout( \
                tf.layers.dense(tf.contrib.layers.flatten(conv), 1024), \
            rate = dropout, training = is_training)
        return tf.layers.dense(fc, n_classes)

def model_fn(features, labels, mode, params):
    # Model behaves differently if it is being trained (dropout is not
    # applied when we evaluate this model)
    # logits_test is called only twice: upon model building & evaluating
    if params['model'] == 1:
        logits_train = cnn1(features, num_classes, dropout, is_training=True)
        logits_test = cnn1(features, num_classes, dropout, is_training=False)
    elif params['model'] == 2:
        logits_train = cnn2(features, num_classes, dropout, is_training=True)
        logits_test = cnn2(features, num_classes, dropout, is_training=False)
    else:
        logits_train = cnn3(features, num_classes, dropout, is_training=True)
        logits_test = cnn3(features, num_classes, dropout, is_training=False)

    pred_classes = tf.argmax(logits_test, axis=1)

    # mode is passed as tf.estimator.ModeKeys.PREDICT only if
    # tf.estimator.Estimator(...).predict(input_fn) is evoked
    # if mode == tf.estimator.ModeKeys.PREDICT:
    #     return tf.estimator.EstimatorSpec(mode, predictions=pred_classes)

    loss_op = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(
        logits=logits_train, labels=tf.cast(labels, dtype=tf.int32)))
    train_op = tf.train.AdamOptimizer(learning_rate = learning_rate). \
        minimize(loss_op, global_step=tf.train.get_global_step())

    # metrics provide optional evaluations besides loss
    acc_op = tf.metrics.accuracy(labels = labels, predictions = pred_classes)

    return tf.estimator.EstimatorSpec(
        mode=mode,
        predictions=pred_classes,
        loss=loss_op,
        train_op=train_op,
        eval_metric_ops={'accuracy': acc_op})

def train_session(model):
    input_fn = tf.estimator.inputs.numpy_input_fn(
        x = {'images': mnist.train.images}, y = mnist.train.labels,
        batch_size = batch_size, shuffle = True)
    model.train(input_fn, steps = num_steps)

def evaluate_session(model):
    input_fn = tf.estimator.inputs.numpy_input_fn(
        x = {'images': mnist.test.images}, y = mnist.test.labels,
        batch_size = batch_size, shuffle = False)
    e = model.evaluate(input_fn)
    return e['loss'], e['accuracy'] * 100

### MAIN BLOCK ###

model1 = tf.estimator.Estimator(model_fn, params = {'model': 1})
model2 = tf.estimator.Estimator(model_fn, params = {'model': 2})
model3 = tf.estimator.Estimator(model_fn, params = {'model': 3})

m1l = []
m1a = []
m2l = []
m2a = []
m3l = []
m3a = []

print("Training model 1...")
for i in range(num_watch):
    train_session(model1)
    l, a = evaluate_session(model1)
    m1l.append(l)
    m1a.append(a)
    print("Step: {}, Loss: {:.3f}, Accuracy: {:.2f}".format( \
        i * num_steps, l, a))

print("Training model 2...")
for i in range(num_watch):
    train_session(model2)
    l, a = evaluate_session(model2)
    m2l.append(l)
    m2a.append(a)
    print("Step: {}, Loss: {:.3f}, Accuracy: {:.2f}".format( \
        i * num_steps, l, a))

print("Training model 3...")
for i in range(num_watch):
    train_session(model3)
    l, a = evaluate_session(model3)
    m3l.append(l)
    m3a.append(a)
    print("Step: {}, Loss: {:.3f}, Accuracy: {:.2f}".format( \
        i * num_steps, l, a))

print("Finished training.")

### PLOTTING ###

axis = [i for i in range(0, num_steps * num_watch, num_steps)]

# loss
plt.figure(1)
plt.plot(axis, m1l, label = 'Geo 1')
plt.plot(axis, m2l, label = 'Geo 2')
plt.plot(axis, m3l, label = 'Geo 3')
plt.xlabel('Steps')
plt.ylabel('Loss (cross entropy)')
plt.legend()
plt.savefig('Loss.png')

# accuracy
plt.figure(2)
plt.plot(axis, m1a, label = 'Geo 1')
plt.plot(axis, m2a, label = 'Geo 2')
plt.plot(axis, m3a, label = 'Geo 3')
plt.xlabel('Steps')
plt.ylabel('Accuracy (%)')
plt.legend()
plt.savefig('Accuracy.png')
plt.show()
