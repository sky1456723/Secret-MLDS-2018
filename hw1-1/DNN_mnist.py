# -*- coding: utf-8 -*-
"""
Created on Sun Sep 23 05:25:13 2018

@author: Jimmy
"""

import tensorflow as tf
import numpy as np
import pandas as pd

tf.reset_default_graph()


mnist = tf.keras.datasets.mnist
(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train = (np.reshape(x_train, (60000,784,)).astype(np.float32))/255
x_test = (np.reshape(x_test, (10000,784,)).astype(np.float32))/255
y_train = tf.one_hot(y_train, depth = 10, dtype = tf.int64)
y_test = tf.one_hot(y_test, depth = 10, dtype = tf.int64)

batch_size = 16
epoch = 20

train_dataset = tf.data.Dataset.from_tensor_slices((x_train, y_train))
train_dataset = train_dataset.batch(batch_size)
train_dataset = train_dataset.repeat()
train_iter = train_dataset.make_initializable_iterator()

#x_train = tf.train.batch(list(x_train), batch_size)
#y_train = tf.train.batch(y_train, batch_size)
#x_test = tf.train.batch(x_test, batch_size)
#y_test = tf.train.batch(y_test, batch_size)

x, y = train_iter.get_next()

layer1 = tf.layers.dense(x, units=100)
layer1 = tf.nn.relu(layer1)
layer2 = tf.layers.dense(layer1, units=100)
layer2 = tf.nn.relu(layer2)
layer3 = tf.layers.dense(layer2, units=100)
layer3 = tf.nn.relu(layer3)
output_layer = tf.layers.dense(layer3, units=10)
output_layer = tf.nn.softmax(output_layer)

prediction = tf.argmax(output_layer,1)
equal = tf.equal(prediction, tf.argmax(y,1) )
acc = tf.reduce_mean(tf.cast(equal, tf.float32))

loss = tf.losses.softmax_cross_entropy(y, output_layer)
training = tf.train.AdamOptimizer()
train_op = training.minimize(loss)

with tf.Session() as sess:
    print("initialize...")
    sess.run(tf.global_variables_initializer())
    sess.run(train_iter.initializer)
    print("Start")
    for i in range(epoch):
        training_loss = 0
        for j in range(60000//batch_size):
            batch_loss = sess.run(loss)
            batch_acc = sess.run(acc)
            print(batch_acc)
            sess.run(train_op)
            training_loss += batch_loss
        training_loss /= 60000//batch_size
        print("epoch: ",i)
        print(training_loss)
        


    