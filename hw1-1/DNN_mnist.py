# -*- coding: utf-8 -*-
"""
Created on Sun Sep 23 05:25:13 2018

@author: Jimmy
"""

import tensorflow as tf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
tf.reset_default_graph()


mnist = tf.keras.datasets.mnist
(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train = (np.reshape(x_train, (60000,784,)).astype(np.float32))/255
x_test = (np.reshape(x_test, (10000,784,)).astype(np.float32))/255
y_train = tf.keras.utils.to_categorical(y_train, num_classes = 10)
y_test = tf.keras.utils.to_categorical(y_test, num_classes = 10)

batch_size = 16
epoch = 20

train_data = tf.placeholder(dtype=tf.float32, shape = (60000,784))
train_label = tf.placeholder(dtype=tf.float32, shape = (60000,10))


train_dataset = tf.data.Dataset.from_tensor_slices((train_data, train_label))
train_dataset = train_dataset.batch(batch_size)
train_dataset = train_dataset.repeat()
train_dataset = train_dataset.shuffle(30000)
train_iter = train_dataset.make_initializable_iterator()

x, y = train_iter.get_next()

layer1 = tf.layers.dense(x, units=100)
layer1 = tf.nn.relu(layer1)
layer2 = tf.layers.dense(layer1, units=100)
layer2 = tf.nn.relu(layer2)
layer3 = tf.layers.dense(layer2, units=100)
layer3 = tf.nn.relu(layer3)
output_layer = tf.layers.dense(layer3, units=10)
output_layer2 = tf.nn.softmax(output_layer)
prediction = tf.argmax(output_layer2, 1)

#=========================
#model2
layer1_model2 = tf.layers.dense(x, units=100)
layer1_model2 = tf.nn.relu(layer1_model2)
layer2_model2 = tf.layers.dense(layer1_model2, units=191)
layer2_model2 = tf.nn.relu(layer2_model2)
output1_model2 = tf.layers.dense(layer2_model2, units = 10)
output2_model2 = tf.nn.softmax(output1_model2)
prediction_model2 =  tf.argmax(output2_model2, 1)
#=========================

correct_ans = tf.argmax(y, 1)
acc_model1 = tf.reduce_mean( tf.cast( tf.equal(correct_ans, prediction), dtype = tf.float32))

acc_model2 = tf.reduce_mean( tf.cast( tf.equal(correct_ans, prediction_model2), dtype = tf.float32))

loss_model1 = tf.losses.softmax_cross_entropy(y, output_layer)
loss_model2 = tf.losses.softmax_cross_entropy(y, output1_model2)

training1 = tf.train.AdamOptimizer()
train_op1 = training1.minimize(loss_model1)
training2 = tf.train.AdamOptimizer()
train_op2 = training2.minimize(loss_model2)

model1_acc_list = []
model1_loss_list = []
model2_acc_list = []
model2_loss_list = []
with tf.Session() as sess:
    print("initialize...")
    sess.run(tf.global_variables_initializer())
    sess.run(tf.local_variables_initializer())
    sess.run(train_iter.initializer, feed_dict={train_data : x_train, train_label : y_train})
    print("Start Training")
    for i in range(epoch):
        training_loss1 = 0
        training_acc1 = 0
        training_loss2 = 0
        training_acc2 = 0
        for j in range(60000//batch_size):
            batch_loss1 = sess.run(loss_model1)
            batch_loss2 = sess.run(loss_model2)
            sess.run(train_op1)
            sess.run(train_op2)
            batch_acc1 = sess.run(acc_model1)
            batch_acc2 = sess.run(acc_model2)
            training_loss1 += batch_loss1
            training_loss2 += batch_loss2
            training_acc1 += batch_acc1
            training_acc2 += batch_acc2
        training_loss1 /= (60000//batch_size)
        training_loss2 /= (60000//batch_size)
        training_acc1 /= (60000//batch_size)
        training_acc2 /= (60000//batch_size)
        #sess.run(acc_op)
        print("epoch: ",i)
        print("model1(deep):")
        print(training_acc1)
        print(training_loss1)
        model1_acc_list.append(training_acc1)
        model1_loss_list.append(training_loss1)
        print("model2(shallow):")
        print(training_acc2)
        print(training_loss2)
        model2_acc_list.append(training_acc2)
        model2_loss_list.append(training_loss2)
        #sess.run(tf.local_variables_initializer())

epoch_num = list(range(epoch))    
plt.plot(epoch_num, model1_acc_list, label='deep')
plt.plot(epoch_num, model2_acc_list, label='shallow')
plt.xlabel("epoch", fontsize=16)
plt.ylabel("acc", fontsize=16)
plt.legend(loc = 'lower right', fontsize=16)
plt.show()
plt.plot(epoch_num, model1_loss_list, label='deep')
plt.plot(epoch_num, model2_loss_list, label='shallow')
plt.xlabel("epoch", fontsize=16)
plt.ylabel("cross_entropy", fontsize=16)
plt.legend(loc = 'upper right', fontsize=16)
plt.show()
    
