#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Sep 23 18:36:00 2018

@author: jimmy
"""

import tensorflow as tf
import numpy as np
import keras

mnist = tf.keras.datasets.mnist
(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train = (np.reshape(x_train, (60000,784,)).astype(np.float32))/255
x_test = (np.reshape(x_test, (10000,784,)).astype(np.float32))/255
y_train = keras.utils.to_categorical(y_train)
y_test = keras.utils.to_categorical(y_test)


model = keras.models.Sequential()
model.add(keras.layers.Dense(input_dim=(784,),units=100,activation='relu'))
