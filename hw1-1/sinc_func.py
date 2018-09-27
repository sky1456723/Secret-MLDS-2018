#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Sep 25 19:07:14 2018

@author: jeffchen
"""

import numpy as np
# import tensorflow as tf
# import keras
from keras.models import Sequential
from keras.layers import Dense
import pylab as pl
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler

pi = np.pi
X0 = np.linspace(-4.0 * pi, 4.0 * pi, 2000).reshape(-1, 1)
Y0 = np.sin(pi * X0) / (pi * X0) # since X is never 0 here.
np.random.seed(7)

X_scaler = MinMaxScaler(feature_range=(-1.0, 1.0))
Y_scaler = MinMaxScaler(feature_range=(-1.0, 1.0))

X = X_scaler.fit_transform(X0)
Y = Y_scaler.fit_transform(Y0)

""" # Tried Networks
epochs_tried, batches, neurons = 200, 80, [25,25,100,5,25,3,60,25,100,100,25,5,100,20,25]
# 24,324 paras	# loss: 7.9459e-04	 - mean_absolute_error: 0.0214 - difference = 0.025
epochs_tried, batches, neurons = 200, 80, [20,100,50,30,5,8,40,5,100,100,12,25,10,10]
# 22,106 paras	# loss: 8.4974e-04	 - mean_absolute_error: 0.0235 - difference = 0.040
epochs_tried, batches, neurons = 200, 80, [25,25,200,15,25,25,100,25,5,100,20,25]
# 16,131 paras	# loss: 8.5727e-04	 - mean_absolute_error: 0.0223 - difference = 0.050
epochs_tried, batches, neurons = 200, 80, [5000]
# 15,001 paras	# loss: 0.0109  	 - mean_absolute_error: 0.0804 - difference = 0.200
epochs_tried, batches, neurons = 200, 80, [100,140]
# 14,481 paras	# loss: 0.0034  	 - mean_absolute_error: 0.0462 - difference = 0.075
epochs_tried, batches, neurons = 200, 80, [100,80,80]
# 14,481 paras	# loss: 0.0017  	 - mean_absolute_error: 0.0326 - difference = 0.050
epochs_tried, batches, neurons = 200, 80, [100,70,40,50,40]
# 14,241 paras	# loss: 0.0013  	 - mean_absolute_error: 0.0288 - difference = 0.050
epochs_tried, batches, neurons = 200, 80, [100,70,40,20,20,20,20,20,20,20]
# 13,471 paras 	# loss: 0.0013  	 - mean_absolute_error: 0.0278 - difference = 0.05
epochs_tried, batches, neurons = 200, 80, [100,70,40,20,20,20,20,20,20,10,10,5,5,5]
# 13,471 paras	# loss: 0.0019  	 - mean_absolute_error: 0.0348 - difference = 0.05
epochs_tried, batches, neurons = 200, 80, [5,5,10,20,20,20,40,100,50,20,20,20,10,5]
# 13,281 paras	# loss: 0.0015  	 - mean_absolute_error: 0.0305 - difference = 0.05
epochs_tried, batches, neurons = 200, 80, [5,5,10,20,70,20,80,50,20,20,30,50,10,10,5]
# 13,241 paras	# loss: 0.0011  	 - mean_absolute_error: 0.0263 - difference = 0.05
epochs_tried, batches, neurons = 200, 80, [25,25,10,20,100,10,90,50,5,20,10,100,10,25]
# 12,826 paras	# loss: 0.0011  	 - mean_absolute_error: 0.0263 - difference = 0.050
epochs_tried, batches, neurons = 200, 40, [25,25,10,20,100,10,90,50,5,20,10,100,10,25]
# 12,826 paras	# loss: 2.7615e-04	 - mean_absolute_error: 0.0127 - difference = 0,020
epochs_tried, batches, neurons = 50, 32, [25,25,10,20,100,10,90,50,5,20,10,100,10,25]
# 12,826 paras	# loss: 0.0014  	 - mean_absolute_error: 0.0315 - difference = 0.05
epochs_tried, batches, neurons = 500, 32, [25,25,10,20,100,10,90,50,5,20,10,100,10,25]
# 12,826 paras	# loss: 1.0678e-05	 - mean_absolute_error: 0.0025 - difference = 0.01
"""
res = []
res_rscl = []
history = []
legend_buffer = []
color_buffer = ['ro', 'yo', 'bo']
tries = 3

for j in range(tries):
	epochs_tried, batches = 20000, 32
	if j == 0:
		neurons = [25,25,10,20,100,10,90,50,5,20,10,100,10,25]
	# 24,324 paras	# loss: 1.0564e-04	- mean_absolute_error: 0.0080 - difference = 0.02
	if j == 1:
		neurons = [5,2,4,8,20,5,40,25,40,100,25,20,50,20,35]
	# 24,324 paras	# loss: 1.0564e-04	- mean_absolute_error: 0.0080 - difference = 0.02
	if j == 2:
		neurons = [19,15,10,10,70,20,80,50,20,20,30,50,10,10,5]
	

	model = Sequential()
	model.add(Dense(units=neurons[0], activation='relu', input_dim=X.shape[1]))
	if len(neurons) > 1:
	    for i in neurons[1:]:
	        model.add(Dense(units=i, activation='relu'))
	model.add(Dense(units=1, activation='linear'))
	model.summary()
#'''
	model.compile(loss='mse', optimizer='adam', metrics=['mae'])
	history.append(model.fit(X, Y, epochs=epochs_tried, batch_size=batches, verbose=2))

	res.append(model.predict(X, batch_size=batches))
	res_rscl.append(Y_scaler.inverse_transform(res[-1]))
	
Y_rscl = Y_scaler.inverse_transform(Y)
pl.xlim((-4.0 * np.pi, 4.0 * np.pi))
pl.subplot2grid(((7, 6)), (0, 0), rowspan=4, colspan=3)
pl.plot(X0, Y_rscl, 'k.',markersize=1)
legend_buffer = ['real']
for j in range(tries):
	pl.plot(X0, res_rscl[j], color_buffer[j],markersize=1)
	legend_buffer.append('learned'+str(j+1))
	
pl.legend(legend_buffer)
pl.xlabel('x')
pl.ylabel('y=sinc(x)')

pl.subplot2grid((7, 6), (5, 0), rowspan=2, colspan=3)
for j in range(tries):
	pl.plot(X0, res_rscl[j] - Y_rscl, color_buffer[j][0])
pl.legend(legend_buffer[1:])
pl.xlabel('x')
pl.ylabel('difference')

plt.subplot2grid((7, 6), (0, 4), rowspan=3, colspan=2)
for j in range(tries):
	plt.plot(history[j].history['loss'], color_buffer[j][0])
plt.title('Model error (loss)')
pl.legend(legend_buffer[1:])
plt.ylabel('mean squared error')
plt.xlabel('Epoch')

plt.subplot2grid((7, 6), (4, 4), rowspan=3, colspan=2)
for j in range(tries):
	plt.plot(history[j].history['mean_absolute_error'], color_buffer[j][0])
pl.legend(legend_buffer[1:])
plt.title('Model error')
plt.ylabel('mean absolute error')
plt.xlabel('Epoch')

plt.show()
#'''