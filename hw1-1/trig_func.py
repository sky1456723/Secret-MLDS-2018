#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep 26 05:42:22 2018

@author: jeffchen
"""

import numpy as np
from keras.models import Sequential
from keras.layers import Dense
import pylab as pl
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler

# Generating triangular wave
X0 = np.linspace(-4.0 * np.pi, 4.0 * np.pi, 2000).reshape(-1, 1)
X00 = list(X0.reshape(1,-1)[0])
Y0 = [0]*len(X00)
freq = 4.0
for j in range(len(X00)):
	Y0[j] = abs(X00[j]%freq-freq/2.0)/(freq/4.0)-1
Y0 = np.array(Y0).reshape(-1, 1)

# Rescale X and Y
X_scaler = MinMaxScaler(feature_range=(-1.0, 1.0))
Y_scaler = MinMaxScaler(feature_range=(-1.0, 1.0))
X = X_scaler.fit_transform(X0)
Y = Y_scaler.fit_transform(Y0)

res = []
res_rscl = []
history = []
legend_buffer = []
color_buffer = ['ro', 'yo', 'bo']
tries = 3

for j in range(tries):
	epochs_tried, batches = 1000, 32
	if j == 0:
		neurons = [51,100,80,3,10]
	if j == 1:
		neurons = [100,26,18,21,25,70,60,25,67]
	if j == 2:
		neurons = [120,19,14,8,23,11,40,30,90,23,70,20,26]

	model = Sequential()
	model.add(Dense(units=neurons[0], activation='relu', input_dim=X.shape[1]))
	if len(neurons) > 1:
		for i in neurons[1:]:
			model.add(Dense(units=i, activation='relu'))
	model.add(Dense(units=1, activation='linear'))
	model.summary()

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
