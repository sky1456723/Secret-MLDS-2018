#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan 14 00:35:34 2019

@author: jimmy
"""


import numpy as np
import torch
import matplotlib.pyplot as plt

base_dqn = np.array([0.56, 0.9, 2.04, 2.54, 2.62, 1.96, 2.36, 5.48, 3.98, 3.54, \
                     3.58, 4.04, 4.52, 4.32, 6.02, 4.84, 4.98, 6.0, 6.48, 6.28, \
                     7.56, 5.86, 8.48, 10.52, 9.7, 12.5, 14.68, 17.24, 15.8, 28.4, \
                     35.86, 62.1, 40.46, 67.76, 82.65, 75.26])
improved_dqn = np.array([0.26, 0.29, 2.07, 5.72, 5.46, 6.08, 8.3, 6.9, 13.97, 17.52,\
                         16.99, 28.8, 20.74, 29.71, 32.6, 51.23, 40.34, 53.39, 47.76, 33.14, 70.89])
base_line = np.array([70.89]*len(base_dqn))
x_axis1 = np.array(range(1,len(base_dqn)+1))*500
x_axis2 = np.array(range(1,len(improved_dqn)+1))*500
fig1, axes = plt.subplots()
axes.plot(x_axis2, improved_dqn, label = "Improved")
axes.plot(x_axis1, base_dqn, label = "Base")
axes.plot(x_axis1, base_line, '--')
axes.set_xlabel("episode", fontsize=16)
axes.set_ylabel("mean reward (last 100 ep)", fontsize=16)
axes.legend(loc = 'best', fontsize=12)
fig1.suptitle('Testing Result', fontsize=16)
fig1.tight_layout(rect=[0, 0, 1, 0.95])
fig1.savefig("4-2_testing_curve.png")
fig1.show()

base_dqn = torch.load("./model/4-2/base_dqn_17500.ckpt")["curve"]
improved_dqn = torch.load("./model/4-2/test2_dqn_3000_7500.ckpt")["curve"]
x_axis1 = np.array(range(1,len(base_dqn)+1))
x_axis2 = np.array(range(1,len(improved_dqn)+1))
fig1, axes = plt.subplots()
axes.plot(x_axis2, improved_dqn, label = "Improved")
axes.plot(x_axis1, base_dqn, label = "Base")
#axes.plot(x_axis1, base_line, '--')
axes.set_xlabel("episode", fontsize=16)
axes.set_ylabel("mean reward (last 100 ep)", fontsize=16)
axes.legend(loc = 'best', fontsize=12)
fig1.suptitle('Training Result', fontsize=16)
fig1.tight_layout(rect=[0, 0, 1, 0.95])
fig1.savefig("4-2_training_curve.png")
fig1.show()
