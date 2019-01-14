#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan 14 00:35:34 2019

@author: jimmy
"""


import numpy as np
import matplotlib.pyplot as plt

base_pg1 = np.load("./training_curve_4500_base.npy")
base_pg2 = np.load("./training_curve_500_base_cont.npy")
base_pg = np.concatenate((base_pg1, base_pg2))

var_reduction1 = np.load("./curve/training_curve_4000_var.npy")
var_reduction2 = np.load("./curve/training_curve_1000_cont.npy")
var_reduction = np.concatenate((var_reduction1, var_reduction2))

x_axis = np.array(range(1,len(var_reduction)+1))


fig1, axes = plt.subplots()
axes.plot(x_axis, var_reduction, label = "Variance Reduction")
axes.plot(x_axis, base_pg, label = "Base")
axes.set_xlabel("episode", fontsize=16)
axes.set_ylabel("mean reward (last 30 ep)", fontsize=16)
axes.legend(loc = 'best', fontsize=12)
fig1.tight_layout()
fig1.savefig("4-1_training_curve.png")
fig1.show()