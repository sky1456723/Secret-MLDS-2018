import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import math
import os
from sklearn.manifold import TSNE

### HYPERPARAMETERS ###

learning_rate = 0.05
beta = 0.01
size = 40
sd_init = 0.5
sd_neighbor = 0.25


### I/O ###

x_plot = np.arange(0.1, math.pi*2*2, 0.1)
# return an array of len=125
# [0. 0.1 0.2 0.3...]
x_train = x_plot.reshape(-1, 1)
# reshaped into a matrix of shape=(125, 1)
# [[0] [0.1] [0.2]...]
x_tf = tf.placeholder(tf.float32, [None, 1])

y_plot = [math.sin(i) * i / 12 for i in x_plot]
y_train = np.asarray(y_plot).reshape(-1, 1)
y_tf = tf.placeholder(tf.float32, [None, 1])

dict = {x_tf: x_train, y_tf: y_train}


### MODEL ###

W = tf.Variable(tf.random_normal([size], stddev=sd_init), dtype=tf.float32, name='W')
b = tf.Variable(tf.zeros([size]), dtype=tf.float32, name='b')
# gaussian-initialized b is bad for simulation
model = tf.reduce_sum(tf.tanh(tf.matmul(x_tf, tf.expand_dims(W, 0)) + b), 1, keepdims=True)
# expand W to (1, 30)

neighbor_W = W + tf.random_normal(shape=[size], stddev=sd_neighbor)
neighbor_b = b + tf.random_normal(shape=[size], stddev=sd_neighbor)
# each call to sess.run(neighbor, ...) samples different values
neighbor_loss = tf.losses.mean_squared_error(y_tf, \
    tf.reduce_sum(tf.tanh(tf.matmul(x_tf, tf.expand_dims(neighbor_W, 0)) + neighbor_b), 1, keepdims=True)) + \
    beta * (tf.nn.l2_loss(neighbor_W) + tf.nn.l2_loss(neighbor_b))

loss1 = tf.losses.mean_squared_error(y_tf, model)
loss2 = tf.losses.mean_squared_error(y_tf, model) + beta * (tf.nn.l2_loss(W) + tf.nn.l2_loss(b))
train1 = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss1)
train2 = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss2)


### DATA ANALYSIS ###

axis = []
loss_list = []
path_list = []
neighbor_list = []
neighbor_loss_list = []


### SESSION ###

with tf.Session() as sess:
    tf.global_variables_initializer().run(session = sess)

    for i in range(100000):
        predict, _, curr_loss = sess.run([model, train1, loss1], feed_dict=dict)

        if i % 100 == 0:
            axis.append(i)
            loss_list.append(curr_loss)

    plt.figure(0)
    pred_plot = predict.reshape(1, -1)[0]
    plt.cla()
    plt.plot(x_plot, y_plot)
    plt.plot(x_plot, pred_plot, 'g--')
    plt.xlabel('X Value')
    plt.ylabel('Prediction')
    plt.title(['Step: ', str(100000), ' Loss: ', curr_loss])
    plt.savefig('Sim.png')
    # save simulation result

    print("Start second order optimization...")
    for i in range(100):
        predict, _ = sess.run([model, train2], feed_dict=dict)

        if i % 10 == 0:
            # axis.append(i + 100000)
            # loss_list.append(curr_loss)
            _w, _b = sess.run([W, b], feed_dict=dict)
            path_list.append(_w + _b)

            for j in range(200):
                # randomly generate 100 weights, 1000 neighbors at total
                nw, nb, nl = sess.run([neighbor_W, neighbor_b, neighbor_loss], feed_dict=dict)
                neighbor_list.append(nw + nb)
                # concatenate two arrays
                neighbor_loss_list.append(nl)


print("Start dimension reduction...")
emb = TSNE(n_components=2).fit_transform(path_list + neighbor_list)
print("Dimension reduction finished.")
# returns (10 + 1000)x2 matrix
path_emb = emb[:10]
neighbor_emb = emb[10:]
ne = np.c_[neighbor_emb, neighbor_loss_list]
# append loss list as an extra column
ne = ne[ne[:, 2].argsort(kind='heapsort')]
# sort according to loss column so that greatest loss will be last drawn

plt.figure(1)
plt.scatter(ne[:, 0], ne[:, 1], c=ne[:, 2], s=75)
# plot loss landscape
plt.plot(path_emb[:, 0], path_emb[:, 1], '.', color='red', linestyle=':')
# plot training path
plt.plot(path_emb[0, 0], path_emb[0, 1], 'o', color='red')
# plot starting position
plt.savefig('Path.png')

plt.figure(2)
plt.plot(axis, loss_list)
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.savefig('Loss.png')

# no main - simplicity first
