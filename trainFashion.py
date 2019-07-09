# -*- coding: utf-8 -*-

import time
import os

import tensorflow as tf
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt

from tensorflow.examples.tutorials.mnist import input_data
data = input_data.read_data_sets('data/fashion')

import argparse
import pickle
from scat import *
from utils import *

# =============================================================================
# settings
# =============================================================================

parser = argparse.ArgumentParser()
parser.add_argument("-s", "--scat", help="choose the scattering method 'S'/'D'", default='S')
parser.add_argument("-g", "--gaussianization", help="choose the gaussianization method 'W'/'N'", default='N')
args = parser.parse_args()



# =============================================================================
# create graph 
# =============================================================================

dim_of_img = 28
num_of_vertex = 784
sigma = 1.;
w1 = np.exp(-1./sigma/sigma);
w2 = np.exp(-2./sigma/sigma);

# create graph
G = nx.Graph()
edgelist = []
for i in range(num_of_vertex):
    if i % dim_of_img == 0:
        if i < (dim_of_img-1) * dim_of_img:
            edgelist.extend([(i, i+1, w1),
                             (i, i+dim_of_img, w1),
                             (i, i+dim_of_img+1, w2)])
        else:
            edgelist.extend([(i, i+1, w1)])
    elif i % dim_of_img == dim_of_img - 1:
        if i < (dim_of_img-1) * dim_of_img:
            edgelist.extend([(i, i+dim_of_img-1, w2),
                             (i, i+dim_of_img, w1)])
    else:
        if i < (dim_of_img-1) * dim_of_img:
            edgelist.extend([(i, i+1, w1),
                             (i, i+dim_of_img-1, w2),
                             (i, i+dim_of_img, w1),
                             (i, i+dim_of_img+1, w2)])
        else:
            edgelist.extend([(i, i+1, w1)])    
G.add_weighted_edges_from(edgelist)

#G = nx.generators.lattice.grid_2d_graph(dim_of_img, dim_of_img)

W = nx.linalg.graphmatrix.adjacency_matrix(G, nodelist=list(range(num_of_vertex)))
L = nx.linalg.laplacianmatrix.laplacian_matrix(G, nodelist=list(range(num_of_vertex)))

if args.scat == 'S':
    lamb, V = np.linalg.eigh(L.toarray())



train_images = data[0]._images
train_labels = data[0]._labels
test_images = data[2]._images
test_labels = data[2]._labels

train_images = train_images[train_labels==9, :]  - .5
test_images = test_images[test_labels==9, :] - .5

# for plot
num_of_row = 8
num_of_col = 8

# gaussianization dim
dim_final_feature = 784


# =============================================================================
# encoder
# =============================================================================

# processing: get input features

print("Processing features ...")

if args.scat == 'D':
    train_feature = diffusion_scat(train_images.T, W)
    test_feature = diffusion_scat(test_images.T, W)
else:
    train_feature = getRep(train_images.T, lamb, V, N=3/8)
    test_feature = getRep(test_images.T, lamb, V, N=3/8)


train_feature = train_feature.T
test_feature = test_feature.T




if args.gaussianization == 'W':
    train_gaussian = gaussianization_whiten(train_feature, num_of_components=dim_final_feature)
    test_gaussian = gaussianization_whiten(test_feature, num_of_components=dim_final_feature)
else:
    train_gaussian = gaussianization_spherize(train_feature, num_of_components=dim_final_feature)
    test_gaussian = gaussianization_spherize(test_feature, num_of_components=dim_final_feature)
    train_mu = np.zeros(np.shape(np.mean(train_gaussian, axis=0)))
    train_cov = np.cov(train_gaussian.T)
    
dim_final_feature = np.shape(train_gaussian)[1]


# =============================================================================
# decoder
# =============================================================================

dim_hidden = 512
X = tf.placeholder(tf.float32, shape=[None, dim_final_feature])
keep_prob = tf.placeholder_with_default(0.5, shape=())

W1 = tf.Variable(xavier_init([dim_final_feature, dim_hidden]), name='W1')
b1 = tf.Variable(tf.zeros(shape=[dim_hidden]), name='b1')
W2 = tf.Variable(xavier_init([dim_hidden, dim_hidden]), name='W2')
b2 = tf.Variable(tf.zeros(shape=[dim_hidden]), name='b2')
W3 = tf.Variable(xavier_init([dim_hidden, 784]), name='W3')
b3 = tf.Variable(tf.zeros(shape=[784]), name='b3')
theta = [W1, b1, W2, b2, W3, b3]

def fcn(x):
    out1 = leaky_relu(tf.matmul(x, W1) + b1)
    out1 = tf.nn.dropout(out1, keep_prob=keep_prob)
    out2 = leaky_relu(tf.matmul(out1, W2) + b2)
    out2 = tf.nn.dropout(out2, keep_prob=keep_prob)
    out3 = tf.matmul(out2, W3) + b3
    out = tf.tanh(out3)/2
    return out



fcn_loss = tf.reduce_mean(tf.abs(train_images - fcn(X)))
fcn_solver = (tf.train.AdamOptimizer(learning_rate=0.001)
        .minimize(fcn_loss, var_list=theta))
       


# =============================================================================
# train
# =============================================================================

os.mkdir('./img_gen')

sess = tf.Session() 
sess.run(tf.global_variables_initializer())


loss_all = np.zeros(2000)

for it in range(2000):
    _, loss_curr = sess.run(
            [fcn_solver, fcn_loss],
            feed_dict={X: train_gaussian, keep_prob: 0.4}
            )
    
    
    loss_all[it] = loss_curr
       
    if it % 200 == 0:
        print('Iter: {}; loss: {:.4}'
              .format(it, loss_curr))   
        
           
        samples = sess.run(fcn(X), feed_dict={X: train_gaussian[:num_of_row*num_of_col,:], keep_prob: 1})
        fig, axs = plt.subplots(num_of_row, num_of_col, figsize=(10,10))
        for idx_now in range(num_of_col):    
            for j in range(num_of_row):
                axs[j,idx_now].imshow(samples[idx_now*num_of_col+j,:].reshape([28,28]), cmap='gray')    
                axs[j,idx_now].get_xaxis().set_visible(False)
                axs[j,idx_now].get_yaxis().set_visible(False)
                
        fig.savefig("./img_gen/" + args.scat + args.gaussianization + str(it) + "-th_iter.png")
        
        samples = sess.run(fcn(X), feed_dict={X: test_gaussian[:num_of_row*num_of_col,:], keep_prob: 1})
        fig, axs = plt.subplots(num_of_row, num_of_col, figsize=(10,10))
        for idx_now in range(num_of_col):    
            for j in range(num_of_row):
                axs[j,idx_now].imshow(samples[idx_now*num_of_col+j,:].reshape([28,28]), cmap='gray')    
                axs[j,idx_now].get_xaxis().set_visible(False)
                axs[j,idx_now].get_yaxis().set_visible(False)
                
        fig.savefig("./img_gen/" + args.scat + args.gaussianization + str(it) + "-th_iter_test.png")
        

# =============================================================================
# generation from random samples
# =============================================================================
        
samples = []      

if args.gaussianization == 'W':
    for _ in range(num_of_col):
        samples.append(sess.run(fcn(X), feed_dict={X: sample_z_2d(num_of_row, dim_final_feature), keep_prob: 1}))   
else:
    for _ in range(num_of_col):
        samples.append(sess.run(fcn(X), feed_dict={X: sample_z_full(train_mu, train_cov, num_of_row), keep_prob: 1}))     
    
fig, axs = plt.subplots(num_of_row, num_of_col, figsize=(10,10))
for idx_now in range(num_of_col):    
    for j in range(num_of_row):
        axs[j,idx_now].imshow(samples[idx_now][j,:].reshape([28,28]), cmap='gray')    
        axs[j,idx_now].get_xaxis().set_visible(False)
        axs[j,idx_now].get_yaxis().set_visible(False)
    
fig.savefig("./img_gen/" + args.scat + args.gaussianization + "_random_generated.png")
