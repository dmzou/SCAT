# -*- coding: utf-8 -*-

import time
import os

import tensorflow as tf
import numpy as np
import scipy.sparse as sp
import networkx as nx
import matplotlib.pyplot as plt

from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

import argparse
import pickle
from scat import *
from utils import *

# =============================================================================
# settings
# =============================================================================

parser = argparse.ArgumentParser()
parser.add_argument("-s", "--scat", help="choose the scattering method 'S'/'D'", default='S')
parser.add_argument("-d", "--dataset", help="choose the dataset 'cora'/'citeseer'/'pubmed'", default='cora')
args = parser.parse_args()



# =============================================================================
# data preparation
# =============================================================================

def parse_index_file(filename):
    index = []
    for line in open(filename):
        index.append(int(line.strip()))
    return index


def load_data(dataset):
    # load the data: x, tx, allx, graph
    names = ['x', 'tx', 'allx', 'graph']
    objects = []
    for i in range(len(names)):
        with open("data/ind.{}.{}".format(dataset, names[i]), 'rb') as f:
            objects.append(pickle.load(f, encoding='latin1'))
    x, tx, allx, graph = tuple(objects)
    test_idx_reorder = parse_index_file("data/ind.{}.test.index".format(dataset))
    test_idx_range = np.sort(test_idx_reorder)

    if dataset == 'citeseer':
        # Fix citeseer dataset (there are some isolated nodes in the graph)
        # Find isolated nodes, add them as zero-vecs into the right position
        test_idx_range_full = range(min(test_idx_reorder), max(test_idx_reorder)+1)
        tx_extended = sp.lil_matrix((len(test_idx_range_full), x.shape[1]))
        tx_extended[test_idx_range-min(test_idx_range), :] = tx
        tx = tx_extended

    features = sp.vstack((allx, tx)).tolil()
    features[test_idx_reorder, :] = features[test_idx_range, :]
    adj = nx.adjacency_matrix(nx.from_dict_of_lists(graph))

    return adj, features


def sparse_to_tuple(sparse_mx):
    if not sp.isspmatrix_coo(sparse_mx):
        sparse_mx = sparse_mx.tocoo()
    coords = np.vstack((sparse_mx.row, sparse_mx.col)).transpose()
    values = sparse_mx.data
    shape = sparse_mx.shape
    return coords, values, shape


def preprocess_graph(adj):
    adj = sp.coo_matrix(adj)
    adj_ = adj + sp.eye(adj.shape[0])
    rowsum = np.array(adj_.sum(1))
    degree_mat_inv_sqrt = sp.diags(np.power(rowsum, -0.5).flatten())
    adj_normalized = adj_.dot(degree_mat_inv_sqrt).transpose().dot(degree_mat_inv_sqrt).tocoo()
    return sparse_to_tuple(adj_normalized)


def construct_feed_dict(adj_normalized, adj, features, placeholders):
    # construct feed dictionary
    feed_dict = dict()
    feed_dict.update({placeholders['features']: features})
    feed_dict.update({placeholders['adj']: adj_normalized})
    feed_dict.update({placeholders['adj_orig']: adj})
    return feed_dict


def mask_test_edges(adj):
    # Function to build test set with 10% positive links
    # NOTE: Splits are randomized and results might slightly deviate from reported numbers in the paper.
    # TODO: Clean up.

    # Remove diagonal elements
    adj = adj - sp.dia_matrix((adj.diagonal()[np.newaxis, :], [0]), shape=adj.shape)
    adj.eliminate_zeros()
    # Check that diag is zero:
    assert np.diag(adj.todense()).sum() == 0

    adj_triu = sp.triu(adj)
    adj_tuple = sparse_to_tuple(adj_triu)
    edges = adj_tuple[0]
    edges_all = sparse_to_tuple(adj)[0]
    num_test = int(np.floor(edges.shape[0] / 10.))
    num_val = int(np.floor(edges.shape[0] / 20.))

    all_edge_idx = list(range(edges.shape[0]))
    np.random.shuffle(all_edge_idx)
    val_edge_idx = all_edge_idx[:num_val]
    test_edge_idx = all_edge_idx[num_val:(num_val + num_test)]
    test_edges = edges[test_edge_idx]
    val_edges = edges[val_edge_idx]
    train_edges = np.delete(edges, np.hstack([test_edge_idx, val_edge_idx]), axis=0)

    def ismember(a, b, tol=5):
        rows_close = np.all(np.round(a - b[:, None], tol) == 0, axis=-1)
        return np.any(rows_close)

    test_edges_false = []
    while len(test_edges_false) < len(test_edges):
        idx_i = np.random.randint(0, adj.shape[0])
        idx_j = np.random.randint(0, adj.shape[0])
        if idx_i == idx_j:
            continue
        if ismember([idx_i, idx_j], edges_all):
            continue
        if test_edges_false:
            if ismember([idx_j, idx_i], np.array(test_edges_false)):
                continue
            if ismember([idx_i, idx_j], np.array(test_edges_false)):
                continue
        test_edges_false.append([idx_i, idx_j])

    val_edges_false = []
    while len(val_edges_false) < len(val_edges):
        idx_i = np.random.randint(0, adj.shape[0])
        idx_j = np.random.randint(0, adj.shape[0])
        if idx_i == idx_j:
            continue
        if ismember([idx_i, idx_j], train_edges):
            continue
        if ismember([idx_j, idx_i], train_edges):
            continue
        if ismember([idx_i, idx_j], val_edges):
            continue
        if ismember([idx_j, idx_i], val_edges):
            continue
        if val_edges_false:
            if ismember([idx_j, idx_i], np.array(val_edges_false)):
                continue
            if ismember([idx_i, idx_j], np.array(val_edges_false)):
                continue
        val_edges_false.append([idx_i, idx_j])

    assert ~ismember(test_edges_false, edges_all)
    assert ~ismember(val_edges_false, edges_all)
    assert ~ismember(val_edges, train_edges)
    assert ~ismember(test_edges, train_edges)
    assert ~ismember(val_edges, test_edges)

    data = np.ones(train_edges.shape[0])

    # Re-build adj matrix
    adj_train = sp.csr_matrix((data, (train_edges[:, 0], train_edges[:, 1])), shape=adj.shape)
    adj_train = adj_train + adj_train.T

    # NOTE: these edge lists only contain single direction of edge!
    return adj_train, train_edges, val_edges, val_edges_false, test_edges, test_edges_false

# Load data
adj, features = load_data(args.dataset)

# Store original adjacency matrix (without diagonal entries) for later
adj_orig = adj
adj_orig = adj_orig - sp.dia_matrix((adj_orig.diagonal()[np.newaxis, :], [0]), shape=adj_orig.shape)
adj_orig.eliminate_zeros()

if not os.path.exists("./"+args.dataset+"_split.data"):
    adj_train, train_edges, val_edges, val_edges_false, test_edges, test_edges_false = mask_test_edges(adj)
    with open("./"+args.dataset+"_split.data", "wb") as f:
        pickle.dump(adj_train, f)
        pickle.dump(train_edges, f)
        pickle.dump(val_edges, f)
        pickle.dump(val_edges_false, f)
        pickle.dump(test_edges, f)
        pickle.dump(test_edges_false, f)
else:
    with open("./"+args.dataset+"_split.data", "rb") as f:
        adj_train = pickle.load(f)
        train_edges = pickle.load(f)
        val_edges = pickle.load(f)	
        val_edges_false = pickle.load(f)
        test_edges = pickle.load(f)
        test_edges_false = pickle.load(f)
    
adj = adj_train
features = features.todense()

train_edges_append = np.matlib.repmat(np.array(range(np.shape(adj)[0])), 2, 1).T
train_edges = np.concatenate((train_edges, train_edges_append), axis=0)


# =============================================================================
# contruct graph
# =============================================================================

W = adj
G = nx.from_scipy_sparse_matrix(W)
L = nx.linalg.laplacianmatrix.laplacian_matrix(G)

if args.scat == 'S':
    lamb, V = np.linalg.eigh(L.toarray())

# =============================================================================
# encoder
# =============================================================================


if args.scat == 'D':
    y_features = diffusion_scat(features, W, layer=2)
else:    
    y_features = getRep(features, lamb, V, layer=2)


dim_final_feature = 128

def dim_reduction(A, pca=True, num_of_components=128):
    if not pca:
        num_of_components = A.shape[1]
    pca = PCA(n_components=num_of_components)
    A_pca = pca.fit_transform(A)
    scaler = StandardScaler()
    for i in range(np.shape(A_pca)[0]):
        A_pca[i,:] = scaler.fit_transform(A_pca[i,:].reshape(-1,1)).reshape(-1)
    return A_pca

y_features_pca = dim_reduction(y_features, num_of_components=dim_final_feature)

# construct all training / validation / testing data

train_features = np.zeros([train_edges.shape[0], 2 * dim_final_feature])
for i in range(train_edges.shape[0]):
    train_features[i, :] = np.concatenate([y_features_pca[train_edges[i,0],:], 
                  y_features_pca[train_edges[i,1],:]], axis=0)

val_features = np.zeros([val_edges.shape[0], 2 * dim_final_feature])
for i in range(val_edges.shape[0]):
    val_features[i, :] = np.concatenate([y_features_pca[val_edges[i,0],:], 
                  y_features_pca[val_edges[i,1],:]], axis=0)

val_edges_false = np.matrix(val_edges_false)    
val_features_false = np.zeros([val_edges_false.shape[0], 2 * dim_final_feature])
for i in range(val_edges_false.shape[0]):
    val_features_false[i, :] = np.concatenate([y_features_pca[val_edges_false[i,0],:], 
                  y_features_pca[val_edges_false[i,1],:]], axis=0)
    
test_features = np.zeros([test_edges.shape[0], 2 * dim_final_feature])
for i in range(test_edges.shape[0]):
    test_features[i, :] = np.concatenate([y_features_pca[test_edges[i,0],:],
                  y_features_pca[test_edges[i,1],:]], axis=0)
 
test_edges_false = np.matrix(test_edges_false)        
test_features_false = np.zeros([test_edges_false.shape[0], 2 * dim_final_feature])
for i in range(test_edges_false.shape[0]):
    test_features_false[i, :] = np.concatenate([y_features_pca[test_edges_false[i,0],:], 
                  y_features_pca[test_edges_false[i,1],:]], axis=0)
    
    
# =============================================================================
# decoder 
# =============================================================================
    
dim_hidden = 512
X = tf.placeholder(tf.float32, shape=[None, 2 * dim_final_feature])
W = tf.Variable(xavier_init([dim_final_feature, dim_hidden]), name='W')
b = tf.Variable(tf.zeros(shape=[dim_hidden]), name='b')
theta = [W, b]

def fcn(x):
    x1 = x[:, :dim_final_feature]
    x2 = x[:, dim_final_feature:]
    out1 = tf.nn.relu(tf.matmul(x1, W) + b)
    out2 = tf.nn.relu(tf.matmul(x2, W) + b)
    out_linear = tf.multiply(out1, out2)
    out_linear = tf.reduce_mean(out_linear, axis=1)
    out = 1 / (1 + tf.exp(-out_linear))
    return out

fcn_loss = -tf.reduce_mean(tf.log(fcn(X)))

fcn_solver = (tf.train.AdamOptimizer(learning_rate=0.001)
            .minimize(fcn_loss, var_list=theta))



# =============================================================================
# training and testing
# =============================================================================
    

# roc score
def get_roc_score(test_features, test_features_false):
    preds = sess.run(
            fcn(X),
            feed_dict={X: test_features}
            )
    preds_neg = sess.run(
                fcn(X),
                feed_dict={X: test_features_false}
                )
    # predict on test set of edges
    preds_all = np.hstack([preds, preds_neg])
    labels_all = np.hstack([np.ones(len(preds)), np.zeros(len(preds))])
    roc_score = roc_auc_score(labels_all, preds_all)
    ap_score = average_precision_score(labels_all, preds_all)
    return roc_score, ap_score    


sess = tf.Session() 
sess.run(tf.global_variables_initializer())
for it in range(1000):
    _, loss_curr = sess.run(
            [fcn_solver, fcn_loss],
            feed_dict={X: train_features}
            )

    if (it+1) % 100 == 0:
        roc_score_val, ap_score_val = get_roc_score(val_features, val_features_false)

        print('Iter: {}; loss: {:.4}'
              .format(it, loss_curr))
        print('Validation ROC score: {:.4}; AP score: {:.4}'
              .format(roc_score_val, ap_score_val))
       
# testing
roc_score, ap_score = get_roc_score(test_features, test_features_false)
print('ROC score: {:.4}; AP score: {:.4}'
      .format(roc_score, ap_score))
