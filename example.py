import networkx as nx
import numpy as np
import pickle
import scipy.sparse as sp
import sys
import time
from sklearn.decomposition import PCA
from scat import *
from utils import *

from keras.callbacks import EarlyStopping
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.optimizers import SGD

""" example of semisupervised learning for cora """

def parse_index_file(filename):
    """Parse index file."""
    index = []
    for line in open(filename):
        index.append(int(line.strip()))
    return index


def sample_mask(idx, l):
    """Create mask."""
    mask = np.zeros(l)
    mask[idx] = 1
    return np.array(mask, dtype=np.bool)


def load_data(dataset_str):
    """Load data."""
    names = ['x', 'y', 'tx', 'ty', 'allx', 'ally', 'graph']
    objects = []
    for i in range(len(names)):
        with open("data/ind.{}.{}".format(dataset_str, names[i]), 'rb') as f:
            if sys.version_info > (3, 0):
                objects.append(pickle.load(f, encoding='latin1'))
            else:
                objects.append(pickle.load(f))

    x, y, tx, ty, allx, ally, graph = tuple(objects)
    test_idx_reorder = parse_index_file("data/ind.{}.test.index".format(dataset_str))
    test_idx_range = np.sort(test_idx_reorder)

    if dataset_str == 'citeseer':
        # Fix citeseer dataset (there are some isolated nodes in the graph)
        # Find isolated nodes, add them as zero-vecs into the right position
        test_idx_range_full = range(min(test_idx_reorder), max(test_idx_reorder)+1)
        tx_extended = sp.lil_matrix((len(test_idx_range_full), x.shape[1]))
        tx_extended[test_idx_range-min(test_idx_range), :] = tx
        tx = tx_extended
        ty_extended = np.zeros((len(test_idx_range_full), y.shape[1]))
        ty_extended[test_idx_range-min(test_idx_range), :] = ty
        ty = ty_extended

    features = sp.vstack((allx, tx)).tolil()
    features[test_idx_reorder, :] = features[test_idx_range, :]
    adj = nx.adjacency_matrix(nx.from_dict_of_lists(graph))

    labels = np.vstack((ally, ty))
    labels[test_idx_reorder, :] = labels[test_idx_range, :]

    idx_test = test_idx_range.tolist()
    idx_train = range(len(y))
    idx_val = range(len(y), len(y)+500)

    train_mask = sample_mask(idx_train, labels.shape[0])
    val_mask = sample_mask(idx_val, labels.shape[0])
    test_mask = sample_mask(idx_test, labels.shape[0])

    y_train = np.zeros(labels.shape)
    y_val = np.zeros(labels.shape)
    y_test = np.zeros(labels.shape)
    y_train[train_mask, :] = labels[train_mask, :]
    y_val[val_mask, :] = labels[val_mask, :]
    y_test[test_mask, :] = labels[test_mask, :]

    return adj, features, y_train, y_val, y_test, train_mask, val_mask, test_mask



adj, features, y_train, y_val, y_test, train_mask, val_mask, test_mask = load_data('cora')
A = adj
G = nx.from_scipy_sparse_matrix(A)
L = nx.linalg.laplacianmatrix.laplacian_matrix(G)

lamb, V = np.linalg.eigh(L.toarray())

# get representation for all

features = features.todense()
tStart = time.time()
y_features = getRep(features, lamb, V, layer=3)
tEnd = time.time()
pca = PCA(n_components=1000)
y_features = pca.fit_transform(y_features)


train_feature = y_features[train_mask,:]
val_feature = y_features[val_mask,:]
test_feature = y_features[test_mask,:]
train_labels = y_train[train_mask,:]
val_labels = y_val[val_mask,:]
test_labels = y_test[test_mask,:]

#%%
# semi-supervised training
# simple fully-connected network

input_dim = np.shape(train_feature)[1]
model = Sequential()
model.add(Dropout(rate=0., input_shape=(input_dim,)))
model.add(Dense(2048, activation='relu'))
model.add(Dropout(rate=0.5))
model.add(Dense(2048, activation='relu'))
model.add(Dropout(rate=0.5))
model.add(Dense(7, activation='softmax'))

optimizer = SGD(lr=0.1, momentum=0., decay=0.01, nesterov=False)

model.compile(optimizer=optimizer,
              loss='categorical_crossentropy',
              metrics=['accuracy'])

early_stopping = EarlyStopping(monitor='val_loss', patience=10)

history = model.fit(train_feature, train_labels,
                    batch_size=140,
                    validation_data = (val_feature, val_labels),
                    epochs=200,
                    callbacks=[early_stopping])
    

score = model.evaluate(test_feature, test_labels)
print('Test loss: ', score[0])
print('Test accuracy: ', score[1])