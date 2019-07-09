# -*- coding: utf-8 -*-

from rdkit import RDLogger, Chem
from rdkit.Chem import MolToSmiles, MolFromSmiles, Draw
from rdkit.Chem.QED import qed
from chainer_chemistry import datasets
from chainer_chemistry.dataset.preprocessors.ggnn_preprocessor import GGNNPreprocessor

import numpy as np
import networkx as nx
import tensorflow as tf
import os.path
import time


from keras.utils import to_categorical

import argparse
import pickle
from scat import *
from utils import *
from utilMol import *


# =============================================================================
# settings
# =============================================================================

parser = argparse.ArgumentParser()
parser.add_argument("-s", "--scat", help="choose the scattering method 'S'/'D'", default='S')
parser.add_argument("-g", "--gaussianization", help="choose the gaussianization method 'W'/'N'", default='N')
args = parser.parse_args()


# =============================================================================
# prepare data
# =============================================================================

num_train = 133885

if not os.path.exists("./qm9.data"):
    preprocessor = GGNNPreprocessor()
    dataset, dataset_smiles = datasets.get_qm9(preprocessor, labels=None, return_smiles=True)
    num_of_data = len(dataset)

    features = []
    adjs = []
    for idx in range(num_of_data):
        atom, adj, labels = dataset[idx]
        if len(atom) < 9:
            n_temp = len(atom)
            atom_temp = np.zeros(9).astype(int)
            atom_temp[:n_temp] = atom
            atom_to_append = atom_to_hot(atom_temp)
        else:
            atom_to_append = atom_to_hot(atom)
        if len(atom) < 9:
            adj_temp = adj[0] + 2 * adj[1] + 3 * adj[2]            
            adj_to_append = np.zeros((9,9)).astype(int)
            adj_to_append[:n_temp, :n_temp] = adj_temp
        else:
            adj_to_append = adj[0] + 2 * adj[1] + 3 * adj[2]

        features.append( atom_to_append )
        adjs.append( adj_to_append )
        
    # make training / validation / testing dataset
    train_idx = np.random.choice(len(features), size=num_train, replace=False)
    train_data = []
    train_features = []
    train_adj = []
    for idx in train_idx:
        train_data.append(dataset_smiles[idx])
        train_features.append(features[idx])
        train_adj.append(adjs[idx])
     
    with open("./qm9.data", "wb") as f:
        pickle.dump(train_data, f)
        pickle.dump(train_features, f)
        pickle.dump(train_adj, f)
        
        
else:
    
    with open("./qm9.data", "rb") as f:
        train_data = pickle.load(f)
        train_features = pickle.load(f)
        train_adj = pickle.load(f)

print("QM9 data loaded.")
    

# =============================================================================
# encoder
# =============================================================================

feature_final = []


for idx in range(num_train):
    G = nx.from_numpy_matrix(train_adj[idx])
    
    if args.scat == 'D':
        y_features = diffusion_scat( train_features[idx].T, nx.adjacency_matrix(G) )
    else:
        L = nx.linalg.laplacianmatrix.laplacian_matrix(G)
        lamb, V = np.linalg.eigh(L.toarray())
        y_features = getRep(train_features[idx].T, lamb, V) 
	
    y_features = y_features.reshape(-1)
    feature_final.append(y_features)
feature_final = np.asarray(feature_final)

print("Scattering finished.")


feature_final = feature_final.reshape((num_train, -1))

if args.gaussianization == 'W':
    feature_final = gaussianization_whiten(feature_final, pca=True, num_of_components=15*9)
else:
    feature_final = gaussianization_spherize(feature_final, pca=True, num_of_components=15*9)
    train_mu = np.zeros(np.shape( np.mean(feature_final, axis=0) ))
    train_cov = np.cov(feature_final.T)


feature_final = feature_final.reshape((num_train, 9, -1))
    

# =============================================================================
# decoder 
# =============================================================================

dim_atom = 9
dim_bond_type = 4
dim_atom_type = 5
dim_final_feature = 15
dim_final_1 = dim_atom_type
dim_final_2 = dim_atom * dim_bond_type * 15

X = tf.placeholder(tf.float32, shape=[None, dim_atom, dim_final_feature])
W1 = tf.Variable(xavier_init([dim_atom * dim_final_feature, 128]))
b1 = tf.Variable(tf.zeros(shape=[128]))
W11 = tf.Variable(xavier_init([128, 256]))
b11 = tf.Variable(tf.zeros(shape=[256]))
W12 = tf.Variable(xavier_init([256, 512]))
b12 = tf.Variable(tf.zeros(shape=[512]))
W13 = tf.Variable(xavier_init([512, dim_atom * dim_final_1]))
b13 = tf.Variable(tf.zeros(shape=[dim_atom * dim_final_1]))

W2 = tf.Variable(xavier_init([dim_atom * dim_final_feature, 128]))
b2 = tf.Variable(tf.zeros(shape=[128]))
W21 = tf.Variable(xavier_init([128, 256]))
b21 = tf.Variable(tf.zeros(shape=[256]))
W22 = tf.Variable(xavier_init([256, 512]))
b22 = tf.Variable(tf.zeros(shape=[512]))
W23 = tf.Variable(xavier_init([512, dim_final_2]))
b23 = tf.Variable(tf.zeros(shape=[dim_final_2]))

theta = [W1, b1, W11, b11, W12, b12, W13, b13, 
         W2, b2, W21, b21, W22, b22, W23, b23]

def fcn(x):
    out1 = tf.reshape(x, (-1, dim_atom * dim_final_feature))
    out1 = leaky_relu( tf.matmul(out1, W1) + b1 )
    out1 = leaky_relu( tf.matmul(out1, W11) + b11 )
    out1 = leaky_relu( tf.matmul(out1, W12) + b12 )
    out1 = leaky_relu( tf.matmul(out1, W13) + b13 )
    out1 = tf.reshape(out1, (-1, dim_atom, dim_final_1))


    out2 = tf.reshape(x, (-1, dim_atom * dim_final_feature))
    out2 = leaky_relu( tf.matmul(out2, W2) + b2 )
    out2 = leaky_relu( tf.matmul(out2, W21) + b21 )
    out2 = leaky_relu( tf.matmul(out2, W22) + b22 )
    out2 = leaky_relu( tf.matmul(out2, W23) + b23 )
    out2 = tf.reshape(out2, [-1, dim_atom, dim_bond_type, 15])
    out2 = leaky_relu( tf.matmul(tf.transpose(out2, perm=[0,2,1,3]), tf.transpose(out2, perm=[0,2,3,1])) )
    out2 = tf.transpose(out2, perm=[0,2,3,1])
    return [out1, out2]


Y_adj = tf.placeholder(tf.float32, shape=[None, dim_atom, dim_atom, dim_bond_type])
Y_features = tf.placeholder(tf.float32, shape=[None, dim_atom, dim_atom_type])


fcn_loss_1 = tf.nn.softmax_cross_entropy_with_logits(labels=Y_features, logits=fcn(X)[0])
fcn_loss_2 = tf.nn.softmax_cross_entropy_with_logits(labels=Y_adj, logits=fcn(X)[1])
fcn_loss_2 = tf.matrix_band_part(fcn_loss_2,0,-1) - tf.matrix_band_part(fcn_loss_2,0,0)

fcn_loss = tf.reduce_mean(fcn_loss_1) + 2 * tf.reduce_mean(fcn_loss_2)

fcn_solver = (tf.train.AdamOptimizer(learning_rate=0.001)
            .minimize(fcn_loss, var_list=theta))


train_adj_array = to_categorical(np.asarray(train_adj), num_classes=dim_bond_type)
train_features_array = np.transpose(np.asarray(train_features), axes=[0,2,1])


random_idx = list(range(num_train))
shuffle(random_idx)

feature_final = feature_final[random_idx]
train_adj_array = train_adj_array[random_idx]
train_features_array = train_features_array[random_idx]

sess = tf.Session() 
sess.run(tf.global_variables_initializer())

num_epoch = 300
for it in range(num_epoch):    
    for i_batch in range(round(num_train/num_epoch)+1):
        train_sample = feature_final[i_batch * num_epoch : (i_batch+1) * num_epoch] 
        train_adj_sample = train_adj_array[i_batch * num_epoch : (i_batch+1) * num_epoch] 
        train_features_sample = train_features_array[i_batch * num_epoch : (i_batch+1) * num_epoch] 
        _, loss_curr = sess.run(
            [fcn_solver, fcn_loss],
            feed_dict={X: train_sample, Y_features: train_features_sample, Y_adj: train_adj_sample}
            )
        
    
    if it % 10 == 0:
        print('Iter: {}; loss: {:.4}'
              .format(it, loss_curr))

print("Training finished.")

# =============================================================================
# evaluation
# =============================================================================

z = []
if args.gaussianization == 'W':
    for _ in range(100000):
        z_sample = sample_z(dim_atom * dim_final_feature).reshape(dim_atom,-1)
        z.append(z_sample)
    z = np.asarray(z)
else:
    z = sample_z_full(mu=train_mu, cov=train_cov, size=100000).reshape(100000, dim_atom, -1)
samples = sess.run(fcn(X), feed_dict={X: z})
samples[0] = np.argmax(samples[0], axis=2)
samples[1] = np.argmax(samples[1], axis=3)
samples[1] = sess.run(samples[1] - tf.matrix_band_part(samples[1],0,0))
num_of_sample = 10000
atom_dict = {0: 'C', 1: 'O', 2: 'N', 3: 'F'}
mols = []
for idx in range(100000):
    node_list = samples[0][idx,:]
    adjacency_matrix = samples[1][idx,:,:]  
    where_to_cut = np.where(node_list != 4)
    node_list = node_list[where_to_cut]
    adjacency_matrix = adjacency_matrix[where_to_cut].T[where_to_cut]
    node_name = []
    for idx_node in range(len(node_list)):
        node_name.append(atom_dict[node_list[idx_node]])
    mol = MolFromGraphs(node_name, adjacency_matrix)
    if not '.' in MolToSmiles(mol):
        mols.append(mol)   
    if len(mols) == num_of_sample:
        break
    
''' validity check '''    
num_valid = 0
svgs = []
qeds = np.zeros(num_of_sample)
for idx in range(num_of_sample):
    
    temp = MolFromSmiles(MolToSmiles(mols[idx]))
    if temp is not None:
        
        mols[idx] = temp
        num_valid += 1
        qeds[idx] = qed(mols[idx])
print( "Validity is {:.2%}".format( num_valid/10000 ) )
     
''' uniqueness check '''
num_of_unique_gen = len(set([MolToSmiles(mol) for mol in mols]))
print( "Uniqueness is {:.2%}".format( num_of_unique_gen / num_of_sample ) )

''' novelty check '''
data_tgt = [MolFromSmiles(i) for i in train_data]
data_tgt += mols
num_of_novel = len(set([MolToSmiles(mol) for mol in data_tgt])) + num_of_sample - len(train_data) - num_of_unique_gen
print( "Novelty is {:.2%}".format( num_of_novel / num_of_sample ) )
                        


# =============================================================================
# draw, optional
# =============================================================================

# mols_unique = list(set([MolToSmiles(mol) for mol in mols]))
# mols_unique = [MolFromSmiles(mol) for mol in mols_unique]

# mols_uv = []
# qeds_uv = []
# for idx in range(len(mols_unique)):
#     temp = mols_unique[idx]
#     if temp is not None:
#         mols_uv.append(temp)
#         qeds_uv.append(qed(temp))
        
        
# img = Draw.MolsToGridImage(mols_uv[:25], molsPerRow=5, legends=[str("{:10.4f}".format(x)) for x in qeds_uv])
# img