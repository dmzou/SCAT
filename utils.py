import numpy as np
import tensorflow as tf

from numpy.random import shuffle
from sklearn.preprocessing import normalize

from sklearn.metrics import roc_auc_score
from sklearn.metrics import average_precision_score

# =============================================================================
# neural network utils to be compatible with old ver tf
# =============================================================================

# xavier initialization
def xavier_init(size):
    in_dim = size[0]
    xavier_stddev = 1. / tf.sqrt(in_dim / 2.)
    return tf.random_normal(shape=size, stddev=xavier_stddev)

# leaky relu
def leaky_relu(x, alpha=0.2):
  return tf.nn.relu(x) - alpha * tf.nn.relu(-x)


# sample gaussian noise
def sample_z(n):
    return np.random.normal(loc=0.0, scale=1.0, size=[n])

def sample_z_2d(m,n):
    return np.random.normal(loc=0.0, scale=1.0, size=[m, n])


def sample_z_full(mu, cov, size):
    gaus = np.random.multivariate_normal(mean=mu, cov=cov, size=size)
    return gaus    

# convert tensor to list
def tensor_tolist(A):
    A_list = []
    for i in range(A.shape[0]):
        A_list.append(A[i,:])
    return A_list

# create one-hot vector
def one_hot(data, n=0):
    targets = np.array(data).reshape(-1)
    return np.eye(n)[targets]
