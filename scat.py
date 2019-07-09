# -*- coding: utf-8 -*-
import numpy as np
import networkx as nx
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler, normalize

# =============================================================================
# scattering transform
# =============================================================================
    
''' get one-step scattering coefficients using Paley-Littlewood wavelet '''
def propLayer(f, lamb, V, K=4, N=1):
    # K: scale; N: bandwidth
    idx = []
    for k in range(K):
        if k == 0:
            idx.append( lamb < N )
        else:
            idx.append( (lamb >= 2**(k-1)*N) * (lamb < 2**k*N) )

    y = []
    for k in range(K):
        y.append( np.matmul(np.matmul(V[:,idx[k]], V[:,idx[k]].T), f) )

    return y


''' get one-step scattering coefficients using a general wavelet '''
''' change the name of the function propLayerHaar to propLayer in order to use '''
''' using haar wavelet as an example, replace it with any wavelet '''
def phi(lamb):
    phi = np.sinc(2*lamb)
    return phi

def psi(lamb):
    psi = np.sinc(lamb) * ( 1 - np.cos(np.pi*lamb) )
    return psi

def propLayerHaar(f, lamb, V, J=3): # to replace propLayer
    y = []
    for k in range(J):
        j = J - k
        if j == J:
            H = phi(2**j * lamb)
        else:
            H = psi(2**(-j) * lamb)
        H = np.diag(H)
        y.append( np.matmul(np.matmul(np.matmul(V, H), V.T), f) )
    return y


''' get all scattering coefficients '''
def getRep(f, lamb, V, layer=3, N=1):
    y_out = []
    y_next = []
    y = propLayer(f, lamb, V, N=N)
    y_out.append(y.pop(0))
    y_next.extend(y)
    for i in range(layer-1):
        for k in range(len(y_next)):
            ftemp = y_next.pop(0)
            ftemp = np.absolute(ftemp)
            y = propLayer(ftemp, lamb, V, N=N)
            y_out.append(y.pop(0))
            y_next.extend(y)
    y_out = np.concatenate(tuple(y_out), axis=1) # use this to form a single matrix
    return y_out

# =============================================================================
# diffusion transform
# =============================================================================

def diffusion_scat(f, W, K=3, t=3, layer=3):
    G = nx.from_scipy_sparse_matrix(W)
    D = np.array([ np.max((d,1)) for (temp,d) in list(G.degree) ])
    Dhalf = np.diag( 1/np.sqrt( D ) )
    A = np.matmul( np.matmul( Dhalf , W.todense() ) , Dhalf ) 
    T = ( np.eye(np.shape(D)[0]) + A ) / 2
    U = np.linalg.matrix_power(T, t)
    psi = []
    for idx in range(K):
        if idx == 0:
            psi.append( np.eye(np.shape(D)[0]) - T )
        else:
            T0 = T
            T = np.matmul(T0,T0)
            psi.append( T0 - T )
            
    y_next = [ f ]
    y_out = [ np.matmul( U , np.absolute(f) ) ]

    for i in range(layer-1):
        for k in range(len(y_next)):
            y_next_new = []
            ftemp = y_next.pop(0)
            ftemp = np.absolute(ftemp)
            y = [ np.matmul( fltr , ftemp ) for fltr in psi  ]
            y_out.extend( [ np.matmul( U , np.absolute(y_temp) ) for y_temp in y ] )
            y_next_new.extend( y )
        y_next = y_next_new
    y_out = np.concatenate(tuple(y_out), axis=0) # use this to form a single matrix
    return y_out


# =============================================================================
# gaussianization
# =============================================================================
    
def gaussianization_whiten(A, pca=True, num_of_components=8):
    '''A is data matrix with size (# of sample) X (# of dimension)'''
    if not pca:
        num_of_components = A.shape[1]
    pca = PCA(n_components=num_of_components)
    pca.fit(A)
    A_mean = pca.mean_
    V = pca.components_
    lamb = pca.explained_variance_
    lamb_invhalf = 1 / np.sqrt(lamb)
    Sigma_invhalf = np.matmul(np.diag(lamb_invhalf), V)
    A_gaussian = np.matmul(Sigma_invhalf, (A-A_mean).T)
    return A_gaussian.T


def gaussianization_spherize(A, pca=True, num_of_components=8):
    '''A is data matrix with size (# of sample) X (# of dimension)'''       
    
    A = A - np.mean(A, axis=0)
    if pca:
        pca_model = PCA(n_components=num_of_components)
        A = pca_model.fit_transform(A)
    return normalize(A)
