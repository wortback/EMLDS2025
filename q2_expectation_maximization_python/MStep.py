import numpy as np
from getLogLikelihood import getLogLikelihood
from regularize_cov import regularize_cov


def MStep(gamma, X):
    # Maximization step of the EM Algorithm
    #
    # INPUT:
    # gamma          : NxK matrix of responsibilities for N datapoints and K Gaussians.
    # X              : Input data (NxD matrix for N datapoints of dimension D).
    #
    # N is number of data points
    # D is the dimension of the data points
    # K is number of Gaussians
    #
    # OUTPUT:
    # logLikelihood  : Log-likelihood (a scalar).
    # means          : Mean for each gaussian (KxD).
    # weights        : Vector of weights of each gaussian (1xK).
    # covariances    : Covariance matrices for each component(DxDxK).

    #####Insert your code here for subtask 6c#####

    K = len(gamma[0])
    D = len(X[0])
    N = int(X.size / D)


    means = np.zeros((K,D))
    weights = np.zeros(K)
    covariances = np.zeros((D,D,K))
    n_ks = np.zeros(K)
    
    #determining means

    for k in range(K):
        n_k = np.sum(gamma[:, k])
        means[k] = np.sum(gamma[:, k][:, np.newaxis] * X, axis=0) / n_k

    #determining covariances 

    for k in range(K):
        temp = np.zeros((D, D))  
        n_k = np.sum(gamma[:, k])  
        
        if n_k > 0:  # only compute if there are assigned points (this broke me fr)
            for n in range(N):
                diff = X[n] - means[k]
                temp += gamma[n, k] * np.outer(diff, diff) 
                
            covariances[:, :, k] = temp / n_k 
        else:
            covariances[:, :, k] = np.eye(D) 
    
    #determining weights

    for k in range(K):
         n_k = np.sum(gamma[:, k])  
         weights[k] = n_k / N 
    
    logLikelihood = getLogLikelihood(means,weights,covariances,X)


    return weights, means, covariances,logLikelihood
