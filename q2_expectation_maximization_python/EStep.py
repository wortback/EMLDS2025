import numpy as np
from getLogLikelihood import getLogLikelihood
from multidimGaussian import multidimGaussian


def EStep(means, covariances, weights, X):
    # Expectation step of the EM Algorithm
    #
    # INPUT:
    # means          : Mean for each Gaussian KxD
    # weights        : Weight vector 1xK for K Gaussians
    # covariances    : Covariance matrices for each Gaussian DxDxK
    # X              : Input data NxD
    #
    # N is number of data points
    # D is the dimension of the data points
    # K is number of Gaussians
    #
    # OUTPUT:
    # logLikelihood  : Log-likelihood (a scalar).
    # gamma          : NxK matrix of responsibilities for N datapoints and K Gaussians.

    #####Insert your code here for subtask 6b#####

    K = len(weights)
    D = len(means[0])
    N = int(X.size / D)

    logLikelihood = getLogLikelihood(means,weights,covariances,X)
    gamma = np.zeros((N,K))


    for n in range(N):
        underSum = 0
        for j in range(K):
            underSum += weights[j] * multidimGaussian(X[n],means[j],covariances[:,:,j])
        for k in range(K):
            actualSum = (weights[k] * multidimGaussian(X[n], means[k], covariances[:,:,k])) / underSum
            gamma[n,k] = actualSum
    
    return [logLikelihood, gamma]
