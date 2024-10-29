import numpy as np
from multidimGaussian import multidimGaussian


def getLogLikelihood(means, weights, covariances, X):
    # Log Likelihood estimation
    #
    # INPUT:
    # means          : Mean for each Gaussian KxD
    # weights        : Weight vector for K Gaussians
    # covariances    : Covariance matrices for each gaussian DxDxK
    # X              : Input data NxD
    # where N is number of data points
    # D is the dimension of the data points
    # K is number of gaussians
    #
    # OUTPUT:
    # logLikelihood  : Log-likelihood (a scalar)

    #####Insert your code here for subtask 6a#####


    K = len(weights)
    D = len(means[0])
    N = int(X.size / D)

    logLikelihood = 0

    for n in range(N):
        temp = 0
    
        for k in range (K):
            temp += weights[k] *  multidimGaussian(X[n], means[k], covariances[:,:,k])
        
        temp = np.log(temp)
        logLikelihood += temp

    return logLikelihood

