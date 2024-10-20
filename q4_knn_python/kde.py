import numpy as np


def kde(samples, h):
    """
    compute density estimation from samples with KDE and a Gaussian kernel
    Input
     samples    : (N,) vector of data points
     h          : standard deviation of the Gaussian kernel
    Output
     estimatedDensity : (200, 2) matrix of estimated density in the range of [-5, 5]
    """

    # Get the number of the samples x_n
    N = len(samples)

    # Create a linearly spaced vector of 200 points between -5 and 5
    pos = np.arange(-5.0, 5.0, 0.05)  # "x"


    #####Insert your code here for subtask 5a#####
    # Estimate the density from the samples


    # Form the output variable
    estimatedDensity = np.stack((pos, estimatedDensity), axis=1)

    return estimatedDensity
