import numpy as np


def knn(samples, k):
    """
    compute density estimation from samples with k-NN
    Input
     samples    : (N,) vector of data points
     k          : number of neighbors
    Output
     estimatedDensity : (200, 2) estimated density in the range of [-5, 5]
    """

    #####Insert your code here for subtask 5b#####
    # Get the number of the samples x_n
    N = len(samples)

    # Create a linearly spaced vector of 200 points between -5 and 5
    pos = np.arange(-5.0, 5.0, 0.05)  # "x"

    # Initialise an array to store the estimated density
    estimatedDensity = np.zeros_like(pos)

    for it, dataPoin in enumerate(pos):
        # Calc the distance from the current point to all the points in samples
        distances = np.abs(samples - dataPoin)
        
        # Find the distance to the k-th nearest neighbour
        distances.sort()
        d_k = distances[k-1]
        
        estimatedDensity[it] = k / (N * 2 * d_k)
    
    # Stack the position (x) and estimated density into a 200x2 matrix
    estimatedDensity = np.stack((pos, estimatedDensity), axis=1)

    return estimatedDensity
