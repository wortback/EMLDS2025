import numpy as np
import scipy
import scipy.stats 

#as to why is dot product and np.sum being used: 
#https://stackoverflow.com/questions/64179872/how-to-take-advantage-of-vectorization-when-computing-the-pdf-for-a-multivariate
#and yes I refuse to use scipy built in function!!
#return scipy.stats.multivariate_normal.pdf(x,mu,sigma)

def multidimGaussian(x,mu,sigma):
    return np.reciprocal(((2 * np.pi)**(int(len(mu) / 2)))* np.sqrt(np.linalg.det(sigma))) * np.exp(-0.5 * np.sum(np.transpose(x-mu).dot((np.linalg.inv(sigma) * (x-mu)))))