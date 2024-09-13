import pandas as pd
import numpy as np
from scipy import stats
from scipy.stats import rankdata as rank 
from scipy.stats import beta
import timeit

class EmpiricalBetaCopula:
    def __init__(self, data_orig, ties="average"):
        """
        Parameters
        ----------
        data: ndarray
            Observed multi-dimensional data
            
        method: { 'average', 'min', 'max', 'dense', 'ordinal' }
            Method used to assign ranks to tied elements.
        """
        self.data_orig = np.asarray(data_orig)
        self.ties = ties
        self.data_rank = self.rank_data()
        
    def rank_data(self):
        """Returns rank matrix of the original data"""
        (N,d) = self.data_orig.shape
        data_rank = np.ndarray(shape = (N,d), dtype = int) 
        for i in range(0,d):
            data_rank[:,i] = rank(self.data_orig[:,i],self.ties)
        return data_rank
    
    def pdf(self, u):
        """
        Parameters
        ----------
        u: ndarray
            Vector of the pseudo-observations of the observed data. (n x d)
            Must have values between 0 and 1. 

        Returns
        -------
        ndarray
            The pdf of the random variates
        """
        start = timeit.default_timer()
        n = len(self.data_rank)
        output = np.array([sum([np.prod(beta.pdf(row, a=row_rank, b=n + 1 - row_rank),axis=1)
            for row_rank in self.data_rank]) for row in u]) / n
        stop = timeit.default_timer()
        print("Runtime PDF Empirical Beta: ", np.round((stop-start),2))
        return output
    
    def cdf(self, u):
        start = timeit.default_timer()
        n = len(self.data_rank)
        output = np.array([sum([np.prod(beta.cdf(row, a=row_rank, b=n + 1 - row_rank),axis=1)
            for row_rank in self.data_rank]) for row in u]) / n
        stop = timeit.default_timer()
        print("Runtime CDF Empirical Beta: ", np.round((stop-start),2))
        return output  
    
    def sample(self,n,seed=1234):
        """
        sample n points from C(u)
        """
        (N,d) = self.data_orig.shape
        samples = np.empty((n,d))
        for i in range(n):
            I = np.random.randint(0,N,1)
            #V = np.empty((1,d))
            for j in range(d):
                a = self.data_rank[I,j]; b = N+1-a
                #V[0,j] = beta.rvs(a=a,b=b,size=1,random_state=seed+i)
                samples[i,:] = beta.rvs(a=a,b=b,size=1,random_state=seed+i)
        #samples[i,:] = beta.rvs(a=a,b=b,size=1,random_state=seed+i)
        return samples


