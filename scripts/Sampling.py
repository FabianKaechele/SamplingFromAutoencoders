import numpy as np
from Copula import *
from scipy.stats import gaussian_kde
from scipy.special import ndtr
import scipy.interpolate as interpolate
import math
import torch


from sklearn.mixture import GaussianMixture
from sklearn.neighbors import KernelDensity
from sklearn.model_selection import GridSearchCV


def indep_time(z, y, n:int, seed=500):
   
    """
    Parameters
    ----------
    z: ndarray / tensor
        samples from multivariate distribution
    y: ndarray / tensor
        samples from multiple margins
    Returns
    -------
    tensor
        samples from new multivariate distribution
    """
    # check input type
    if type(z) != np.ndarray:
        z = z.detach().numpy()
    if type(y) != np.ndarray:
        y = y.detach().numpy()
    ## sample u from empirical beta copula
    #print('get copula')
    
    s0=timeit.default_timer()
    u = np.random.uniform(size=(n,y.shape[1]))
    sample_time=timeit.default_timer()-s0
    learning_time=0
   # print('got copula samples')
    ## estimate distribution of y with kde
    y_sample = np.empty((n,))
    for i,yi in enumerate(y.T): #enumerate through columns of y, d times
        #print("loop {}".format(i))
        s0=timeit.default_timer()
        kde = gaussian_kde(yi, bw_method="silverman") 
        bw = kde.factor
        xvalue = np.linspace(np.min(yi)-5*bw, np.max(yi)+5*bw, 400)
        cdf = tuple(ndtr(np.ravel(item - yi) / bw).mean() for item in xvalue)
        
        try:
            icdf = interpolate.interp1d(cdf, xvalue, kind='linear')
            
        except ValueError as e:
            print(e)
            if str(e) == "A value in x_new is below the interpolation range.":
                n_ = 5; error = True
                while error == True:
                    try:
                        error = False
                        y_sample_ = icdf(u[:,i]) #(n,0)                    
                    except ValueError:
                        n_ = n_+0.2
                        xvalue = np.linspace(np.min(yi)-n_*bw, np.max(yi)+5*bw, 400)
                        cdf = tuple(ndtr(np.ravel(item - yi) / bw).mean() for item in xvalue)
                        icdf = interpolate.interp1d(cdf, xvalue, kind='linear')
                        error = True
            if str(e) == "A value in x_new is above the interpolation range.":
                print(e)
                n_ = 5; error = True
                while error == True:
                    try:
                        error = False
                        y_sample_ = icdf(u[:,i]) #(n,0)                    
                    except ValueError:
                        n_ = n_+0.2
                        xvalue = np.linspace(np.min(yi)-5*bw, np.max(yi)+n_*bw, 400)
                        cdf = tuple(ndtr(np.ravel(item - yi) / bw).mean() for item in xvalue)
                        icdf = interpolate.interp1d(cdf, xvalue, kind='linear')
                        error = True
            if str(e) == "Expect x to not have duplicates":
                print(e)
                n_ = 400; error = True
                while error == True:
                    try:
                        error = False
                        icdf = interpolate.interp1d(cdf, xvalue, kind='linear')
                    except ValueError:
                        n_ = n_-10
                        xvalue = np.linspace(np.min(yi)-5*bw, np.max(yi)+5*bw, n_)
                        cdf = tuple(ndtr(np.ravel(item - yi) / bw).mean() for item in xvalue)
                        error = True
                    
                    
        learning_time=learning_time+timeit.default_timer()-s0
        
        s0=timeit.default_timer()
        y_sample_ = icdf(u[:,i])
        y_sample = np.vstack([y_sample,y_sample_])
        sample_time=sample_time+timeit.default_timer()-s0
        
    y_sample = y_sample.T[:,1:]
    
    return torch.tensor(y_sample).float(),learning_time,sample_time
    
    
def indep_sampling(z, y, n:int, seed=500):
   
    """
    Parameters
    ----------
    z: ndarray / tensor
        samples from multivariate distribution
    y: ndarray / tensor
        samples from multiple margins
    Returns
    -------
    tensor
        samples from new multivariate distribution
    """
    # check input type
    if type(z) != np.ndarray:
        z = z.detach().numpy()
    if type(y) != np.ndarray:
        y = y.detach().numpy()

    
    # get i.i.d. uniform samples
    u = np.random.uniform(size=(n,y.shape[1]))
    
   # print('got copula samples')
    ## estimate distribution of y with kde
    y_sample = np.empty((n,))
    for i,yi in enumerate(y.T): #enumerate through columns of y, d times
        
        kde = gaussian_kde(yi, bw_method="silverman") 
        bw = kde.factor
        xvalue = np.linspace(np.min(yi)-5*bw, np.max(yi)+5*bw, 400)
        cdf = tuple(ndtr(np.ravel(item - yi) / bw).mean() for item in xvalue)
        
        try:
            icdf = interpolate.interp1d(cdf, xvalue, kind='linear')
            
        except ValueError as e:
            print(e)
            if str(e) == "A value in x_new is below the interpolation range.":
                n_ = 5; error = True
                while error == True:
                    try:
                        error = False
                        y_sample_ = icdf(u[:,i]) #(n,0)                    
                    except ValueError:
                        n_ = n_+0.2
                        xvalue = np.linspace(np.min(yi)-n_*bw, np.max(yi)+5*bw, 400)
                        cdf = tuple(ndtr(np.ravel(item - yi) / bw).mean() for item in xvalue)
                        icdf = interpolate.interp1d(cdf, xvalue, kind='linear')
                        error = True
            if str(e) == "A value in x_new is above the interpolation range.":
                print(e)
                n_ = 5; error = True
                while error == True:
                    try:
                        error = False
                        y_sample_ = icdf(u[:,i]) #(n,0)                    
                    except ValueError:
                        n_ = n_+0.2
                        xvalue = np.linspace(np.min(yi)-5*bw, np.max(yi)+n_*bw, 400)
                        cdf = tuple(ndtr(np.ravel(item - yi) / bw).mean() for item in xvalue)
                        icdf = interpolate.interp1d(cdf, xvalue, kind='linear')
                        error = True
            if str(e) == "Expect x to not have duplicates":
                print(e)
                n_ = 400; error = True
                while error == True:
                    try:
                        error = False
                        icdf = interpolate.interp1d(cdf, xvalue, kind='linear')
                    except ValueError:
                        n_ = n_-10
                        xvalue = np.linspace(np.min(yi)-5*bw, np.max(yi)+5*bw, n_)
                        cdf = tuple(ndtr(np.ravel(item - yi) / bw).mean() for item in xvalue)
                        error = True
                    
                    
        
        y_sample_ = icdf(u[:,i])
        y_sample = np.vstack([y_sample,y_sample_])

        
    y_sample = y_sample.T[:,1:]
    
    return torch.tensor(y_sample).float()

def sampling1_time(z, y, n:int, seed=500):
   
    """
    Parameters
    ----------
    z: ndarray / tensor
        samples from multivariate distribution
    y: ndarray / tensor
        samples from multiple margins
    Returns
    -------
    tensor
        samples from new multivariate distribution
    """
    # check input type
    if type(z) != np.ndarray:
        z = z.detach().numpy()
    if type(y) != np.ndarray:
        y = y.detach().numpy()
    ## sample u from empirical beta copula
    #print('get copula')
  
    
    
    s0=timeit.default_timer()
    cop = EmpiricalBetaCopula(z)
    learning_time=timeit.default_timer()-s0
    
    s0=timeit.default_timer()
    u = cop.sample(n,seed = seed)
    sample_time=timeit.default_timer()-s0
   # print('got copula samples')
    ## estimate distribution of y with kde
    y_sample = np.empty((n,))
    for i,yi in enumerate(y.T): #enumerate through columns of y, d times
        #print("loop {}".format(i))
        s0=timeit.default_timer()
        kde = gaussian_kde(yi, bw_method="silverman") 
        bw = kde.factor
        xvalue = np.linspace(np.min(yi)-5*bw, np.max(yi)+5*bw, 400)
        cdf = tuple(ndtr(np.ravel(item - yi) / bw).mean() for item in xvalue)
        
        try:
            icdf = interpolate.interp1d(cdf, xvalue, kind='linear')
            
        except ValueError as e:
            print(e)
            if str(e) == "A value in x_new is below the interpolation range.":
                n_ = 5; error = True
                while error == True:
                    try:
                        error = False
                        y_sample_ = icdf(u[:,i]) #(n,0)                    
                    except ValueError:
                        n_ = n_+0.2
                        xvalue = np.linspace(np.min(yi)-n_*bw, np.max(yi)+5*bw, 400)
                        cdf = tuple(ndtr(np.ravel(item - yi) / bw).mean() for item in xvalue)
                        icdf = interpolate.interp1d(cdf, xvalue, kind='linear')
                        error = True
            if str(e) == "A value in x_new is above the interpolation range.":
                print(e)
                n_ = 5; error = True
                while error == True:
                    try:
                        error = False
                        y_sample_ = icdf(u[:,i]) #(n,0)                    
                    except ValueError:
                        n_ = n_+0.2
                        xvalue = np.linspace(np.min(yi)-5*bw, np.max(yi)+n_*bw, 400)
                        cdf = tuple(ndtr(np.ravel(item - yi) / bw).mean() for item in xvalue)
                        icdf = interpolate.interp1d(cdf, xvalue, kind='linear')
                        error = True
            if str(e) == "Expect x to not have duplicates":
                print(e)
                n_ = 400; error = True
                while error == True:
                    try:
                        error = False
                        icdf = interpolate.interp1d(cdf, xvalue, kind='linear')
                    except ValueError:
                        n_ = n_-10
                        xvalue = np.linspace(np.min(yi)-5*bw, np.max(yi)+5*bw, n_)
                        cdf = tuple(ndtr(np.ravel(item - yi) / bw).mean() for item in xvalue)
                        error = True
                    
                    
        learning_time=learning_time+timeit.default_timer()-s0
        
        s0=timeit.default_timer()
        y_sample_ = icdf(u[:,i])
        y_sample = np.vstack([y_sample,y_sample_])
        sample_time=sample_time+timeit.default_timer()-s0
    y_sample = y_sample.T[:,1:]
   
    return torch.tensor(y_sample).float(),learning_time,sample_time


def sampling1(z, y, n:int, seed=500):
    """
    Parameters
    ----------
    z: ndarray / tensor
        samples from multivariate distribution
    y: ndarray / tensor
        samples from multiple margins

    Returns
    -------
    tensor
        samples from new multivariate distribution
    """
    # check input type
    if type(z) != np.ndarray:
        z = z.detach().numpy()
    if type(y) != np.ndarray:
        y = y.detach().numpy()
    ## sample u from empirical beta copula
    print('get copula')
    cop = EmpiricalBetaCopula(z)
    u = cop.sample(n,seed = seed) 
    print('got copula samples')
    ## estimate distribution of y with kde
    y_sample = np.empty((n,))
    for i,yi in enumerate(y.T): #enumerate through columns of y, d times
        #print("loop {}".format(i))
        kde = gaussian_kde(yi, bw_method="silverman") 
        bw = kde.factor
        xvalue = np.linspace(np.min(yi)-5*bw, np.max(yi)+5*bw, 400)
        cdf = tuple(ndtr(np.ravel(item - yi) / bw).mean() for item in xvalue)
        
        try:
            icdf = interpolate.interp1d(cdf, xvalue, kind='linear')
            y_sample_ = icdf(u[:,i])
        except ValueError as e:
            print(e)
            if str(e) == "A value in x_new is below the interpolation range.":
                n_ = 5; error = True
                while error == True:
                    try:
                        error = False
                        y_sample_ = icdf(u[:,i]) #(n,0)                    
                    except ValueError:
                        n_ = n_+0.2
                        xvalue = np.linspace(np.min(yi)-n_*bw, np.max(yi)+5*bw, 400)
                        cdf = tuple(ndtr(np.ravel(item - yi) / bw).mean() for item in xvalue)
                        icdf = interpolate.interp1d(cdf, xvalue, kind='linear')
                        error = True
            if str(e) == "A value in x_new is above the interpolation range.":
                print(e)
                n_ = 5; error = True
                while error == True:
                    try:
                        error = False
                        y_sample_ = icdf(u[:,i]) #(n,0)                    
                    except ValueError:
                        n_ = n_+0.2
                        xvalue = np.linspace(np.min(yi)-5*bw, np.max(yi)+n_*bw, 400)
                        cdf = tuple(ndtr(np.ravel(item - yi) / bw).mean() for item in xvalue)
                        icdf = interpolate.interp1d(cdf, xvalue, kind='linear')
                        error = True
            if str(e) == "Expect x to not have duplicates":
                print(e)
                n_ = 400; error = True
                while error == True:
                    try:
                        error = False
                        icdf = interpolate.interp1d(cdf, xvalue, kind='linear')
                    except ValueError:
                        n_ = n_-10
                        xvalue = np.linspace(np.min(yi)-5*bw, np.max(yi)+5*bw, n_)
                        cdf = tuple(ndtr(np.ravel(item - yi) / bw).mean() for item in xvalue)
                        error = True
        y_sample_ = icdf(u[:,i])
        y_sample = np.vstack([y_sample,y_sample_])
    y_sample = y_sample.T[:,1:]
    return torch.tensor(y_sample).float() 

