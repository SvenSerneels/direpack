#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Dec 16 13:44:19 2018

@author: sven
"""

import scipy.stats as sps
import scipy.spatial as spp
import numpy as np
import copy
from ..utils.utils import MyException

def trim_mean(x,trimming,axis=0):
    """
    computes the trimmed mean of array x according to axis.
    Input :
        x : input data as numpy array
        trimming, float : trimming percentage to be used
        axis, int or None : Axis along which the trimmed means are computed
        
    Output:
        The trimmed mean of x according to axis.
    
    """
    
    if trimming == 0: 
        return(np.mean(x,axis=axis))
    else:
        return(sps.trim_mean(x,trimming,axis=axis))

def trimvar(x,trimming):
    
    """
    computes the trimmed variance of array x .
    Input :
        x : input data as numpy array
        trimming, float : trimming percentage to be used
     
    Output:
        The trimmed variance of x.
    
    """
        # division by n
    return(sps.trim_mean(np.square(x - sps.trim_mean(x,trimming)),trimming))
        
def identity(x): 
    return(x)
        
def trim_mom(x,y,locest,order,trimming,option,fscorr=True):
    """
    computes trimmed comoment   between x and y. order represents the order of 
    the comoment.
    input :
        x : Input data as matrix 
        y : Input data as matrix or 1d vector
        order, int : order of the comoment
        trimming, float : trimming percentage to be used.
        option, int : option to select the type of co-moment (order 3: option 1 = com(x,x,y))
        fscor, bool: if True, a finite sample correction is applied to the comoment. 
    
    output : 
        the trimmed comoment between x and y 
    """
        # division by n
        
    if order == 0:
        como = 0
    elif order == 1: 
        como = locest(x,trimming)
    else:
        if order > 2: 
            iter_stop_2 = option 
            iter_stop_1 = order - option
        else: 
            iter_stop_1 = 1
            iter_stop_2 = 1
        
        if locest == np.median:
            trimming = 0
            factor = 1
            if (x==y).all():
                wrapper = abs
                power = 1/order
                if power == 0.5:
                    factor = 1.4826
            else:
                wrapper = identity
                power = 1
        else:
            n = len(x)
            wrapper = identity
            power = 1
            if fscorr:
                ntrim = round(n * (1-trimming)) 
                factor = ntrim
                factor /= np.product(ntrim - np.arange(max(1,order-2),order))
            else:
                factor = 1
    
        xc = wrapper(x - locest(x,trimming))
        yc = wrapper(y - locest(y,trimming))
    
        factor1 = np.power(xc,iter_stop_1)
        factor2 = np.power(yc,iter_stop_2)
    
        como = locest(np.power(np.multiply(factor1,factor2),power),trimming)*factor
#        como = sps.trim_mean(np.multiply(x - sps.trim_mean(x,trimming),y - sps.trim_mean(y,trimming)),trimming)*ntrim/(ntrim-1)
    if len(como.shape)>1: 
        como = como[0,0]
    else:
        if type(como) is np.ndarray:
            como = como[0]
            
        
    return(como)

def double_center_flex(a, center='mean', **kwargs):
    """
    Double centered function adapted to accommodate for location types different
    from mean.
    Input : 
        a : input data as matrix
        center, str : which location estimate to use for centering. either 'mean or 'median'
        kwargs :
            trimming, float : trimming percentage to be used.
            biascorr, bool : if True, bias correction is applied during double centering.
    Output : 
        The double centered version of the matrix a.

    """
    
    # print(kwargs)
    
    if 'trimming' not in kwargs:
        trimming = 0
    else:
        trimming = kwargs.get('trimming')
        # print('trimming is: ' + str(trimming))
        
    if 'biascorr' not in kwargs:
        biascorr = False
    else:
        biascorr = kwargs.get('biascorr')
    
    out = copy.deepcopy(a)
    
    dim = np.size(a, 0)
    n1 = dim

    # mu = np.sum(a) / (dim * dim)
    if center=='mean':
        mu = trim_mean(a.reshape((dim**2,1)),trimming)
        if biascorr:
            n1 = np.round(dim*(1-trimming))
            # print(n1)
            mu *= (n1**2) / ((n1-1) * (n1-2))
        mu_cols = trim_mean(a, trimming, axis=0).reshape((1,dim))
        mu_rows = trim_mean(a, trimming, axis=1).reshape((dim,1))
        if biascorr:
            mu_cols *= n1/(n1 - 2)
            mu_rows *= n1/(n1 - 2)
        mu_cols = np.ones((dim, 1)).dot(mu_cols)
        mu_rows = mu_rows.dot(np.ones((1, dim)))
    elif center=='median':
        mu = np.median(a.reshape((dim**2,1)))
        mu_cols = np.median(a,axis=0).reshape((1,dim))
        mu_rows = np.median(a,axis=1).reshape((dim,1))
        mu_cols = np.ones((dim, 1)).dot(mu_cols)
        mu_rows = mu_rows.dot(np.ones((1, dim)))
    else:
        raise(ValueError('Center should be mean or median'))
        

    # Do one operation at a time, to improve broadcasting memory usage.
    out -= mu_rows
    out -= mu_cols
    out += mu
    
    if biascorr:
        out[np.eye(dim, dtype=bool)] = 0

    return out,n1


def distance_matrix_centered(x,**kwargs):
    """
    Computes the trimmed double centered distance matrix of x.
    Input : 
        x : input data as matrix.
        kwargs :
            trimming, float : trimming percentage to be used.
            biascorr, bool : if True, bias correction is applied during double centering.
            center, str : which location estimate to use for centering. either 'mean or 'median'
            dmetric, str : which distance metric to use. Default is euclidean distance.
    Output :
        the trimmed double centered distance matrix of x            
    """
    
    if 'trimming' not in kwargs:
        trimming = 0
    else:
        trimming = kwargs.get('trimming')
            
    if 'biascorr' not in kwargs:
        biascorr = False
    else:
        biascorr = kwargs.get('biascorr')
        
    if 'center' not in kwargs:
        center = 'mean'
    else:
        center = kwargs.get('center')
        
    if 'dmetric' not in kwargs:
        dmetric = 'euclidean'
    else:
        dmetric = kwargs.get('dmetric')
        
    dx = spp.distance.squareform(spp.distance.pdist(x,metric=dmetric))
    dmx, n1 = double_center_flex(dx,biascorr=biascorr,
                                 trimming=trimming,center=center)
    
    return dmx,n1


def distance_moment(dmx,dmy,**kwargs):
    """
    Computes the trimmed distance comoment between x and y based on their distance matrices.
    Input : 
        dmx : distance matrix of x 
        dmy : distance matrix of y 
        
        kwargs :
            trimming, float : trimming percentage to be used.
            biascorr, bool : if True, bias correction is applied during double centering.
            center, str : which location estimate to use for centering. either 'mean or 'median'
            dmetric, str : which distance metric to use. Default is euclidean distance.
            order, int : order  of the comoment to be computed, default is 2 for covariance.
            option, int : option to be used during the computation. 
    Output :
        The trimmed  distance comoment  between x and y
    
    """
    
    if 'trimming' not in kwargs:
        trimming = 0
    else:
        trimming = kwargs.get('trimming')
            
    if 'biascorr' not in kwargs:
        biascorr = False
    else:
        biascorr = kwargs.get('biascorr')
        
    if 'center' not in kwargs:
        center = 'mean'
    else:
        center = kwargs.get('center')
        
    if 'order' not in kwargs:
        order = 2
    else:
        order = kwargs.get('order')
        
    if order > 2: 
        if 'option' not in kwargs:
            option = 1
        else:
            option = kwargs.get('option')
            
        iter_stop_2 = option 
        iter_stop_1 = order - option
    else:
        option = 0
        iter_stop_1 = 1
        iter_stop_2 = 1
        
    nx = dmx.shape[0]
    ny = dmy.shape[0]
    if nx!=ny:
        raise(ValueError)
        
        
    if biascorr: 
        
        if trimming == 0:
            n1 = nx
        elif 'n1' not in kwargs:
            raise(MyException('n1 needs to be provided when correcting for bias'))
        else:
            n1 = kwargs.get('n1')
        
        corr4bias = n1**2/(n1*(n1-3))
        
    else:
        corr4bias = 1
        
    if order>2:
        i = 1
        while i < iter_stop_1:  
            dmx *= dmx
            i += 1
        i = 1
        while i < iter_stop_2:  
            dmy *= dmy
            i += 1
        
        
    if center=='mean':
        moment = trim_mean((dmx*dmy).reshape((nx**2,1)),trimming)
        moment *= corr4bias
        moment = moment[0]
        moment = (-1)**order*abs(moment)**(1/order)
    elif center=='median':
        moment = np.median(dmx*dmy)
        
    return(moment)
    
    
def difference_divergence(X,Y,**kwargs):
    
    
    """
    This function computes the (U)Martingale Difference Divergence of Y given X.
    
    input :
    
        X : A  matrix or data frame, where rows represent samples, and columns represent variables.
        Y : The response variable or matrix.
        biascorr, bool : if True, uses U centering to produce an unbiased estimator of MDD
        
    output:
        returns the squared martingale difference divergence of Y given X.
       
    """
    
    if 'trimming' not in kwargs:
        trimming = 0
    else:
        trimming = kwargs.get('trimming')
        
    if 'biascorr' not in kwargs:
        biascorr = False
    else:
        biascorr = kwargs.get('biascorr')
    if 'center' not in kwargs:
        center = 'mean'
    else:
        center = kwargs.get('center')
        
    if 'dmetric' not in kwargs:
        dmetric = 'euclidean'
    else:
        dmetric = kwargs.get('dmetric')
    
   
    
    
    
    A, Adim = distance_matrix_centered(X,biascorr=biascorr,trimming=trimming,center=center)  
    dy=  spp.distance.squareform(spp.distance.pdist(Y.reshape(-1, 1),metric=dmetric)**2)
    B,Bdim = double_center_flex(0.5*dy,biascorr=biascorr,trimming=trimming,center=center)
    if biascorr:
        return(U_inner(A,B,trimming))
    else:
        return(D_inner(A,B,trimming))
       
        
def U_inner(X,Y,trimming=0):
    
    """
        Computes the inner product in the space of U centered matrices, between matrices X and Y.
        The matrices have to be  square matrices.
    
    """
        
    nx = X.shape[0]
    ny = Y.shape[0]
    
    if nx != ny:
        raise(MyException('Please feed x and y data of equal length'))
        
    #((1/(nx*(nx-3))) *(np.sum(arr)))
    arr= np.multiply(X,Y)
    arr=arr.flatten()
    lowercut = int(trimming * (nx**2))
    uppercut = (nx**2) - lowercut
    atmp = np.partition(arr, (lowercut, uppercut - 1), axis=0)
    sl = [slice(None)] * atmp.ndim
    sl[0] = slice(lowercut, uppercut)
    res= atmp[tuple(sl)]
    n = np.sqrt(len(res))
    return((1/(n*(n-3)))*np.sum(atmp[tuple(sl)], axis=0))


def D_inner(X,Y,trimming=0):
    
    """
    Computes the inner product in the space of D centered matrices, between Double centered matrices X and Y.
    The matrices have to be square matrices.
    
    """
        
    nx = X.shape[0]
    ny = Y.shape[0]
    
    if nx != ny:
        raise(MyException('Please feed x and y data of equal length'))
        
    #arr= (1/(nx*nx))*np.multiply(X,Y)
    arr= np.multiply(X,Y)
    arr=arr.flatten()
    lowercut = int(trimming * (nx**2))
    uppercut = (nx**2) - lowercut
    atmp = np.partition(arr, (lowercut, uppercut - 1), axis=0)
    sl = [slice(None)] * atmp.ndim
    sl[0] = slice(lowercut, uppercut)
    res= atmp[tuple(sl)]
    n = np.sqrt(len(res))
    return((1/(n*n))*np.sum(res, axis=0))

def MDDM(X,Y): 
    
    """Computes the MDDM(Y|X)
    for more details, see the article by Chung Eun Lee & Xiaofeng Shao;
    Martingale Difference Divergence Matrix and Its
    Application to Dimension Reduction for Stationary
    Multivariate Time Series;                                                 
    Journal of the American Statistical Association; 2018;521;
    216--229
    
    
    Input: 
    X  ---  ndarray of shape (n,p)
    Y  --- ndarray of shape(n,q)
    
    Output: 
    MDDM(Y|X)
    """

    n,q = Y.shape
    n,p = X.shape
    MDDM = np.zeros((q,q))
    Y_mean = np.mean(Y,axis=0).reshape(1,-1)
    Y_center = Y - np.matmul(np.ones((n,1)),Y_mean)
    
    for i in range(n):
        if(p==1):
            X_dist = np.abs(X[i]-X)
            
        else: 
            X_diff= (( X.T - np.vstack(X[i,:])).T)**2
            X_sum = np.sum(X_diff,axis=1)
            X_dist = np.sqrt(X_sum).reshape(-1,n)
            
        
        MDDM = MDDM + np.matmul(Y_center[i,:].reshape(q,-1), np.matmul(X_dist,Y_center))
        
    
    MDDM = (-MDDM)/(n**2)
    
    return(MDDM)
