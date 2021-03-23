#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr 13 16:08:22 2020

@author: sven
"""

import pandas as ps
import numpy as np

class MyException(Exception):
        pass

def convert_X_input(X):
    
    if type(X) == ps.core.frame.DataFrame:
            X = X.to_numpy().astype('float64')
    return(X)
            
def convert_y_input(y):

    if type(y) in [ps.core.frame.DataFrame,ps.core.series.Series]:
        y = y.to_numpy().T.astype('float64')
    return(y)


def const_xscale(beta, *args):
    X= args[0]
    h= args[1]
    i = args[2]
    j = args[3]
    beta = np.reshape(beta,(-1,h),order = 'F')
    covx = np.cov(X, rowvar=False)
    ans = np.matmul(np.matmul(beta.T,covx), beta) - np.identity(h)
    return(ans[i,j])

def const_zscale(beta, *args):
    X= args[0]
    h= args[1]
    i = args[2]
    j = args[3]
    beta = np.reshape(beta,(-1,h),order = 'F')
    covx = np.identity(X.shape[1])
    ans = np.matmul(np.matmul(beta.T,covx), beta) - np.identity(h)
    return(ans[i,j])

def _predict_check_input(Xn):
    if type(Xn) == ps.core.series.Series:
        Xn = Xn.to_numpy()
    if Xn.ndim==1:
        Xn = Xn.reshape((1,-1))
    if type(Xn) == ps.core.frame.DataFrame:
        Xn = Xn.to_numpy()
    n,p = Xn.shape
    return (n,p,Xn)

def _check_input(X): 
    
    if(type(X) in (np.matrix,ps.core.frame.DataFrame,ps.core.series.Series)): 
        X = np.array(X)
        
    if (X.dtype == np.dtype('O')):
        X = X.astype('float64')
    
    if X.ndim == 1: 
        X = X.reshape((1,-1))
    
    n,p = X.shape 
    
    if n==1:
        if p >= 2: 
            X = X.reshape((-1,1))
    return(X)