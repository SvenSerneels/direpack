#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#@author: Sven Serneels, Ponalytics
# Created on Sun Feb 4 2018
# Updated on Sun Dec 16 2018
# Refactored on Sat Dec 21 2019
# Refactored on Sat Mar 28 2020


# Class for classical and robust centering and scaling of input data for 
# regression and machine learning 

# Version 2.0: Code entirely restructured compared to version 1.0. 
# Code made consistent with sklearn logic: fit(data,params) yields results. 
# Code makes more effciient use of numpy builtin estimators.
# Version 3.0:
# Code now takes strings or functions as input to centring and scaling. 
# Utility functions have been moved to _preproc_utilities.py 
# Code now supplied for l1median cetring, with options to use different 
# scipy.optimize optimization algorithms
# Version 4.0: 
# Made the API compatible for ScikitLearn pipelines. However, some nonstandard 
# functions and output remain for backwards compatibility. Functionality for
# sparse matrices still has to be implemented.  


# Ancillary functions in _preproc_utilities.py:

#         -   `scale_data(X,m,s)`: centers and scales X on center m (as vector) and scale s (as vector).
#         -   `mean(X,trimming)`: Column-wise mean.
#         -   `median(X)`: Column-wise median.
#         -   `l1median(X)`: L1 or spatial median. Optional arguments: 
#         -   `x0`: starting point for optimization, defaults to column wise median  
#         -   `method`: optimization algorithm, defaults to 'SLSQP' 
#         -   `tol`: tolerance, defaults to 1e-8
#         -   `options`: list of options for `scipy.optimize.minimize`
#         -   `kstepLTS(X): k-step LTS estimator of location.
#         -   `maxit`: int, number of iterations to compute maximally 
#         -   `tol`: float, tolerance for convergence
#         -   `std(X,trimming)`: Column-wise std.
#         -   `mad(X,c)`: Column-wise median absolute deviation, with consistency factor c.
#         -   `scaleTau2(x0, c1 = 4.5, c2 = 3, consistency = True)`: Tau estimator of scale 
#             with consistency parameters c1 and c2 and option for consistency correction
#             (True, False or 'finiteSample')



from __future__ import absolute_import, division, print_function
from __future__ import unicode_literals

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.metaestimators import _BaseComposition
from sklearn.utils.validation import check_is_fitted
import numpy as np
from ..utils.utils import MyException, convert_X_input, convert_y_input, _check_input
from ._preproc_utilities import *
from ._preproc_utilities import _check_trimming

__all__ = ['VersatileScaler','robcent','versatile_scale']

class VersatileScaler(_BaseComposition,TransformerMixin,BaseEstimator):

    """
    VersatileScaler Center and Scale data about classical or robust location and scale estimates  
    
    Parameters
    ----------
        `center`: str or callable, location estimator. String has to be name of the 
                function to be used, or 'None'. 
        `scale`: str or callable, scale estimator
        `trimming`: trimming percentage to be used in location and scale estimation. 


    Attributes
    ----------   
    Arguments for methods: 
        -   `X`: array-like, n x p, the data.
        -   `trimming`: float, fraction to be trimmed (must be in (0,1)). 
        
    

                
    Remarks
    -------
    Options for classical estimators 'mean' and 'std' also give access to robust 
    trimmed versions.


"""
    
    def __init__(self,center='mean',scale='std',trimming=0):
        
        """
        Initialize values. Check if correct options provided. 
        """
        
        self.center = center
        self.scale = scale 
        self.trimming = trimming
        
    
    def fit(self,X):
        
        """
        Estimate location and scale, store these in the class object.  
        Trimming fraction can be provided as keyword argument.
        """
        
        X = _check_input(X)
        
        _check_trimming(self.trimming)
        
        if type(self.center) is str: 
            center = eval(self.center)
        else:
            center = self.center
            
        if type(self.scale) is str: 
            scale = eval(self.scale)
        else:
            scale = self.scale
            

        n = X.shape
        if len(n) > 1:
            p = n[1]
        else:
            p = 1
        n = n[0] 
            
        if self.center == "None":
            m = np.repeat(0,p)
        else:
            m = center(X,trimming=self.trimming)
         
        # Keeping col_loc_ for older version compatibility
        setattr(self,"col_loc_",m)
        # sklearn standard 
        setattr(self,"center_",m)    
            
        if self.scale == "None":
            s = np.repeat(1,p)
        else:
            s = scale(X,trimming=self.trimming)
        
        # Keeping col_sca_ for older version compatibility
        setattr(self,"col_sca_",s)
        # sklearn standard 
        setattr(self,"scale_",s)
        
    
    def transform(self,X):
        
        """
        Center and/or scale training data to pre-estimated location and scale
        """
        
        X = _check_input(X)
        check_is_fitted(self,['center_','scale_'])
        
        Xs = scale_data(X,self.center_,self.scale_)
        setattr(self,'datas_',Xs)
            
        return Xs
        
    
    def predict(self,Xn):
        
        """
        Standardize new data on previously estimated location and scale. 
        Number of columns needs to match.
        """
        
        Xn = _check_input(Xn)
        Xns = scale_data(Xn,self.col_loc_,self.col_sca_)
        setattr(self,'datans_',Xns)
        return(Xns)
        
    def fit_transform(self,X):
        
        """
        Estimate center and scale for training data and scale these data 
        """
        
        self.fit(X)
        self.transform(X)
        return(self.datas_)
        
    def inverse_transform(self,Xs=None):
        
        """
        Transform scaled data back to their original scale  
        """
        
        check_is_fitted(self,['center_','scale_'])
        if Xs is not None:
            Xs = _check_input(Xs)
        else:
            Xs = self.datas_
        return(np.multiply(Xs,self.scale_) + self.center_)
        
        
# For backwards compatibility
robcent = VersatileScaler
            
        
def versatile_scale(X,center='l1median',scale='mad',trimming=0):
    
    """
    Wrapper to scale based on present robcent implementation that uses 
    `fit` instead of `transform`
    """
    
    rc = VersatileScaler(center=center,scale=scale,trimming=trimming)
    return(rc.fit_transform(X))
        