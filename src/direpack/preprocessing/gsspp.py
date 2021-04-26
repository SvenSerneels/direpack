#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# Created on Wed Mar 25 09:01:53 2020

# @author: Sven Serneels, Ponalytics, Mar 2020. 


__all__ = ['GenSpatialSignPrePprocessor','gen_ss_pp','gen_ss_covmat']

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.validation import check_is_fitted
from .robcent import VersatileScaler, versatile_scale
from ._preproc_utilities import *
from ..utils.utils import _check_input
from ._gsspp_utils import *
from ._gsspp_utils import _norms, _gsspp

__all__ = ['GenSpatialSignPreProcessor', 'gen_ss_covmat', 'gen_ss_pp']
    
class GenSpatialSignPreProcessor(TransformerMixin,BaseEstimator):
    
    """
    GenSpatialSignPreProcessor Generalized Spatial Sign Pre-Processing as a scikit-learn compatible object
    that can be used in ML pipelines. 
    
    Parameters
    ---------- 
        center: str or function, 
            location estimator for centring.str options: 'mean', 'median', 'l1median', 'kstepLTS', 'None' 

        fun: str or function, 
            radial transformation function, str options: 'ss' (the non-generalized spatial sign, equivalent to sklearn's Normalizer), 'ball', 'shell', 'quad' (quadratic), 'winsor', or 'linear_redescending'
            Methods: sklearn API: `fit(X)`, `transform(X)` and `fit_transform(X)` with 

        
    Attributes
    ---------- 
    Attributes always provided : 

        -  `gss_` : the generalized spatial signs 
        -  `Xm_` : the centred data 
        -  `centring_` : VersatileScaler centring object 
        -  `X_gss_pp_` : Data preprocessed by Generalized Spatial Sign
    """    

    def __init__(self,center='l1median',fun='linear_redescending'):
        
        self.center = center
        self.fun = fun 
        
    def fit(self,X): 
        
        """ 
        Calculate and store generalized spatial signs
        """
        
        X = _check_input(X)
        n,p = X.shape
        if type(self.fun) is str: 
            fun = eval(self.fun)
        else:
            fun = self.fun
        vs = VersatileScaler(center=self.center,scale='None')
        Xm = vs.fit_transform(X)
        gss_ = fun(_norms(Xm),p,n)
        setattr(self,'gss_',gss_)
        setattr(self,'Xm_',Xm)
        setattr(self,'centring_',vs)
        
    def transform(self,X):
        
        """
        Calculate Generalized Spatial Sign Pre-Pprocessed Data
        """
        
        check_is_fitted(self,('gss_','Xm_'))
        Xgss = np.multiply(self.Xm_,self.gss_)
        setattr(self,'X_gsspp_',Xgss)
        return(Xgss)

    def fit_transform(self,X): 

        self.fit(X)
        self.transform(X)
        return(self.X_gsspp_)        
        
    
    

def gen_ss_pp(X,center='l1median',fun='linear_redescending'):
    
    """
    Generalized Spatial Sign Pre-Processing as a one pass function
    Inputs: 
        X: Data matrix 
        center: str or function, location estimator for centring.
                str options: 'mean', 'median', 'l1median', 'kstepLTS', 'None' 
        fun: str or function, radial transformation function,
                str options: 'ss' (the non-generalized spatial sign, equivalent
                to sklearn's Normalizer), 'ball', 'shell', 'quad' (quadratic), 
                'winsor', or 'linear_redescending'
    Outputs: the pre-processed data 
    """    
    
    if type(center) is str: 
        center = eval(center)
            
    if type(fun) is str: 
        fun = eval(fun)
    
    X = _check_input(X)        
    n = X.shape
    if len(n) > 1:
        p = n[1]
    else:
        p = 1
    n = n[0]
    
    if center != 'None':
        X = versatile_scale(X,center=center,scale='None')
        
    return(_gsspp(X,p,n,fun=fun))
    
    
def gen_ss_covmat(X,center='kstepLTS',fun=linear_redescending): 
    
    """
    Generalized Spatial Sign Covariance Matrix 
    Is equivalent to the covariance matrix of generalized spatial sign 
    pre-processed data. 
    
    First published in: 
        A generalized spatial sign covariance matrix,
        Jakob Raymaekers, Peter Rousseeuw, 
        Journal of Multivariate Analysis, 171 (2019), 94–111.
        
    Inputs: 
        X: Data matrix 
        center: str or function, location estimator for centring.
                str options: 'mean', 'median', 'l1median', 'kstepLTS', 'None' 
        fun: str or function, radial transformation function,
                str options: 'ss' (the non-generalized spatial sign, equivalent
                to sklearn's Normalizer), 'ball', 'shell', 'quad' (quadratic), 
                'winsor', or 'linear_redescending'
                
    Outputs: the generalized spatial sign covariance matrix
    """    
    
    X = _check_input(X)  
    rc = VersatileScaler(center=center, scale='None')
    n,p = X.shape
    Xm = rc.fit_transform(X)
    Xgss = _gsspp(Xm,p,n,fun=fun)
    return(Xgss.T*Xgss/n)
    
    

    
    
        