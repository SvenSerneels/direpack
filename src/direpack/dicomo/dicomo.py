#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Dec  2 13:14:51 2018

@author: Sven Serneels, Ponalytics 
"""
import numpy as np
import scipy.stats as sps
import scipy.spatial as spp
from sklearn.base import BaseEstimator
from sklearn.utils.metaestimators import _BaseComposition
from statsmodels import robust as srs
from ._dicomo_utils import *

class MyException(Exception):
        pass

class dicomo(_BaseComposition,BaseEstimator):
    
    """
    DICOMO Diverse set of Co-Moments 
    
    Inputs
    mode, str: which (co-)moment to estimate. General options are 'mom' or 'com'
        (univariate of bivariate). Many other specific flags are allowed that will 
        specifically scale the co-moments, or are just shortcuts to 'mom'/'com':
            'var'       - variance 
            'std'       - standard deviation 
            'skew'      - skewness 
            'kurt'      - kurtosis 
            'cov'       - co-variance 
            'cos'       - co-skewness
            'cok'       - co-kurtosis 
            'corr'      - correlation 
            'continuum' - continuum = com * mom^(alpha-1), alpha defaults to 1
                          and can be passed as an argument to fit 
            'M3'        - non-scaled third order co-moment 
        Third and fourth order co-moments are by default the 'x-heavy' version,
        i.e. M3(x,x,y). To access e.g. M3(x,y,y), pass option=2 to fit function. 
        
    est, str: which type of co-moment to estimate. Presently defaults to 
    'arithmetic', will be extended to more options. 
    
    center, str: Which internal location estimator to use in (co-)moment 
        calculation. Works with 'mean' and 'median'; to access trimmed means,
        pass trimming = k, k>0 to fit function. 
        
    Example
        covest = dicomo(mode='com')
        covest.fit(x,y=y,order=3,option=1,trimming=.1)
        
    Returns: float with estimated moment. Also stored as output moment_. 
        (Co-)moments estimated in the course of calculating the result will 
        also be stored in the object, e.g. if est = 'corr', center = 'mean', 
            moment_     = correlation coefficient 
            co_moment_  = co-variance 
            x_moment_   = x std
            y_moment_   = y std 
    
    """

    def __init__(self,est='arithmetic',mode='mom',center='mean'):
        self.mode = mode
        self.est=est
        self.center=center
        self.liest = ['arithmetic','distance','sign','entropy']
        self.limo = ['mom','var','std','skew','kurt','com','cov','cok','cos','corr','continuum','M3']
        self.licenter = ['mean','median']
        if not(self.mode in self.limo):
            raise(MyException("Only models allowed are: 'mom','var','skew','kurt','com','cov','cos','cok','corr','continuum','M3'"))
        if not(self.est in self.liest):
            raise(MyException('Only estimator classes allowed are: "arithmetic", "distance", "sign", "entropy"'))
        if not(self.center in self.licenter):
            raise(MyException('Only centring classes allowed are: "mean", "median"'))
        
        
        
    def fit(self,x,**kwargs):
        
        """
        Fit function takes kwargs: 
            
            y: y data 
            
            trimming, float, as fraction (e.g. .15 for 15% trimming)
            
            biascorr, bool. Bias correction at normal distribution (e.g. (n-1) 
                division for co-variance if True, (n) division is False))
            
            dmetric, str. Internal distance metric used. Defaults to 'euclidean' 
            
            order, int: order of co-moemnt to estimate. E.g. mode='com', order=2
                is equivalent to mode='cov' 
                
            option, int: which version of co-moment to estimate. in np.arange(1,5). 
                e.g. option = 2 yields M3(x,y,y) or M4(x,x,y,y). 
                
            calcmode, str: defaults to 'fast'. Not all options have different 
                calculation modes. 
                
        Kwargs only relevant for continuum:  
            
            alpha, float. continuum parameter.
                
        Kwargs only relevant for kurtosis: 
            
            Fisher, bool: use Fisher's "-3" correction if True.  
        
        
        """
        
        
        if 'trimming' not in kwargs:
            trimming = 0
        else:
            trimming = kwargs.get('trimming')
            
        if 'biascorr' not in kwargs:
            biascorr = False
        else:
            biascorr = kwargs.get('biascorr')
            
        if 'alpha' not in kwargs:
            alpha = 1
        else:
            alpha = kwargs.get('alpha')
            
        if 'dmetric' not in kwargs:
            dmetric = 'euclidean'
        else:
            dmetric = kwargs.get('dmetric')
            
        if 'calcmode' not in kwargs:
            calcmode = 'fast'
        else:
            calcmode = kwargs.get('calcmode')
            
        if 'order' not in kwargs:
            order = 2
        else:
            order = kwargs.get('order')
            
        if self.mode == 'var':
            mode = 'mom' 
            order=2
        elif self.mode == 'cov':
            mode = 'com'
            order=2
        elif self.mode == 'std':
            self.center = 'mean'
            mode = 'mom'
            order=2
            num_power = 0.5
        elif self.mode == 'skew':
            self.center = 'mean'
            mode = 'mom'
            order = 3
            num_power = 1.5
        elif self.mode == 'kurt':
            self.center = 'mean'
            mode = 'mom'
            order = 4
            num_power = 2
            if 'Fisher' not in kwargs:
                Fisher = True
            else:
                Fisher = kwargs.get('Fisher')
        elif self.mode == 'cos':
            self.center = 'mean'
            mode = 'com'
            order = 3
            if 'standardized' not in kwargs:
                standardized = True
            else:
                standardized = kwargs.get('standardized')
        elif self.mode == 'M3':
            self.center = 'mean'
            mode = 'com'
            order = 3
            if 'standardized' not in kwargs:
                standardized = False
            else:
                standardized = kwargs.get('standardized')
        elif self.mode == 'cok':
            self.center = 'mean'
            mode = 'com'
            order = 4
            if 'standardized' not in kwargs:
                standardized = True
            else:
                standardized = kwargs.get('standardized')
            if 'Fisher' not in kwargs:
                Fisher = True
            else:
                Fisher = kwargs.get('Fisher')
        else:
            mode = self.mode
            
        if order > 2: 
            if 'option' not in kwargs:
                option = 1
            else:
                option = kwargs.get('option')
        else:
            option = 0
            
        n = len(x)
        ntrim = round(n * (1-trimming)) 
        
        if len(x.shape)==1:
            x = np.matrix(x).reshape((n,1))
        
        if mode=='corr':
            alpha = 1
        
        if n==0:
            raise(MyException('Please feed data with length > 0'))
            
        if self.center == 'median':
            locest = np.median
        else:
            locest = trim_mean
        
        # Classical variance, covariance and continuum as well as robust alternatives
        if self.est=='arithmetic':
            
            # X moment 
            if mode!='com':
                xmom = trim_mom(x,x,locest,order,trimming,option,biascorr) 
                self.x_moment_ = xmom
                
                # Standardization
                if self.mode in ('std','skew','kurt'):
                    x2mom = trim_mom(x,x,locest,2,trimming,option,False)
                    xmom /= (x2mom**num_power) 
                    if biascorr:
                        if self.mode == 'skew':
                            xmom *= (ntrim-1)**2
                            xmom /= np.sqrt(ntrim**2 - ntrim)
                        elif self.mode == 'kurt':
                            xmom = xmom*ntrim - xmom/ntrim
                            xmom -= 3*(ntrim-1)**2.0 / ((ntrim-2)*(ntrim-3))
                            if not Fisher:
                                xmom += 3
                
                # Store Output 
                if mode=='mom':
                    self.moment_ = xmom
            
            # Covariance or continuum
            if mode!='mom':
                
                # get y data and run checks 
                if 'y' not in kwargs:
                    raise(MyException('Please supply second data vector'))
                else:
                    y = kwargs.get('y')
                
                n1 = len(y)
                if n1==0:
                    raise(MyException('Please feed data with length > 0'))
                if n1!=n:
                    raise(MyException('Please feed x and y data of equal length'))
                if len(y.shape)==1:
                    y = np.matrix(y).reshape((n,1))
                
                # calculate co-moment 
                como = trim_mom(x,y,locest,order,trimming,option,biascorr)
                
                self.co_moment_ = como
                
                # Bias correction 
                if (biascorr and (order > 2)):
                    como *= ntrim
                
                # Store output 
                if mode=='com':
                    self.moment_ = como
                
                # Standardization    
                if self.mode in ('cok','cos'):
                    x2sd = np.sqrt(trim_mom(x,x,locest,2,trimming,option,biascorr))
                    y2sd = np.sqrt(trim_mom(y,y,locest,2,trimming,option,biascorr))
                        
                    if ((self.mode == 'cok') and biascorr): # biascorr is only exact if standardized
                        como -= como/(ntrim**2)
                        como -= 3*(ntrim-1)**2.0 / ((ntrim-2)*(ntrim-3))

                # Calculate y moment when needed
                if mode in ['corr','continuum']:
                
                    ymom = trim_mom(y,y,locest,order,trimming,option,biascorr)
                    
                    self.y_moment_ = ymom
                    
        # Calculate composite moments when needed
        if mode == 'corr': 
            como /= (np.sqrt(xmom)*np.sqrt(ymom))
            self.moment_=como
        elif mode == 'continuum':
            como *= como * (np.sqrt(xmom)**(alpha -1))
            self.moment_=como
        
        # Standardization where not yet done 
        if (self.mode in ('cok','cos') and standardized):
            iter_stop_2 = option 
            iter_stop_1 = order - option
            como /= np.power(x2sd,iter_stop_1)
            como /= np.power(y2sd,iter_stop_2)
            if ((self.mode == 'cok') and not Fisher): # Not very meaningful for co-moment
                como += 3
            self.moment_=como
        
                        
        # Store output 
        if type(self.moment_)==np.ndarray:
            self.moment_ = self.moment_[0]
        return(self.moment_)
        
                    
                    
                
            
        
            
            
                
        
        
        