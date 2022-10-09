#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# Created on Sun Dec  2 13:14:51 2018

# @author: sven

import numpy as np
import scipy.stats as sps
import scipy.spatial as spp
from sklearn.base import BaseEstimator
from sklearn.utils.metaestimators import _BaseComposition
from statsmodels import robust as srs
from ._dicomo_utils import *
import dcor as dc
# Optional: if Ballcov required, import Ball 
# import Ball

class MyException(Exception):
        pass

class dicomo(_BaseComposition,BaseEstimator):
    
    """
    The `dicomo` class implements (co)-moment statistics, covering both clasical product-moment 
    statistics, as well as more recently developed energy statistics. 
    The `dicomo` class also serves as a plug-in into `capi` and  `ppdire`. 
    It has been written consistently with `ppdire` such that it provides a wide range of 
    projection indices based on (co-)moments. Ancillary functions for (co-)moment 
    estimation are in `_dicomo_utils.py`.
    
    Parameters
    ------------ 

        est :  str
             mode of estimation. The set of options are `'arithmetic'` (product-moment)  or `'distance'` (energy statistics)

        mode : str
                 type of moment. Options include :
                * `'mom'`: moment 
                * `'var'`: variance 
                * `'std'`: standard deviation 
                * `'skew'`: skewness 
                * `'kurt'`: kurtosis
                * `'com'`: co-moment 
                * `'M3'`: shortcut for third order co-moment
                * `'cov'`: covariance 
                * `'cos'`: co-skewness
                * `'cok'`: co-kurtosis 
                * `'corr'`: correlation, 
                * `'continuum'`: continuum association 
                * `'mdd'`: martingale difference divergence (requires `est = 'distance'`)
                * `'mdc'`: martingale difference correlation (requires `est = 'distance'`)
                * `'ballcov'`: ball covariance (requires installing `Ball` and uncommenting the `import` statement)

        center : str 
                 internal centring used in calculation. Options are `mean` or `median`.  

    Attributes
    ------------
    Attributes always provided: 

        - `moment_`: The resulting (co-)moment Depending on the options picked, intermediate results are stored as well, such as `x_moment_`, `y_moment_` or `co_moment_`
    
    """
    

    def __init__(self,est='arithmetic',mode='mom',center='mean'):
        self.mode = mode
        self.est=est
        self.center=center
        self.liest = ['arithmetic','distance','sign','entropy']
        self.limo = ['mom','var','std','skew','kurt','com','cov','cok','cos','corr','continuum','M3', 'mdd','ballcov']
        self.licenter = ['mean','median']
        if not(self.mode in self.limo):
            raise(MyException("Only models allowed are: 'mom','var','skew','kurt','com','cov','cos','cok','corr','continuum','M3','mdd','mdc','ballcov'"))
        if not(self.est in self.liest):
            raise(MyException('Only estimator classes allowed are: "arithmetic", "distance", "sign", "entropy"'))
        if not(self.center in self.licenter):
            raise(MyException('Only centring classes allowed are: "mean", "median"'))
        if (self.est=='arithmetic' and self.mode in ['mdd','mdc']):
            raise(MyException('MDD only defined when est="distance"'))
        if (self.mode=='ballcov'): 
            print("To use the Ballcov functionality, uncomment the import statement and comment this line")
        
        
        
    def fit(self,x,**kwargs):
        """
        Fit a dicomo model

        Parameters
        ------------
            X : numpy array or pandas DataFrame
                input data


        Remarks:
        The `fit` function takes several optional input arguments. These are options that 
        apply to individual settings: 
            `biascorr`, Bool, when `True`, correct for bias. For classical product-moment statistics, this 
                is the small sample correction. For energy statistics, this leads to the estimates 
                that are unbiased in high dimension
                (but not preferred in low dimension). 
            `alpha`, float, parameter for continuum association. Has no effect for other options.  
            `option`, int, determines which higher order co-moment to calculate, 
                e.g. for co-skewness, `option=1` calculates CoS(x,x,y)
            `order`, int, which order (co-)moment to calculate. Can be overruled by `mode`, 
                e.g. if `mode='var'`, `order` is set to 2. 
            `calcmode`, str, to use the efficient or naive algorithm to calculate distance statistics. Defaults to `fast` when available.
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
            x = np.array(x).reshape((n,1))
        
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
            
            # Variance
            if mode!='com':
#                if self.center=='mean':
#                    xvar = trimvar(x,trimming)*ntrim/(ntrim-1)
#                elif self.center=='median':
#                    xvar = srs.mad(x)**2
                xmom = trim_mom(x,x,locest,order,trimming,option,biascorr) 
                self.x_moment_ = xmom
                
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
                
                if mode=='mom':
                    self.moment_ = xmom
            
            # Covariance or continuum
            if mode!='mom':
                
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
                    y = np.array(y).reshape((n,1))
                
                como = trim_mom(x,y,locest,order,trimming,option,biascorr)
                
                self.co_moment_ = como
                
                if (biascorr and (order > 2)):
                    como *= ntrim
                
                if mode=='com':
                    self.moment_ = como
                
                    
                if self.mode in ('cok','cos'):
                    x2sd = np.sqrt(trim_mom(x,x,locest,2,trimming,option,biascorr))
                    y2sd = np.sqrt(trim_mom(y,y,locest,2,trimming,option,biascorr))
                        
                    if ((self.mode == 'cok') and biascorr): # biascorr is only exact if standardized
                        como -= como/(ntrim**2)
                        como -= 3*(ntrim-1)**2.0 / ((ntrim-2)*(ntrim-3))

                
                if mode in ['corr','continuum']:
                    
#                    if self.center=='mean':
#                        yvar = trimvar(y,trimming)*ntrim/(ntrim-1)
#                    
#                    elif self.center=='median':
#                        yvar = srs.mad(y)
                    ymom = trim_mom(y,y,locest,order,trimming,option,biascorr)
                    
                    self.y_moment_ = ymom
                    
        
        # Distance based metrics
        elif self.est=='distance':
            
            if 'dmetric' not in kwargs:
                dmetric = 'euclidean'
            else:
                dmetric = kwargs.get('dmetric')
                
            if (mode in ['ballcov','mdd','mdc']):
        
                if 'y' not in kwargs:
                        raise(MyException('Please supply second data vector'))
                else:
                    y = kwargs.get('y')
                    n1 = len(y)
                    if n1==0:
                        raise(MyException('Please feed data with length > 0'))
                    if n1!=n:
                        raise(MyException('Please feed x and y data of equal length')) 
                    if (mode in ['mdd','mdc']):
                        como=np.sqrt(difference_divergence(x,y,center=self.center,trimming=trimming,biascorr=biascorr))
                        self.co_moment_ = como
                        if mode=='mdd':
                            self.moment_ = como 
                        else:
                            xmom=difference_divergence(x,x,center=self.center,trimming=trimming,biascorr=biascorr)
                            ymom=difference_divergence(y,y,center=self.center,trimming=trimming,biascorr=biascorr)
                            self.x_moment_ = xmom
                            self.y_moment_ = ymom
                            mode = 'corr'

                    else:
                        dmy, n2 = distance_matrix_centered(y,biascorr=biascorr,
                                                   trimming=trimming,
                                                   center=self.center,
                                                   dmetric=dmetric) 
                        bcov_res=Ball.bcov_test(x,y,num_permutations=0)[0]
                        self.moment_ = bcov_res
                        
                
                
            
            elif (calcmode=='fast' and self.center =='mean' and trimming == 0 and order==2):
                if mode != 'com':
                    if biascorr:
                        xmom = np.sqrt(dc.u_distance_covariance_sqr(x,x))
                    else:
                        xmom = np.sqrt(dc.distance_covariance_sqr(x,x))
                    self.moment_ = xmom
                if mode !='mom':
                    if 'y' not in kwargs:
                        raise(MyException('Please supply second data vector'))
                    else:
                        y = kwargs.get('y')
                        n1 = len(y)
                    
                    if biascorr:
                        como = np.sqrt(dc.u_distance_covariance_sqr(x,y))
                    else:
                        como = np.sqrt(dc.distance_covariance_sqr(x,y))
                    if mode=='com':
                        self.co_moment_ = como
                        self.moment_ = como
                    elif mode in ['corr','continuum','cos','cok']:
                        if biascorr:
                            ymom = np.sqrt(dc.u_distance_covariance_sqr(y,y))
                        else:
                            ymom = np.sqrt(dc.distance_covariance_sqr(y,y))
                        self.y_moment_ = ymom
                        
            
                
            else:
                dmx, n1 = distance_matrix_centered(x,biascorr=biascorr,
                                                   trimming=trimming,
                                                   center=self.center,
                                                   dmetric=dmetric)
                
                # Variance
                if mode!='com':
                    
                    xmom = distance_moment(dmx,dmx,n1=n1,biascorr=biascorr,center=self.center,trimming=trimming,order=order,option=option)
                        
                    self.x_moment_ = xmom
                    
                    if self.mode in ('std','skew','kurt'):
                        x2mom =distance_moment(dmx,dmx,n1=n1,biascorr=biascorr,center=self.center,trimming=trimming,order=2,option=option)
                        xmom /= x2mom**num_power
                        
                    if mode=='mom':
                        self.moment_ = xmom
                        
                if mode!='mom':
                    
                    if 'y' not in kwargs:
                        raise(MyException('Please supply second data vector'))
                    else:
                        y = kwargs.get('y')
                        n1 = len(y)
                    if n1==0:
                        raise(MyException('Please feed data with length > 0'))
                    if n1!=n:
                        raise(MyException('Please feed x and y data of equal length'))
                    
                    dmy, n2 = distance_matrix_centered(y,biascorr=biascorr,
                                                   trimming=trimming,
                                                   center=self.center,
                                                   dmetric=dmetric)
                    
                    como = distance_moment(dmx,dmy,n1=n1,biascorr=biascorr,center=self.center,trimming=trimming,order=order,option=option)
                    
                    self.co_moment_ = como
                    
                    if mode=='com':
                        self.moment_ = como
                        
                    if self.mode in ('cok','cos'):
                        x2sd = distance_moment(dmx,dmx,n1=n1,biascorr=biascorr,center=self.center,trimming=trimming,order=2,option=option)
                        x2sd = np.sqrt(x2sd)
                        y2sd = distance_moment(dmy,dmy,n1=n1,biascorr=biascorr,center=self.center,trimming=trimming,order=2,option=option)
                        y2sd = np.sqrt(y2sd)
                    
                    if mode in ['corr','continuum','cok','cos']:
                        
                        ymom = distance_moment(dmy,dmy,n1=n,biascorr=biascorr,center=self.center,trimming=trimming,order=order,option=option)
                        self.y_moment_ = ymom
                    
        if mode == 'corr': 
            como /= (np.sqrt(xmom)*np.sqrt(ymom))
            self.moment_=como
        elif mode == 'continuum':
            como *= como * (np.sqrt(xmom)**(alpha -1))
            self.moment_=como
        
        if (self.mode in ('cok','cos') and standardized):
            iter_stop_2 = option 
            iter_stop_1 = order - option
            como /= np.power(x2sd,iter_stop_1)
            como /= np.power(y2sd,iter_stop_2)
            if ((self.mode == 'cok') and not Fisher): # Not very meaningful for co-moment
                como += 3
            self.moment_=como
        
                        
        
        if type(self.moment_)==np.ndarray:
            self.moment_ = self.moment_[0]
        return(self.moment_)
        
                    
                    
                
