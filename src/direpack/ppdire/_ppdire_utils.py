#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan 2 2020

@author: Sven Serneels, Ponalytics. 
"""

import numpy as np
import pandas as ps


def pp_objective(x,est,X,opt_args):
    
    """
    Optimization objective for ppdire 
    
    """
    
    n = len(x)
    x = np.array(x).reshape((n,1))
    return(-est.fit(np.matmul(X,x),**opt_args))

def gridplane(X,most,pi_arguments={},**kwargs):

    """
    Function for grid search in a plane in two dimensions
    
    Required: X, np.array(n,2), data, 
              most, class object, projection index. Designed for 
                  dicomo or capi classes.
    Optional: pi_arguments, dict: arguments to pass on to projection index, 
                    plus a few local arguments such as optrange and square_pi
                    (see ppdire for explanation)
        
              y, np.array(n,1), second block of data 
              biascorr, to apply bias correction at normal distribution 
              alphamat, np.array: matrix of alpha angles to be scanned. 
        
        
    Values: 
        wi, np.array(p,1): optimal direction 
        maximo, float: optimal value of projection index
        
    Note: this function is written exclusively to be called from within the ppdire class
    
    """
    
            
    if (('biascorr' not in kwargs) and ('biascorr' not in pi_arguments)):
        biascorr = False
    else:
        biascorr = kwargs.get('biascorr')
    
    if len(pi_arguments) == 0:
        
        pi_arguments = {
                        'alpha': 0,
                        'ndir': 1000,
                        'trimming': 0,
                        'biascorr': biascorr, 
                        'dmetric' : 'euclidean',
                        'alphamat': None,
                        'optrange': (-1,1),
                        'square_pi': False
                        }
        
    if ('y' in kwargs):
        y = kwargs.pop('y')
        pi_arguments['y'] = y
        
    optrange = pi_arguments['optrange']
    optmax = optrange[1]
    
    alphamat = kwargs.pop('alphamat',pi_arguments['alphamat'])
    if (alphamat != None):
        optrange = np.sign(optrange)
        stop0s = np.arcsin(optrange[0])
        stop1s = np.arcsin(optrange[1])
        stop1c = np.arccos(optrange[0])
        stop0c = np.arccos(optrange[1])
        anglestart = max(stop0c,stop0s)
        anglestop = max(stop1c,stop1s)
        nangle = np.linspace(anglestart,anglestop,pi_arguments['ndir'],endpoint=False)            
        alphamat = np.array([np.cos(nangle), np.sin(nangle)])
        if optmax != 1:
            alphamat *= optmax
    
    tj = np.matmul(X,alphamat)
    if pi_arguments['square_pi']:
        meas = [most.fit(tj[:,i],**pi_arguments)**2 
        for i in np.arange(0,pi_arguments['ndir'])]
    else:
        meas = [most.fit(tj[:,i],**pi_arguments) 
        for i in np.arange(0,pi_arguments['ndir'])]
        
    maximo = np.max(meas)
    indmax = np.where(meas == maximo)[0]
    if len(indmax)>0:
        indmax = indmax[0]
    wi = np.array(alphamat[:,indmax]).reshape((2,1))
    
    return(wi,maximo)
    
    

def gridplane_2(X,most,q,div,pi_arguments={},**kwargs):

    """
    Function for refining a grid search in a plane in two dimensions
    
    Required: X, np.array(n,2), data
              most, class object, projection index. Designed for 
                  dicomo or capi classes.
              q, np.array(1,1), last obtained suboptimal direction component
              div, float, number of subsegments to divide angle into
    
    Optional: pi_arguments, dict: arguments to pass on to projection index, 
                    plus a few local arguments such as optrange and square_pi
                    (see ppdire for explanation)
        
              y, np.array(n,1), second block of data 
              biascorr, to apply bias correction at normal distribution 
              alphamat, np.array: matrix of alpha angles to be scanned. 
        
    pi_arguments is a dict of arguments passed on to the projection index
        
    Values: 
        wi, np.array(p,1): optimal direction 
        maximo, float: optimal value of projection index
        
    Note: this function is written to be called from within the ppdire class
    
    """
            
    if (('biascorr' not in kwargs) and ('biascorr' not in pi_arguments)):
        biascorr = False
    else:
        biascorr = kwargs.get('biascorr')
        
    if len(pi_arguments) == 0:
        
        pi_arguments = {
                        'alpha': 0,
                        'ndir': 1000,
                        'trimming': 0,
                        'biascorr': biascorr, 
                        'dmetric' : 'euclidean',
                        'alphamat': None,
                        'optrange': (-1,1),
                        'square_pi': False
                        }

        
    if 'y' in kwargs:
        y = kwargs.pop('y')
        pi_arguments['y'] = y

    optrange = pi_arguments['optrange']
    optmax = optrange[1]
   
    alphamat = kwargs.pop('alphamat',pi_arguments['alphamat'])
    if (alphamat != None).any():
        anglestart = min(pi_arguments['_stop0c'],pi_arguments['_stop0s'])
        anglestop = min(pi_arguments['_stop1c'],pi_arguments['_stop1s'])
        nangle = np.linspace(anglestart,anglestop,pi_arguments['ndir'],endpoint=True)
        alphamat = np.array([np.cos(nangle), np.sin(nangle)])
        if optmax != 1:
            alphamat *= optmax
    alpha1 = alphamat
    divisor = np.sqrt(1 + 2*np.multiply(alphamat[0,:].reshape(1,-1),alphamat[1,:].reshape(1,-1))*q[0])
    alpha1 = np.divide(alphamat,np.repeat(divisor,2,0))
    tj = np.dot(X,alpha1)
    
    if pi_arguments['square_pi']:
        meas = [most.fit(tj[:,i],**pi_arguments)**2 
        for i in np.arange(0,pi_arguments['ndir'])]
    else:
        meas = [most.fit(tj[:,i],**pi_arguments) 
        for i in np.arange(0,pi_arguments['ndir'])]

    maximo = np.max(meas)
    indmax = np.where(meas == maximo)[0]
    if len(indmax)>0:
        indmax = indmax[0]
    wi = np.array(alpha1[:,indmax]).reshape((2,1))
    
    return(wi,maximo)
