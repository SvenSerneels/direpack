#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Dec 16 13:44:19 2018

@author: Sven Serneels, Ponalytics. 
"""

import scipy.stats as sps
import scipy.spatial as spp
import numpy as np
import copy

# Use direct mean or trimmed mean where appropriate
# Using trimmed mean, even with 0% trimming, will cause errors in QP optimization
def trim_mean(x,trimming):
    
    if trimming == 0: 
        return(np.mean(x))
    else:
        return(sps.trim_mean(x,trimming)[0])

# Calculate trimmed variance
def trimvar(x,trimming):
        # division by n
        return(trim_mean(np.square(x - trim_mean(x,trimming)),trimming))

# Dummy identity function       
def identity(x): 
    return(x)

# Calculate trimmed moments. 
def trim_mom(x,y,locest,order,trimming,option,fscorr=True):
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