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

def trim_mean(x,trimming,axis=0):
    
    if trimming == 0: 
        return(np.mean(x,axis=axis))
    else:
        return(sps.trim_mean(x,trimming,axis=axis))

def trimvar(x,trimming):
        # division by n
        return(sps.trim_mean(np.square(x - sps.trim_mean(x,trimming)),trimming))
        
def identity(x): 
    return(x)
        
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

def double_center_flex(a, center='mean', **kwargs):
    """
    Double centered function adapted to accommodate for location types different
    from mean. 

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

