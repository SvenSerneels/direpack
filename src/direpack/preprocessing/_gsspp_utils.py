#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar 25 09:02:05 2020

@author: Sven Serneels, Ponalytics. 

Code for radial transform functions largely adapted from 
R code by Jakob Raymaekers

"""

import numpy as np

def quad(dd, p, n): 
    """
    Computes the quadratic radial function
    args:
        dd: vector of distances
        p: number of variables in original data
        n: number of rows in original data
    returns:
        xi: radial function
    """
    d_hmed = np.sort(dd,axis=0)[int(np.floor((n + p + 1) / 2))-1][0]
    idx = np.where(dd > d_hmed)[0]
    xi = np.ones((n,1))
    xi[idx] = (1 / np.square(dd[idx])) * (d_hmed**2)
    return(xi) 

def ss(dd, p,*args,prec=1e-10):

    """
    Computes the spatial sign radial function
    args:
      dd: vector of distances
      p: dimension of original data
      *args flag to be able to pass on n - has no effect
    returns:
      xi: radial function
    """
    dd = np.maximum(dd,prec)
    xi = 1 / dd
    return(xi)

def winsor(dd, p, n) :
    """
    Computes the Winsor radial function
    args:
      dd: vector of distances
      p: number of variables in original data
      n: number of rows in original data
    returns:
      xi: radial function
    """ 
    d_hmed  = np.sort(dd,axis=0)[int(np.floor((n + p + 1) / 2))-1][0]
    idx = np.where(dd > d_hmed)[0]
    xi = np.ones((n,1))
    xi[idx] = (1 / dd[idx]) * d_hmed
    return(xi)   
    
def ball(dd, p, n): 
    
    """
    Computes the Ball radial function
    args:
      dd: vector of distances
      p: number of variables in original data
      n: number of rows in original data
    returns:
      xi: radial function
    """
    
    dWH = np.power(dd,2/3) 
    dWH_hmed = np.sort(dWH,axis=0)[int(np.floor((n + p + 1) / 2))-1][0]
    d_hmed = np.power(dWH_hmed,3/2)
    idx = np.where(dd > d_hmed)[0]
    xi = np.ones((n,1))
    xi[idx] =  0
    return(xi)


def shell(dd, p, n) :
    """
    Computes the Shell radial function
    args:
      dd: vector of distances
      p: number of variables in original data
      n: number of rows in original data
    returns:
      xi: radial function
    """
    
    dWH = np.power(dd,2/3) 
    dWH_hmed = np.sort(dWH,axis=0)[int(np.floor((n + p + 1) / 2))-1][0]
    dWH_hmad = np.sort(np.abs(dWH - dWH_hmed),axis=0)[int(np.floor((n + p + 1) / 2))-1][0]
    cutoff1 = np.power(np.maximum(0, dWH_hmed - dWH_hmad),3/2)
    cutoff2 = np.power(dWH_hmed + dWH_hmad,3/2)
    idxlow = np.where(dd < cutoff1)[0] 
    idxhigh = np.where(dd > cutoff2)[0] 
    xi = np.ones((n,1))
    xi[idxlow] = 0
    xi[idxhigh] = 0
    return(xi)


def linear_redescending(dd, p,n): 
    """
    # Computes the Linear redescending radial function
    args:
      dd: vector of distances
      p: number of variables in original data
      n: number of rows in original data
    returns:
      xi: radial function
    """
    
    dWH = np.power(dd,2/3) 
    dWH_hmed = np.sort(dWH,axis=0)[int(np.floor((n + p + 1) / 2))-1][0]
    dWH_hmad = np.sort(np.abs(dWH - dWH_hmed),axis=0)[int(np.floor((n + p + 1) / 2))-1][0]
    d_hmed = dWH_hmed**(3/2)
    cutoff = (dWH_hmed + 1.4826 * dWH_hmad)**(3/2)
    idxmid = np.where(np.logical_and(dd > d_hmed,dd <= cutoff))[0]
    idxhigh = np.where(dd > cutoff)[0]
    xi = np.ones((n,1))
    xi[idxmid] = 1 - (dd[idxmid,:] - d_hmed) / (cutoff - d_hmed)
    xi[idxhigh] = 0
    return(xi)
    

def _norms(X,**kwargs):
    """
    Casewise norms of a matrix
    """
    return(np.linalg.norm(X,axis=1,keepdims=True,**kwargs))
    
    
def _gsspp(X,p,n,fun=ss):
    """
    Generalized Spatial Sign Pre-Processing for Centred Data
    """
    return(np.multiply(X,fun(_norms(X),p,n)))
        
def _spatial_sign(X,**kwargs):
    """
    Spatial Sign Pre-Processing for Centred Data
    """
    return(X/_norms(X))
    

    
    

    