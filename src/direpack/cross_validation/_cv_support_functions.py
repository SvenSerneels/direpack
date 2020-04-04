#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jan 27 11:42:23 2019

Ancillary tools for plotting sprm results. 
    
    # Deleted: ABLine2D class, was broken in Py 3.7
    cv_score_table (function): tranform sklearn GridSearchCV 
        results into Data Frame

@author: Sven Serneels
"""
import numpy as np
import pandas as ps
from sklearn.metrics import mean_squared_error
from scipy.stats import norm
from ..sprm._m_support_functions import Fair, Huber, Hampel
        
def cv_score_table(res_sprm_cv):
        
    """
    Internal function reorganizing sklearn GridSearchCV results to pandas table. 
    The function adds the cv score table to the object as cv_score_table_
    """
        
    n_settings = len(res_sprm_cv.cv_results_['params'])
    etas = [res_sprm_cv.cv_results_['params'][i]['eta'] for i in range(0,n_settings)]
    components = [res_sprm_cv.cv_results_['params'][i]['n_components'] for i in range(0,n_settings)]
    cv_score_table_ = ps.DataFrame({'etas':etas, 'n_components':components, 'score':res_sprm_cv.cv_results_['mean_test_score']})
    return(cv_score_table_)
    
def robust_loss(y,ypred,lfun=mean_squared_error,fun=Hampel,probct=norm.ppf(0.975),hampelb=norm.ppf(.99),hampelr=norm.ppf(.999)):
    
    """
    Weighted loss function to be used in sklearn cross-validation
    Inputs: 
        y: array or matrix, original predictand
        ypred, array or matrix, predicted values
        lfun, function: an sklearn loss metric that accepts caseweights, 
            e.g. sklearn.metrics.mean_squared_error
        fun: function, weight function, 
            e.g. Fair, Huber or Hampel from sprm.sprm._m_support_functions
        probct, hampelb, hampelr: float, cutoffs for weight functions
    Output:
        loss, float
    """            
    
    if len(ypred.shape) > 1:
        ypred = np.array(ypred).reshape(-1)
    ypred = ypred.astype('float64')
    if len(y.shape) > 1:
        y = np.array(y).reshape(-1)
    y = y.astype('float64')
    r = y - ypred
    w = fun(r,probct,hampelb,hampelr)
    return(lfun(y,ypred,sample_weight=w))
    