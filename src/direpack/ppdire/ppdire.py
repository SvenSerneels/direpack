#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# Created on Sun Dec 30 12:02:12 2018

# ppdire - Projection pursuit dimension reduction

# @author: Sven Serneels (Ponalytics)



#from .dicomo import dicomo
import numpy as np
from statsmodels.regression.quantile_regression import QuantReg
import statsmodels.robust as srs
import scipy.stats as sps
from scipy.linalg import pinv
from scipy.optimize import minimize
import copy
from sklearn.utils.metaestimators import _BaseComposition
from sklearn.base import RegressorMixin,BaseEstimator,TransformerMixin, defaultdict
from sklearn.utils.extmath import svd_flip
from ..sprm.rm import rm 
from ..preprocessing.robcent import VersatileScaler
import warnings
from ..dicomo.dicomo import dicomo 
from ..dicomo._dicomo_utils import * 
from .capi import capi
from ._ppdire_utils import *
from ..preprocessing._preproc_utilities import scale_data
from ..utils.utils import MyException, convert_X_input, convert_y_input
import inspect

class ppdire(_BaseComposition,BaseEstimator,TransformerMixin,RegressorMixin):
    
    """
    PPDIRE Projection Pursuit Dimension Reduction
    
    The class allows for calculation of the projection pursuit optimization
    either through `scipy.optimize` or through the grid algorithm, native to this
    package. The class provides a very flexible way to access optimization of 
    projection indices that can lead to either classical or robust dimension
    reduction. Optimization through scipy.optimize is much more efficient, yet 
    it will only provide correct results for classical projection indices. The
    native grid algorithm should be used when the projection index involves 
    order statistics of any kind, such as ranks, trimming, winsorizing, or 
    empirical quantiles. The grid optimization algorithm for projection pursuit implemented here, 
    was outlined in: 
        
        Filzmoser, P., Serneels, S., Croux, C. and Van Espen, P.J., 
        Robust multivariate methods: The projection pursuit approach,
        in: From Data and Information Analysis to Knowledge Engineering,
        Spiliopoulou, M., Kruse, R., Borgelt, C., Nuernberger, A. and Gaul, W., eds., 
        Springer Verlag, Berlin, Germany,
        2006, pages 270--277.
        
    Parameters
    ------------ 

        projection_index : function or class. 
                            dicomo and capi supplied in this package can both be used, but user defined projection indices can be processed ball covariance can be used. 
            
        pi_arguments : dict 
                        arguments to be passed on to projection index 

        n_components : int 
                        number of components to estimate

        trimming : float
                     trimming percentage to be entered as pct/100 

        alpha : float.
                 Continuum coefficient. Only relevant if ppdire is used to estimate (classical or robust) continuum regression 

        optimizer : str.
                    Presently: either 'grid' (native optimizer) or any of the options in scipy-optimize (e.g. 'SLSQP')

        optimizer_options : dict 
                            with options to pass on to the optimizer 
                            If optimizer == 'grid',
                            ndir: int: Number of directions to calculate per iteration.
                            maxiter: int. Maximal number of iterations.

        optimizer_constraints : dict or list of dicts, 
                                further constraints to be passed on to the optimizer function.

        regopt : str. 
                regression option for regression step y~T. Can be set to 'OLS' (default), 'robust' (will run sprm.rm) or 'quantile' (statsmodels.regression.quantreg). 

        center : str, 
                how to center the data. options accepted are options from sprm.preprocessing 

        center_data : bool 

        scale_data : bool. 
                    Note: if set to False, convergence to correct optimum  is not a given. Will throw a warning. 

        whiten_data : bool. 
                    Typically used for ICA (kurtosis as PI)

        square_pi : bool. 
                    Whether to square the projection index upon evaluation.

        compression : bool. 
                        Use internal data compresion step for flat data. 

        copy : bool. 
                Whether to make a deep copy of the input data or not. 

        verbose : bool. 
                    Set to True prints the iteration number. 

        return_scaling_object : bool.
                                If True, the rescaling object will be returned. 

    Attributes
    ------------ 
    Attributes always provided 
        -  `x_weights_`: X block PPDIRE weighting vectors (usually denoted W)
        -  `x_loadings_`: X block PPDIRE loading vectors (usually denoted P)
        -  `x_scores_`: X block PPDIRE score vectors (usually denoted T)
        -  `x_ev_`: X block explained variance per component
        -  `x_Rweights_`: X block SIMPLS style weighting vectors (usually denoted R)
        -  `x_loc_`: X block location estimate 
        -  `x_sca_`: X block scale estimate
        -  `crit_values_`: vector of evaluated values for the optimization objective. 
        -  `Maxobjf_`: vector containing the optimized objective per component. 

    Attributes created when more than one block of data is provided: 
        -  `C_`: vector of inner relationship between response and latent variables block
        -  `coef_`: vector of regression coefficients, if second data block provided 
        -  `intercept_`: intercept
        -  `coef_scaled_`: vector of scaled regression coefficients (when scaling option used)
        -  `intercept_scaled_`: scaled intercept
        -  `residuals_`: vector of regression residuals
        -  `y_ev_`: y block explained variance 
        -  `fitted_`: fitted response
        -  `y_loc_`: y location estimate
        -  `y_sca_`: y scale estimate

    Attributes created only when corresponding input flags are `True`:
        -   `whitening_`: whitened data matrix (usually denoted K)
        -   `mixing_`: mixing matrix estimate
        -   `scaling_object_`: scaling object from `VersatileScaler`

    
    """

    def __init__(self,
                 projection_index, 
                 pi_arguments = {}, 
                 n_components = 1, 
                 trimming = 0,
                 alpha = 1,
                 optimizer = 'SLSQP',
                 optimizer_options = {'maxiter': 100000}, 
                 optimizer_constraints = {},
                 regopt = 'OLS',
                 center = 'mean',
                 center_data=True,
                 scale_data=True,
                 whiten_data=False,
                 square_pi = False,
                 compression = False,
                 copy=True,
                 verbose=True, 
                 return_scaling_object=True):
        # Called arguments
        self.projection_index = projection_index
        self.pi_arguments = pi_arguments
        self.n_components = n_components
        self.trimming = trimming
        self.alpha = alpha
        self.optimizer = optimizer
        self.optimizer_options = optimizer_options
        self.optimizer_constraints = optimizer_constraints
        self.regopt = regopt
        self.center = center
        self.center_data = center_data
        self.scale_data = scale_data
        self.whiten_data = whiten_data
        self.square_pi = square_pi
        self.compression = compression
        self.copy = copy
        self.verbose = verbose
        self.return_scaling_object = return_scaling_object
        
        # Other global parameters 
        self.constraint = 'norm'
        self.optrange = (-1,1)
        self.licenter = ['mean','median']
        if not(self.center in self.licenter):
            raise(ValueError('Only location estimator classes allowed are: "mean", "median"'))
    

    def fit(self,X,*args,**kwargs):
        
        """
            Fit a projection pursuit dimension reduction model. 

            Parameters
            ------------ 
                
                X : numpy array 
                    Input data.
        
        """

        # Collect optional fit arguments
        biascorr = kwargs.pop('biascorr',False)
            
        if 'h' not in kwargs:
            h = self.n_components
        else:
            h = kwargs.pop('h')
            self.n_components = h
            
        if 'dmetric' not in kwargs:
            dmetric = 'euclidean'
        else:
            dmetric = kwargs.get('dmetric')
            
        if 'mixing' not in kwargs:
            mixing = False
        else:
            mixing = kwargs.get('mixing')
            
        if 'y' not in kwargs:
            na = len(args)
            if na > 0: #Use of *args makes it sklearn consistent
                flag = 'two-block'
                y = args[0]
            else:
                flag = 'one-block'
                y = 0 # to allow calls with 'y=y' in spit of no real y argument present
        else:
            flag = 'two-block'
            y = kwargs.get('y')
                            
            if 'quantile' not in kwargs:
                quantile = .5
            else:
                quantile = kwargs.get('quantile')
                
            if self.regopt == 'robust':
            
                if 'fun' not in kwargs:
                    fun = 'Hampel'
                else:
                    fun = kwargs.get('fun')
                
                if 'probp1' not in kwargs:
                    probp1 = 0.95
                else:
                    probp1 = kwargs.get('probp1')
                
                if 'probp2' not in kwargs:
                    probp2 = 0.975
                else:
                    probp2 = kwargs.get('probp2')
                
                if 'probp3' not in kwargs:
                    probp3 = 0.99
                else:
                    probp3 = kwargs.get('probp3')

            
        if self.projection_index == dicomo:
            
            if self.pi_arguments['mode'] in ('M3','cos','cok'):
            
                if 'option' not in kwargs:
                    option = 1
                else:
                    option = kwargs.get('option')
                
                if option > 3:
                    print('Option value >3 will compute results, but meaning may be questionable')
                
        # Initiate projection index    
        self.most = self.projection_index(**self.pi_arguments)         
        
        # Initiate some parameters and data frames
        if self.copy:
            X0 = copy.deepcopy(X)
            self.X0 = X0
        else:
            X0 = X        
        X = convert_X_input(X0)    
        n,p = X0.shape 
        trimming = self.trimming
        
        # Check dimensions 
        if h > min(n,p):
            raise(MyException('number of components cannot exceed number of samples'))
            
        if (self.projection_index == dicomo and self.pi_arguments['mode'] == 'kurt' and self.whiten_data==False):
            warnings.warn('Whitening step is recommended for ICA')
            
        # Pre-processing adjustment if whitening
        if self.whiten_data:
            self.center_data = True
            self.scale_data = False
            self.compression = False
            print('All results produced are for whitened data')
        
        # Centring and scaling
        if self.scale_data:
            if self.center=='mean':
                scale = 'std'
            elif ((self.center=='median')|(self.center=='l1median')):
                scale = 'mad' 
        else:
            scale = 'None'
            warnings.warn('Without scaling, convergence to optima is not given')
            
         # Data Compression for flat tables if required                
        if ((p>n) and self.compression):
            V,S,U = np.linalg.svd(X.T,full_matrices=False)
            X = np.matmul(U.T,np.diag(S))
            n,p = X.shape
            
            if (srs.mad(X)==0).any(): 
                warnings.warn('Due to low scales in data, compression would induce zero scales.' 
                              + '\n' + 'Proceeding without compression.')
                dimensions = False
                if copy:
                    X = copy.deepcopy(X0)
                else:
                    X = X0
            else:
                dimensions = True
        else:
            dimensions = False
        
        # Initiate centring object and scale X data 
        centring = VersatileScaler(center=self.center,scale=scale,trimming=trimming)      
  
        if self.center_data:
            Xs = centring.fit_transform(X)
            mX = centring.col_loc_
            sX = centring.col_sca_
        else:
            Xs = X
            mX = np.zeros((1,p))
            sX = np.ones((1,p))

        fit_arguments = {}
            
        # Data whitening (best practice for ICA)
        if self.whiten_data:
            V,S,U = np.linalg.svd(Xs.T,full_matrices=False)
            del U
            K = (V/S)[:,:p]
            del V,S
            Xs = np.matmul(Xs, K)
            Xs *= np.sqrt(p)

        # Pre-process y data when available 
        if flag != 'one-block':
            
            ny = y.shape[0]
            y = convert_y_input(y)
            if len(y.shape) < 2:
                y = np.array(y).reshape((ny,1))
#            py = y.shape[1]
            if ny != n:
                raise(MyException('X and y number of rows must agree'))
            if self.copy:
                y0 = copy.deepcopy(y)
                self.y0 = y0
                
            if self.center_data:
                ys = centring.fit_transform(y)
                my = centring.col_loc_
                sy = centring.col_sca_ 
            else:
                ys = y
                my = 0
                sy = 1
            ys = np.array(ys).astype('float64')
        
        else:
            ys = None
                

        # Initializing output matrices
        W = np.zeros((p,h))
        T = np.zeros((n,h))
        P = np.zeros((p,h))
        B = np.zeros((p,h))
        R = np.zeros((p,h))
        B_scaled = np.zeros((p,h))
        C = np.zeros((h,1))
        Xev = np.zeros((h,1))
        assovec = np.zeros((h,1))
        Maxobjf = np.zeros((h,1))

        # Initialize deflation matrices 
        E = copy.deepcopy(Xs)
        f = ys

        bi = np.zeros((p,1))
        
        opt_args = { 
                    'alpha': self.alpha,
                    'trimming': self.trimming,
                    'biascorr': biascorr, 
                    'dmetric' : 'euclidean',
                    }
        
        if self.optimizer=='grid':
            # Define grid optimization ranges
            if 'ndir' not in self.optimizer_options:
                self.optimizer_options['ndir'] = 1000
            optrange = np.sign(self.optrange)
            optmax = self.optrange[1]
            stop0s = np.arcsin(optrange[0])
            stop1s = np.arcsin(optrange[1])
            stop1c = np.arccos(optrange[0])
            stop0c = np.arccos(optrange[1])
            anglestart = max(stop0c,stop0s)
            anglestop = max(stop1c,stop1s)
            nangle = np.linspace(anglestart,anglestop,self.optimizer_options['ndir'],endpoint=False)            
            alphamat = np.array([np.cos(nangle), np.sin(nangle)])
            opt_args['_stop0c'] = stop0c
            opt_args['_stop0s'] = stop0s
            opt_args['_stop1c'] = stop1c
            opt_args['_stop1s'] = stop1s
            opt_args['optmax'] = optmax
            opt_args['optrange'] = self.optrange
            opt_args['square_pi'] = self.square_pi
            if optmax != 1:
                alphamat *= optmax
        
            if p>2:
                anglestart = min(opt_args['_stop0c'],opt_args['_stop0s'])
                anglestop = min(opt_args['_stop1c'],opt_args['_stop1s'])
                nangle = np.linspace(anglestart,anglestop,self.optimizer_options['ndir'],endpoint=True)
                alphamat2 = np.array([np.cos(nangle), np.sin(nangle)])
                if optmax != 1:
                    alphamat2 *= opt_args['optmax']
                
            # Arguments for grid plane
            opt_args['alphamat'] = alphamat,
            opt_args['ndir'] = self.optimizer_options['ndir'],
            opt_args['maxiter'] = self.optimizer_options['maxiter']
            if type(opt_args['ndir'] is tuple): 
                opt_args['ndir'] = opt_args['ndir'][0]
            
            # Arguments for grid plane #2
            grid_args_2 = { 
                     'alpha': self.alpha,
                     'alphamat': alphamat2,
                     'ndir': self.optimizer_options['ndir'],
                     'trimming': self.trimming,
                     'biascorr': biascorr, 
                     'dmetric' : 'euclidean',
                     '_stop0c' : stop0c,
                     '_stop0s' : stop0s,
                     '_stop1c' : stop1c,
                     '_stop1s' : stop1s,
                     'optmax' : optmax,
                     'optrange' : self.optrange,
                     'square_pi' : self.square_pi
                     }
            if flag=='two-block':
                grid_args_2['y'] = f
        
        if flag=='two-block':
            opt_args['y'] = f
            

        # Itertive coefficient estimation
        for i in range(0,h):

            if self.optimizer=='grid':
                if p==2:
                    wi,maximo = gridplane(E,self.most,
                                          pi_arguments=opt_args
                                          )
           
                elif p>2:
                
                    afin = np.zeros((p,1)) # final parameters for linear combinations
                    Z = copy.deepcopy(E)
                    # sort variables according to criterion
                    meas = [self.most.fit(E[:,k].reshape((-1,1)),
                            **opt_args) 
                            for k in np.arange(0,p)]
                    if self.square_pi:
                        meas = np.square(meas)
                    wi,maximo = gridplane(Z[:,0:2],self.most,opt_args)
                    Zopt = np.dot(Z[:,0:2],wi) 
                    afin[0:2]=wi
                    for j in np.arange(2,p):
                        projmat = np.array([np.array(Zopt[:,0]).reshape(-1),
                                         np.array(Z[:,j]).reshape(-1)]).T
                        wi,maximo = gridplane(projmat,self.most,
                                              opt_args
                                              )
                        
                        Zopt = Zopt*float(wi[0]) + Z[:,j].reshape(-1,1)*float(wi[1])
                        afin[0:(j+1)] = afin[0:(j+1)]*float(wi[0])
                        afin[j] = float(wi[1])

                    tj = np.dot(Z,afin)
                    objf = self.most.fit(tj,
                                     **{**fit_arguments,**opt_args}
                                    )
                    if self.square_pi:
                        objf *= objf
    

                    # outer loop to run until convergence
                    objfold = copy.deepcopy(objf)
                    objf = -1000
                    afinbest = afin
                    ii = 0
                    maxiter_2j = 2**round(np.log2(self.optimizer_options['maxiter'])) 
                
                    while ((ii < self.optimizer_options['maxiter'] + 1) and (abs(objfold - objf)/abs(objf) > 1e-4)):
                        for j in np.arange(0,p):
                            projmat = np.array([np.array(Zopt[:,0]).reshape(-1),
                                         np.array(Z[:,j]).reshape(-1)]).T
                            if j > 16:
                                divv = maxiter_2j
                            else:
                                divv = min(2**j,maxiter_2j)
                        
                            wi,maximo = gridplane_2(projmat,
                                                    self.most,
                                                    q=afin[j],
                                                    div=divv,
                                                    pi_arguments=grid_args_2
                                                    )
                            Zopt = Zopt*float(wi[0,0]) + Z[:,j].reshape(-1,1)*float(wi[1,0])
                            afin *= float(wi[0,0])
                            afin[j] += float(wi[1,0])
                        
                        # % evaluate the objective function:
                        tj = np.dot(Z,afin)
                    
                        objfold = copy.deepcopy(objf)
                        objf = self.most.fit(tj,
                                         q=afin,
                                         **opt_args
                                         )
                        if self.square_pi:
                            objf *= objf
                    
                        if  objf!=objfold:
                            if self.constraint == 'norm':
                                afinbest = afin/np.sqrt(np.sum(np.square(afin)))
                            else:
                                afinbest = afin
                            
                        ii +=1
                        if self.verbose:
                            print(str(ii))
                    #endwhile
                
                    afinbest = afin
                    wi = np.zeros((p,1))
                    wi = afinbest
                    Maxobjf[i] = objf
                # endif;%if p>2;
            else: # do not optimize by the grid algorithm
                if self.trimming > 0: 
                    warnings.warn('Optimization that involves a trimmed objective is not a quadratic program. The scipy-optimize result will be off!!')
                if 'center' in self.pi_arguments:
                    if (self.pi_arguments['center']=='median'): 
                        warnings.warn('Optimization that involves a median in the objective is not a quadratic program. The scipy-optimize result will be off!!')   
                constraint = {'type':'eq',
                              'fun': lambda x: np.linalg.norm(x) -1,
                              }
                if len(self.optimizer_constraints)>0: 
                    constraint = [constraint,self.optimizer_constraints]
                wi = minimize(pp_objective,
                              E[0,:].transpose(),
                              args=(self.most,E,opt_args),
                              method=self.optimizer,
                              constraints=constraint,
                              options=self.optimizer_options).x
                wi = np.array(wi).reshape((p,1))
                wi /= np.sqrt(np.sum(np.square(wi)))
                
                
            # Computing projection weights and scores
            ti = np.dot(E,wi)
            if self.optimizer != 'grid':
                Maxobjf[i] = self.most.fit(np.dot(E,wi),**opt_args)
            nti = np.linalg.norm(ti)
            pi = np.dot(E.T,ti) / (nti**2)
            if self.whiten_data:
                wi /= np.sqrt((wi**2).sum())
                wi = K*wi
            wi0 = wi
            wi = np.array(wi)
            if len(W[:,i].shape) == 1:
                wi = wi.reshape(-1)
            W[:,i] = wi
            T[:,i] = np.array(ti).reshape(-1)
            P[:,i] = np.array(pi).reshape(-1)
            
            if flag != 'one-block':
                criteval = self.most.fit(np.dot(E,wi0),
                                         **opt_args
                                         )
                if self.square_pi:
                    criteval *= criteval
                    
                assovec[i] = criteval
                

            # Deflation of the datamatrix guaranteeing orthogonality restrictions
            E -= ti*pi.T
 
            # Calculate R-Weights
            R = np.dot(W[:,0:(i+1)],pinv(np.dot(P[:,0:(i+1)].T,W[:,0:(i+1)]),check_finite=False))
        
            # Execute regression y~T if y is present. Generate regression estimates.
            if flag != 'one-block':
                if self.regopt=='OLS':
                    ci = np.dot(ti.T,ys)/(nti**2)
                elif self.regopt == 'robust':
                    linfit = rm(fun=fun,probp1=probp1,probp2=probp2,probp3=probp3,
                                centre=self.center,scale=scale,
                                start_cutoff_mode='specific',verbose=self.verbose)
                    linfit.fit(ti,ys)
                    ci = linfit.coef_
                elif self.regopt == 'quantile':
                    linfit = QuantReg(y,ti)
                    model = linfit.fit(q=quantile)
                    ci = model.params
                # end regression if
                
                C[i] = ci
                bi = np.dot(R,C[0:(i+1)])
                bi_scaled = bi
                bi = np.multiply(np.reshape(sy/sX,(p,1)),bi)
                B[:,i] = bi[:,0]
                B_scaled[:,i] = bi_scaled[:,0]

        # endfor; Loop for latent dimensions

        # Re-adjust estimates to original dimensions if data have been compressed 
        if dimensions:
            B = np.matmul(V[:,0:p],B)
            B_scaled = np.matmul(V[:,0:p],B_scaled)
            R = np.matmul(V[:,0:p],R)
            W = np.matmul(V[:,0:p],W)
            P = np.matmul(V[:,0:p],P)
            bi = B[:,h-1]
            if self.center_data:
                Xs = centring.fit_transform(X0)
                mX = centring.col_loc_
                sX = centring.col_sca_
            else:
                Xs = X0
                mX = np.zeros((1,p))
                sX = np.ones((1,p))
        
        bi = bi.astype("float64")
        if flag != 'one-block':            
            # Calculate scaled and unscaled intercepts
            if dimensions:
                X = convert_X_input(X0)
            if(self.center == "mean"):
                intercept = sps.trim_mean(y - np.matmul(X,bi),trimming)
            else:
                intercept = np.median(np.reshape(y - np.matmul(X,bi),(-1)))
            yfit = np.matmul(X,bi) + intercept
            if not(scale == 'None'):
                if (self.center == "mean"):
                    b0 = np.mean(ys - np.matmul(Xs.astype("float64"),bi))
                else:
                    b0 = np.median(np.array(ys.astype("float64") - np.matmul(Xs.astype("float64"),bi)))
            else:
                b0 = intercept
            
            # Calculate fit values and residuals 
            yfit = yfit    
            r = y - yfit
            setattr(self,"coef_",B)
            setattr(self,"intercept_",intercept)
            setattr(self,"coef_scaled_",B_scaled)
            setattr(self,"intercept_scaled_",b0)
            setattr(self,"residuals_",r)
            setattr(self,"fitted_",yfit)
            setattr(self,"y_loadings_",C)
            setattr(self,"y_loc_",my)
            setattr(self,"y_sca_",sy)
                
        setattr(self,"x_weights_",W)
        setattr(self,"x_loadings_",P)
        setattr(self,"x_rotations_",R)
        setattr(self,"x_scores_",T)
        setattr(self,"x_ev_",Xev)
        setattr(self,"crit_values_",assovec)
        setattr(self,"Maxobjf_",Maxobjf)
        
        if self.whiten_data:
            setattr(self,"whitening_",K)

        
        if mixing:
            setattr(self,"mixing_",np.linalg.pinv(W))
        
        
        setattr(self,"x_loc_",mX)
        setattr(self,"x_sca_",sX)

        setattr(self,'scaling',scale)
        if self.return_scaling_object:
            setattr(self,'scaling_object_',centring)
        
        return(self)   


    def predict(self,Xn):
        """
        predicts the response  on new data Xn

        Parameters
        ----------
                Xn : matrix or data frame
                     Input data to be transformed 
                
        Returns
        -------
        predictions : numpy array 
                      The predictions from the dimension reduction model
        """
        Xn = convert_X_input(Xn)
        (n,p) = Xn.shape
        (q,h) = self.coef_.shape
        if p!=q:
            raise(ValueError('New data must have seame number of columns as the ones the model has been trained with'))
        return(np.array(np.matmul(Xn,self.coef_[:,h-1]) + self.intercept_).T.reshape(-1))
        
    def transform(self,Xn):
        """
        Computes the dimension reduction of the data Xn based on the fitted sudire model.

        Parameters
        ----------
                Xn : matrix or data frame
                     Input data to be transformed 

        Returns
        -------
        transformed_data : numpy array
                             the dimension reduced data 
        """
        Xn = convert_X_input(Xn)
        (n,p) = Xn.shape
        if p!= self.x_loadings_.shape[0]:
            raise(ValueError('New data must have seame number of columns as the ones the model has been trained with'))
        Xnc = scale_data(Xn,self.x_loc_,self.x_sca_)
        return(Xnc*self.x_rotations_)
        
    @classmethod   
    def _get_param_names(cls):
        """Get parameter names for the estimator"""
        # fetch the constructor or the original constructor before
        # deprecation wrapping if any
        init = getattr(cls.__init__, 'deprecated_original', cls.__init__)
        if init is object.__init__:
            # No explicit constructor to introspect
            return []
    
        # introspect the constructor arguments to find the model parameters
        # to represent
        init_signature = inspect.signature(init)
        # Consider the constructor parameters excluding 'self'
        parameters = [p for p in init_signature.parameters.values()
                      if p.name != 'self' and p.kind != p.VAR_KEYWORD]
        for p in parameters:
            if p.kind == p.VAR_POSITIONAL:
                raise RuntimeError("scikit-learn estimators should always "
                                   "specify their parameters in the signature"
                                   " of their __init__ (no varargs)."
                                   " %s with constructor %s doesn't "
                                   " follow this convention."
                                   % (cls, init_signature))
        # Extract and sort argument names excluding 'self'
        return sorted([p.name for p in parameters])
    
    def get_params(self, deep=False):
        """Get parameters for this estimator.
        Parameters
        ----------
        deep : boolean, optional
            If True, will return the parameters for this estimator and
            contained subobjects that are estimators.
        Returns
        -------
        params : mapping of string to any
            Parameter names mapped to their values.
        ------
        Copied from ScikitLlearn instead of imported to avoid 'deep=True'
        """
        out = dict()
        for key in self._get_param_names():
            value = getattr(self, key, None)
            if deep and hasattr(value, 'get_params'):
                deep_items = value.get_params().items()
                out.update((key + '__' + k, val) for k, val in deep_items)
            out[key] = value
        return out
        
    def set_params(self, **params):
        """Set the parameters of this estimator.
        Copied from ScikitLearn, adapted to avoid calling 'deep=True'
        Returns
        -------
        self
        ------
        Copied from ScikitLlearn instead of imported to avoid 'deep=True'
        """
        if not params:
            # Simple optimization to gain speed (inspect is slow)
            return self
        valid_params = self.get_params()
    
        nested_params = defaultdict(dict)  # grouped by prefix
        for key, value in params.items():
            key, delim, sub_key = key.partition('__')
            if key not in valid_params:
                raise ValueError('Invalid parameter %s for estimator %s. '
                                 'Check the list of available parameters '
                                 'with `estimator.get_params().keys()`.' %
                                 (key, self))
    
            if delim:
                nested_params[key][sub_key] = value
            else:
                setattr(self, key, value)
                valid_params[key] = value
    
        for key, sub_params in nested_params.items():
            valid_params[key].set_params(**sub_params)
    
        return self
        
