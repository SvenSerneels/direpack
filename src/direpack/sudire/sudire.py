# -*- coding: utf-8 -*-

import numpy as np
from scipy.linalg import inv, sqrtm
import copy
from sklearn.utils.metaestimators import _BaseComposition
from sklearn.base import RegressorMixin, BaseEstimator, TransformerMixin, defaultdict
from ..preprocessing.robcent import VersatileScaler
from ._sudire_utils import *
from scipy.linalg import orth
import warnings
import statsmodels.robust.scale as srs
from scipy.stats import trim_mean
import dcor as dc

# import Ball
import statsmodels.api as sm
import inspect
from ..ipopt_temp.ipopt_wrapper import minimize_ipopt
from ..ipopt_temp.jacobian import (
    FunctionWithApproxJacobianCentral,
    FunctionWithApproxJacobian,
)
from ..dicomo._dicomo_utils import *
from ..utils.utils import *


class sudire(_BaseComposition, BaseEstimator, TransformerMixin, RegressorMixin):

    """SUDIRE Sufficient Dimension Reduction 
    
    The class allows for Sufficient Dimension Reduction using a variety of 
    methods. If the method requires optimization of a function, 
    This optimization is done through the Interior Point Optimizer (IPOPT)
    algorithm. 
        
    Parameters
    ----------
        sudiremeth: function or class. sudiremeth in this package can also be used, but user defined functions  can be processed. Built in options are : 

            save : Sliced Average Variance Estimation

            sir  : Slices Inverse Regression

            dr   : Directional Regression 

            iht  : Iterative Hessian Transformations

            dcov-sdr : SDR via Distance Covariance 
            
            mdd-sdr : SDR via Martingale Difference Divergence.

            bcov-sdr : SDR via ball covariance
            
        n_components : int 
                      dimension of the central subspace.
            
        trimming : float
                 trimming percentage to be entered as pct/100 
        
        optimizer_options : dict 
                            with options to pass on to the optimizer.Includes: 

        max_iter : int
                     Maximal number of iterations.

        tol: float
             relative convergence tolerance

        constr_viol_tol : float
                        Desired threshold for the constraint violation.
            
        optimizer_constraints : dict or list of dicts
                                 further constraints to be passed on to the optimizer function.
            
        optimizer_arguments: dict
                             extra arguments to be passed to the sudiremeth function during optimization.
        
        optimizer_start : numpy array
                         starting value for the optimization.
        
        center : str
             how to center the data. options accepted are options from sprm.preprocessing 
            
        center_data : bool 
                    If True, the data will be centered before the dimension reduction   
        
        scale_data : bool
                    if set to False, convergence to correct optimum  is not a given. Will throw a warning. 
            
        compression : bool
                     Use internal data compresion step for flat data. 
        
        n_slices :  int
                    The number of slices for SAVE, SIR, DR
    
        is_distance_mat : bool 
                         if the inputed matrices for x and y are distance matrices.
            
        dmetric :  str
                 distance metric used internally. Defaults to 'euclidean'
        
        fit_ols :  bool 
                 if True, an OLS model is fitted after the dimension reduction.
        
        copy : bool
              Whether to make a deep copy of the input data or not. 
        
        verbose : bool
                 Set to True prints the iteration number. 
        
        return_scaling_object: bool. 
                                If True, the scaling object will be return after the dimension reduction. 

    Attributes
    ----------
    Attributes always provided 
        -  `x_loadings_`: Estimated basis of the central subspace 
        -  `x_scores_`: The projected X data. 
        -  `x_loc_`:  location estimate for X 
        -  `x_sca_`:  scale estimate for X
        - ` ols_obj` : fitted OLS objected
        -  `y_loc_`: y location estimate
        -  `y_sca_`: y scale estimate

    Attributes created only when corresponding input flags are `True`:
        -   `whitening_`: whitened data matrix (usually denoted K)
        -   `scaling_object_`: scaling object from `VersatileScaler`  
        
    
        
    
    """

    def __init__(
        self,
        sudiremeth="dcov-sdr",
        n_components=2,
        trimming=0,
        optimizer_options={"max_iter": 1000},
        optimizer_constraints=None,
        optimizer_arguments=None,
        optimizer_start=None,
        center_data=True,
        center="mean",
        scale_data=True,
        whiten_data=False,
        compression=False,
        n_slices=6,
        dmetric="euclidean",
        fit_ols=True,
        copy=True,
        response_type="continuous",
        verbose=True,
        return_scaling_object=True,
    ):
        # Called arguments
        self.sudiremeth = sudiremeth
        self.n_components = n_components
        self.trimming = trimming
        self.optimizer_options = optimizer_options
        self.optimizer_constraints = optimizer_constraints
        self.optimizer_arguments = optimizer_arguments
        self.optimizer_start = optimizer_start
        self.center = center
        self.center_data = center_data
        self.scale_data = scale_data
        self.whiten_data = whiten_data
        self.compression = compression
        self.n_slices = n_slices
        self.dmetric = dmetric
        self.fit_ols = fit_ols
        self.copy = copy
        self.response_type = response_type
        self.verbose = verbose
        self.return_scaling_object = return_scaling_object

        # Other global parameters
        self.licenter = ["mean", "median"]
        if not (self.center in self.licenter):
            raise (
                ValueError(
                    'Only location estimator classes allowed are: "mean", "median"'
                )
            )
        self.limeths = [
            "sir",
            "save",
            "dr",
            "dcov-sdr",
            "bcov-sdr",
            "mdd-sdr",
            "phd",
            "iht",
        ]
        if not (self.sudiremeth in self.limeths) and not callable(self.sudiremeth):
            raise (
                ValueError(
                    'Only SDR methods allowed are : "sir", "save", "dr", "dcov-sdr","bcov-sdr", "mdd-sdr", "iht","phd"'
                )
            )

    def fit(self, X, y, *args, **kwargs):

        """
        Fit a Sufficient Dimension Reduction Model. 
        
        Parameters
        ----------
                X : matrix or data frame 
                    Input data of predictors  

                y :  vector or 1D matrix
                    Response data
                
                args or kwargs : 
                                Further parameters to user defined sudiremeth can be passed here  
        Returns
        -------
            self
        
        -------
        """

        # Collect optional fit arguments

        if "dmetric" not in kwargs:
            dmetric = "euclidean"
        else:
            dmetric = kwargs.get("dmetric")
        if "biascorr" not in kwargs:
            biascorr = False
        else:
            biascorr = kwargs.get("biascorr")
        if "flag" not in kwargs:
            flag = "two-block"
        else:
            flag = kwargs.get("flag")
        if "is_distance_mat" not in kwargs:
            is_distance_mat = False
        else:
            is_distance_mat = kwargs.pop("is_distance_mat")
        # Initiate some parameters and data frames
        if self.copy:
            X0 = copy.deepcopy(X)
            self.X0 = X0
        else:
            X0 = X
        X = convert_X_input(X0)
        n, p = X0.shape
        trimming = self.trimming

        # Check dimensions
        if self.n_components > min(n, p):
            raise (
                MyException(
                    "number of components cannot exceed number of variables or sample size"
                )
            )
        # Pre-processing adjustment if whitening
        if self.whiten_data:
            self.center_data = True
            self.scale_data = False
            self.compression = False
            print("All results produced are for whitened data")
        # Store original X data mean and varcov matrix.
        varmatx = np.cov(X, rowvar=0)
        meanx = X.mean(axis=0)
        N2 = inv(sqrtm(varmatx))

        # Data Compression for flat tables if required
        if (p > n) and self.compression:
            V, S, U = np.linalg.svd(X.T, full_matrices=False)
            X = np.matmul(U.T, np.diag(S))
            n, p = X.shape

            if (srs.mad(X) == 0).any():
                warnings.warn(
                    "Due to low scales in data, compression would induce zero scales."
                    + "\n"
                    + "Proceeding without compression."
                )
                dimensions = False
                if copy:
                    X = copy.deepcopy(X0)
                else:
                    X = X0
            else:
                dimensions = True
        else:
            dimensions = False

            # Centring and scaling
            # centering :
        if self.center_data:
            if self.center != "mean":
                centring = VersatileScaler(
                    center=self.center, scale="None", trimming=self.trimming
                )
                Xs = centring.fit_transform(X0)
                mX = centring.col_loc_
                sX = centring.col_sca_
            else:
                Xs = X - trim_mean(X, self.trimming, axis=0)
                mX = trim_mean(X, self.trimming, axis=0)
                sX = np.sqrt(np.diag(varmatx))
        else:
            Xs = X0
            mX = np.zeros((1, p))
            sX = np.ones((1, p))
        if self.scale_data:
            if self.center == "mean":
                scale = "std"
                N2 = inv(sqrtm(varmatx))
                Xs = np.matmul(Xs, N2)
            elif (self.center == "median") or (self.center == "l1median"):
                scale = "mad"
                centring = VersatileScaler(
                    center=self.center, scale=scale, trimming=trimming
                )
                Xs = centring.fit_transform(X0)
                mX = centring.col_loc_
                sX = centring.col_sca_
            else:
                raise (
                    MyException(
                        'centering options have to be either "mean", "median", or "l1median"'
                    )
                )
        else:
            scale = "None"
            warnings.warn("Without scaling, convergence to optima is not given")
        # Initiate centring object and scale X data

        if self.whiten_data:
            V, S, U = np.linalg.svd(Xs.T, full_matrices=False)
            del U
            K = (V / S)[:, :p]
            del S
            Xs = np.matmul(Xs, K)
            Xs *= np.sqrt(p)
        # Pre-process y data
        ny = y.shape[0]
        y = convert_y_input(y)
        if len(y.shape) < 2:
            y = np.matrix(y).reshape((ny, 1))
        if ny != n:
            raise (MyException("X and y number of rows must agree"))
        # Pre-process y data when available
        if flag != "one-block":
            ny = y.shape[0]
            y = convert_y_input(y)
            if len(y.shape) < 2:
                y = np.matrix(y).reshape((ny, 1))
            if ny != n:
                raise (MyException("X and y number of rows must agree"))
            if self.copy:
                y0 = copy.deepcopy(y)
                self.y0 = y0
            if self.center_data:
                ys = y  # dont center y
                my = np.mean(y, axis=0)
                sy = np.sqrt(np.var(y, axis=0))
            else:
                ys = y
                my = 0
                sy = 1
            ys = ys.astype("float64")
        else:
            ys = None
        if self.sudiremeth == "sir":
            P = SIR(
                Xs,
                ys,
                self.n_slices,
                self.n_components,
                self.response_type,
                self.center_data,
                self.scale_data,
            )
            if self.scale_data:
                P = np.matmul(N2, P)
            projMat = np.matmul(np.matmul(P, inv(np.matmul(P.T, P))), P.T)
            T = np.matmul(self.X0, P)
        elif self.sudiremeth == "save":
            P = SAVE(
                Xs,
                ys,
                self.n_slices,
                self.n_components,
                self.response_type,
                self.center_data,
                self.scale_data,
            )
            if self.scale_data:
                P = np.matmul(N2, P)
            projMat = np.matmul(np.matmul(P, inv(np.matmul(P.T, P))), P.T)
            T = np.matmul(self.X0, P)
        elif self.sudiremeth == "dr":
            P = DR(
                Xs,
                ys,
                self.n_slices,
                self.n_components,
                self.response_type,
                self.center_data,
                self.scale_data,
            )
            if self.scale_data:
                P = np.matmul(N2, P)
            projMat = np.matmul(np.matmul(P, inv(np.matmul(P.T, P))), P.T)
            T = np.matmul(self.X0, P)
        elif self.sudiremeth == "phd":
            P = PHD(Xs, ys, self.n_components, self.center_data, self.scale_data)
            if self.scale_data:
                P = np.matmul(N2, P)
            projMat = np.matmul(np.matmul(P, inv(np.matmul(P.T, P))), P.T)
            T = np.matmul(self.X0, P)
        elif self.sudiremeth == "iht":
            P = IHT(Xs, ys, self.n_components, self.center_data, self.scale_data)
            if self.scale_data:
                P = np.matmul(N2, P)
            projMat = np.matmul(np.matmul(P, inv(np.matmul(P.T, P))), P.T)
            T = np.matmul(self.X0, P)
        else:  # SDR obtained through optimization of some function

            ## choose starting value for DCOV-SDR
            if self.optimizer_start is None:
                save_start = SAVE(
                    Xs, ys, 3, self.n_components, self.center_data, self.scale_data
                )
                if self.scale_data:
                    save_start = np.matmul(N2, save_start)
                sir_start = SIR(
                    Xs, ys, 6, self.n_components, self.center_data, self.scale_data
                )
                if self.scale_data:
                    sir_start = np.matmul(N2, sir_start)
                beta_save = orth(save_start)
                dc_save = dc.distance_covariance_sqr(np.matmul(Xs, beta_save), y)
                beta_sir = orth(sir_start)
                dc_sir = dc.distance_covariance_sqr(np.matmul(Xs, beta_sir), y)
                DR_start = DR(
                    Xs, y, 6, self.n_components, self.center_data, self.scale_data
                )
                if self.scale_data:
                    DR_start = np.matmul(N2, DR_start)
                beta_DR = orth(DR_start)
                dc_DR = dc.distance_covariance_sqr(np.matmul(Xs, beta_DR), y)
                if dc_save >= dc_sir and dc_save >= dc_DR:
                    self.optimizer_start = save_start.flatten(order="F")
                elif dc_sir >= dc_save and dc_sir >= dc_DR:
                    self.optimizer_start = sir_start.flatten(order="F")
                else:
                    self.optimizer_start = DR_start.flatten(order="F")
            ## add constraints for the optimization
            if self.optimizer_constraints is None:
                const_x = []
                const_z = []
                for i in range(0, self.n_components):
                    for j in range(0, self.n_components):
                        const_x.append(
                            {
                                "type": "eq",
                                "fun": const_xscale,
                                "args": (Xs, self.n_components, i, j),
                            }
                        )  # change const2 to const_func
                        const_z.append(
                            {
                                "type": "eq",
                                "fun": const_zscale,
                                "args": (Xs, self.n_components, i, j),
                            }
                        )
                if self.scale_data:
                    self.optimizer_constraints = tuple(const_z)
                else:
                    self.optimizer_constraints = tuple(const_x)
            if self.optimizer_arguments is None:
                N2 = inv(sqrtm(varmatx))
                self.optimizer_arguments = (
                    Xs,
                    ys,
                    self.n_components,
                    N2,
                    is_distance_mat,
                    self.trimming,
                    self.center,
                    dmetric,
                    biascorr,
                )
            # perform DCOV-SDR optimization
            res = minimize_ipopt(
                dcov_trim,
                self.optimizer_start,
                args=self.optimizer_arguments,
                constraints=self.optimizer_constraints,
                options=self.optimizer_options,
            )
            if self.scale_data:
                dcov_res = np.matmul(
                    N2, np.reshape(res.x, (-1, self.n_components), order="F")
                )
            else:
                dcov_res = np.reshape(res.x, (-1, self.n_components), order="F")
            if self.sudiremeth == "dcov-sdr":
                P = dcov_res
                projMat = np.matmul(np.matmul(P, inv(np.matmul(P.T, P))), P.T)
                T = np.matmul(self.X0, P)
            elif self.sudiremeth == "bcov-sdr":
                self.optimizer_start = dcov_res.flatten(order="F")
                res_ball = minimize_ipopt(
                    ballcov_func,
                    self.optimizer_start,
                    args=self.optimizer_arguments,
                    constraints=self.optimizer_constraints,
                    options=self.optimizer_options,
                )
                if self.scale_data:
                    P = np.matmul(
                        N2, np.reshape(res_ball.x, (-1, self.n_components), order="F")
                    )
                else:
                    P = np.reshape(res_ball.x, (-1, self.n_components), order="F")
                projMat = np.matmul(np.matmul(P, inv(np.matmul(P.T, P))), P.T)
                T = np.matmul(self.X0, P)
            elif self.sudiremeth == "mdd-sdr":
                self.optimizer_start = dcov_res.flatten(order="F")
                res_mdd = minimize_ipopt(
                    mdd_trim,
                    self.optimizer_start,
                    args=self.optimizer_arguments,
                    constraints=self.optimizer_constraints,
                    options=self.optimizer_options,
                )
                if self.scale_data:
                    P = np.matmul(
                        N2, np.reshape(res_mdd.x, (-1, self.n_components), order="F")
                    )
                else:
                    P = np.reshape(res_mdd.x, (-1, self.n_components), order="F")
                projMat = np.matmul(np.matmul(P, inv(np.matmul(P.T, P))), P.T)
                T = np.matmul(self.X0, P)
            else:  ## user defined function
                self.optimizer_start = dcov_res.flatten(order="F")
                opt_res = minimize_ipopt(
                    self.sudiremeth,
                    self.optimizer_start,
                    args=self.optimizer_arguments,
                    constraints=self.optimizer_constraints,
                    options=self.optimizer_options,
                )
                if self.scale_data:
                    P = np.matmul(
                        N2, np.reshape(opt_res.x, (-1, self.n_components), order="F")
                    )
                else:
                    P = np.reshape(opt_res.x, (-1, self.n_components), order="F")
                projMat = np.matmul(np.matmul(P, inv(np.matmul(P.T, P))), P.T)
                T = np.matmul(self.X0, P)
        # perform OLS regression
        T_reg = sm.add_constant(T)  # adding a constant
        ols_obj = sm.OLS(ys, T_reg).fit()

        # Re-adjust estimates to original dimensions if data have been compressed
        if dimensions:
            P = np.matmul(V[:, 0:p], P)
        setattr(self, "x_loadings_", P)
        setattr(self, "x_scores_", T)
        setattr(self, "proj_mat_", projMat)
        if self.whiten_data:
            setattr(self, "whitening_", K)
        setattr(self, "x_loc_", mX)
        setattr(self, "x_sca_", sX)
        setattr(self, "scaling_", scale)
        setattr(self, "ols_obj_", ols_obj)
        if self.return_scaling_object and self.center != "mean":
            setattr(self, "scaling_object_", centring)
        return self

    def transform(self, Xn, distance_mat=False):
        """
        Computes the dimension reduction of the data Xn based on the fitted sudire model.

        Parameters
        ----------
                Xn : matrix or data frame
                     Input data to be transformed 

                distance_mat : numpy array 
                                 distance matrix to represent similarity between observations. 
                
                args or kwargs: 
                                Further parameters to user defined sufdiremeth can be passed here  
        Returns
        -------
        transformed_data : numpy array
                             the dimension reduced data 

         -------
        """
        Xn = convert_X_input(Xn)
        (n, p) = Xn.shape
        (q, h) = self.x_loadings_.shape
        if p != q:
            raise (
                ValueError(
                    "New data must have same number of columns as the ones the model has been trained with"
                )
            )
        return np.matmul(Xn, self.x_loadings_)

    def predict(self, Xn, is_distance_mat=False):
        """
        predicts the response  on new data Xn

        Parameters
        ----------
                Xn : matrix or data frame
                     Input data to be transformed 

                is_distance_mat : bool 
                                 if True, Xn is treated as a distance matrix 
                
        Returns
        -------
        predictions : numpy array 
                      The predictions from the dimension reduction model
        -------
        """
        Xn = convert_X_input(Xn)
        (n, p) = Xn.shape
        (q, h) = self.x_loadings_.shape
        if p != q:
            raise (
                ValueError(
                    "New data must have same number of columns as the ones the model has been trained with"
                )
            )
        Xns = self.transform(Xn)
        Xns = sm.add_constant(Xns)  # adding a constant
        ys = self.ols_obj_.predict(Xns)
        return ys

    @classmethod
    def _get_param_names(cls):
        """Get parameter names for the estimator"""
        # fetch the constructor or the original constructor before
        # deprecation wrapping if any
        init = getattr(cls.__init__, "deprecated_original", cls.__init__)
        if init is object.__init__:
            # No explicit constructor to introspect
            return []
        # introspect the constructor arguments to find the model parameters
        # to represent
        init_signature = inspect.signature(init)
        # Consider the constructor parameters excluding 'self'
        parameters = [
            p
            for p in init_signature.parameters.values()
            if p.name != "self" and p.kind != p.VAR_KEYWORD
        ]
        for p in parameters:
            if p.kind == p.VAR_POSITIONAL:
                raise RuntimeError(
                    "scikit-learn estimators should always "
                    "specify their parameters in the signature"
                    " of their __init__ (no varargs)."
                    " %s with constructor %s doesn't "
                    " follow this convention." % (cls, init_signature)
                )
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
            if deep and hasattr(value, "get_params"):
                deep_items = value.get_params().items()
                out.update((key + "__" + k, val) for k, val in deep_items)
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
            key, delim, sub_key = key.partition("__")
            if key not in valid_params:
                raise ValueError(
                    "Invalid parameter %s for estimator %s. "
                    "Check the list of available parameters "
                    "with `estimator.get_params().keys()`." % (key, self)
                )
            if delim:
                nested_params[key][sub_key] = value
            else:
                setattr(self, key, value)
                valid_params[key] = value
        for key, sub_params in nested_params.items():
            valid_params[key].set_params(**sub_params)
        return self


def estimate_structural_dim(sudiremeth, Xn, y, B, *args, **kwargs):
    """
    Estimates the dimension of the central subspace using 
    the sudiremeth.  This approach is based on the bootstrap method of Sheng and Yin (2016)
    
    Parameters
    ----------

        sudiremeth : str
                    the SDR method to use in the estimation.

        X :  numpy array or dataframe
            Input X data

        Y : vector or 1d matrix
            Input Y data as 

        B : int 
            Number of bootstrap replications

        kwargs:

            n_slices: number of slices for SIR/SAVE/DR

            center_data, bool : if the data  should be centered 

            scale_data, bool :  if the data should be scaled

            center, string :   which centering('mean', 'median')
            
    Returns
    ----------     

    h : int 
        representing the dimension of the central subspace

    ----------
    """
    if "n_slices" not in kwargs:
        n_slices = 6
    else:
        n_slices = kwargs.pop("n_slices")
    if "center_data" not in kwargs:
        center_data = True
    else:
        center_data = kwargs.pop("center_data")
    if "scale_data" not in kwargs:
        scale_data = True
    else:
        scale_data = kwargs.pop("scale_data")
    if "center" not in kwargs:
        center = "mean"
    else:
        center = kwargs.pop("center")
    Xn = convert_X_input(Xn)
    y = convert_y_input(y)
    n, p = Xn.shape

    diff_b = []
    mean_diff = []
    for k in range(1, p + 1):
        print("possible dim", k)
        sdr = sudire(
            sudiremeth,
            center_data=center_data,
            scale_data=scale_data,
            center=center,
            n_slices=n_slices,
            n_components=k,
        )
        sdr.fit(Xn, y=y)
        projMat = sdr.proj_mat_

        for b in range(B):
            idx = np.random.randint(0, n, n)
            X_b = Xn[idx, :].copy()
            sdr_b = sudire(
                sudiremeth,
                center_data=center_data,
                scale_data=scale_data,
                center=center,
                n_slices=n_slices,
                n_components=k,
            )
            sdr_b.fit(X_b, y=y)
            projMat_b = sdr_b.proj_mat_
            uh, sh, vh = np.linalg.svd(projMat - projMat_b)
            diff_b.append(np.nanmax(sh))
        mean_diff.append(np.mean(diff_b))
    return (np.argmin(mean_diff) + 1, mean_diff)
