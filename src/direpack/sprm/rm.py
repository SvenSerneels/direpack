"""
Created on Thu Jan 24 2019

Module containing:
    
    Estimators
    ----------
    Robust M Regression (RM)

Depends on robcent class for robustly centering and scaling data, as well as on
the functions in _m_support_functions. 

@author: Sven Serneels, Ponalytics
"""
from __future__ import absolute_import, division, print_function
from __future__ import unicode_literals
from sklearn.base import RegressorMixin, BaseEstimator
from sklearn.utils.metaestimators import _BaseComposition
import copy
import numpy as np
import pandas as ps
from scipy.stats import norm, chi2
from ..preprocessing.robcent import VersatileScaler
from ..utils.utils import MyException, _predict_check_input, _check_input
from ._m_support_functions import *


class rm(_BaseComposition, BaseEstimator, RegressorMixin):

    """
    Robust M Regression 
    
    Parameters:
    -----------
    fun: str, downweighting function. 'Hampel' (recommended), 'Fair' or 
                'Huber'
    probp1: float, probability cutoff for start of downweighting 
                 (e.g. 0.95)
    probp2: float, probability cutoff for start of steep downweighting 
                 (e.g. 0.975, only relevant if fun='Hampel')
    probp3: float, probability cutoff for start of outlier omission 
                 (e.g. 0.999, only relevant if fun='Hampel')
    centre: str, type of centring (`'mean'`, `'median'` or `'l1median'`, 
            the latter recommended statistically, if too slow, switch to `'median'`)
    scale: str, type of scaling ('std','mad' [recommended] or 'None')
    verbose: boolean, specifying verbose mode
    maxit: int, maximal number of iterations in M algorithm
    tol: float, tolerance for convergence in M algorithm 
    start_cutoff_mode: str, values:
        'specific' will set starting value cutoffs specific to X and y (preferred); 
        any other value will set X and y stating cutoffs identically. 
        The latter yields identical results to the SPRM R implementation available from
        CRAN. 
    copy (def True): boolean, whether to copy data
        Note: copy not yet aligned with sklearn def  
    
    """

    def __init__(
        self,
        fun="Hampel",
        probp1=0.95,
        probp2=0.975,
        probp3=0.999,
        centre="median",
        scale="mad",
        start_cutoff_mode="specific",
        verbose=True,
        maxit=100,
        tol=0.01,
        copy=True,
    ):
        self.fun = fun
        self.probp1 = probp1
        self.probp2 = probp2
        self.probp3 = probp3
        self.centre = centre
        self.scale = scale
        self.start_cutoff_mode = start_cutoff_mode
        self.verbose = verbose
        self.maxit = maxit
        self.tol = tol
        self.copy = copy
        self.probctx_ = "irrelevant"
        self.probcty_ = "irrelevant"
        self.hampelbx_ = "irrelevant"
        self.hampelby__ = "irrelevant"
        self.hampelrx_ = "irrelevant"
        self.hampelry_ = "irrelevant"

    def fit(self, X, y):
        if self.copy:
            self.X = copy.deepcopy(X)
            self.y = copy.deepcopy(y)
        (n, p) = X.shape
        if not (self.fun in ("Hampel", "Huber", "Fair")):
            raise MyException(
                "Invalid weighting function. Choose Hampel, Huber or Fair for parameter fun."
            )
        if (self.probp1 > 1) | (self.probp1 <= 0):
            raise MyException("probp1 is a probability. Choose a value between 0 and 1")
        if self.fun == "Hampel":
            if not (
                (self.probp1 < self.probp2)
                & (self.probp2 < self.probp3)
                & (self.probp3 <= 1)
            ):
                raise MyException(
                    "Wrong choise of parameters for Hampel function. Use 0<probp1<hampelp2<hampelp3<=1"
                )

        if type(X) == ps.core.frame.DataFrame:
            X = X.to_numpy()
        if type(y) in [ps.core.frame.DataFrame, ps.core.series.Series]:
            y = y.to_numpy().astype("float64")
        X = _check_input(X)
        y = _check_input(y)
        if len(y.shape) > 1:
            y = np.array(y).reshape(-1).astype("float64")
        ny = y.shape[0]
        if ny != n:
            raise MyException("Number of cases in y and X must be identical.")

        scaling = VersatileScaler(center=self.centre, scale=self.scale)
        Xs = scaling.fit_transform(X).astype("float64")
        mX = scaling.col_loc_
        sX = scaling.col_sca_
        ys = scaling.fit_transform(y).astype("float64")
        my = scaling.col_loc_
        sy = scaling.col_sca_
        ys = np.array(ys).reshape(-1)

        wx = np.sqrt(np.array(np.sum(np.square(Xs), 1), dtype=np.float64))
        wx = wx / np.median(wx)
        if [self.centre, self.scale] == ["median", "mad"]:
            wy = np.array(abs(ys), dtype=np.float64)
        else:
            wy = (y - np.median(y)) / (1.4826 * np.median(abs(y - np.median(y))))
        self.probcty_ = norm.ppf(self.probp1)
        if self.start_cutoff_mode == "specific":
            self.probctx_ = chi2.ppf(self.probp1, p)
        else:
            self.probctx_ = self.probcty_
        if self.fun == "Fair":
            wx = Fair(wx, self.probctx_)
            wy = Fair(wy, self.probcty_)
        if self.fun == "Huber":
            wx = Huber(wx, self.probctx_)
            wy = Huber(wy, self.probcty_)
        if self.fun == "Hampel":
            self.hampelby_ = norm.ppf(self.probp2)
            self.hampelry_ = norm.ppf(self.probp3)
            if self.start_cutoff_mode == "specific":
                self.hampelbx_ = chi2.ppf(self.probp2, p)
                self.hampelrx_ = chi2.ppf(self.probp3, p)
            else:
                self.hampelbx_ = self.hampelby_
                self.hampelrx_ = self.hampelry_
            wx = Hampel(wx, self.probctx_, self.hampelbx_, self.hampelrx_)
            wy = Hampel(wy, self.probcty_, self.hampelby_, self.hampelry_)
        wx = np.array(wx).reshape(-1)
        w = (wx * wy).astype("float64")
        if (w < 1e-06).any():
            w0 = np.where(w < 1e-06)[0]
            w[w0] = 1e-06
            we = np.array(w, dtype=np.float64)
        else:
            we = np.array(w, dtype=np.float64)
        wye = wy
        WEmat = np.array([np.sqrt(we) for i in range(1, p + 1)], ndmin=1).T
        Xw = np.multiply(Xs, WEmat).astype("float64")
        yw = ys * np.sqrt(we)
        loops = 1
        rold = 1e-5
        difference = 1

        while (difference > self.tol) & (loops < self.maxit):
            b = np.linalg.lstsq(Xw, yw, rcond=None)
            b = np.array(b[0]).reshape(-1, 1)
            yp = np.dot(Xs, b).reshape(-1)
            r = ys - yp
            if len(r) / 2 > np.sum(r == 0):
                r = abs(r) / (1.4826 * np.median(abs(r)))
            else:
                r = abs(r) / (1.4826 * np.median(abs(r[r != 0])))
            wye = r
            if self.fun == "Fair":
                wye = Fair(wye, self.probcty_)
            if self.fun == "Huber":
                wye = Huber(wye, self.probcty_)
            if self.fun == "Hampel":
                wye = Hampel(wye, self.probcty_, self.hampelby_, self.hampelry_)
            b2sum = np.sum(np.square(b))
            difference = abs(b2sum - rold) / rold
            rold = b2sum
            we = (wye * wx).astype("float64")
            w0 = []
            if any(we < 1e-06):
                w0 = np.where(we < 1e-06)[0]
                we[w0] = 1e-06
                we = np.array(we, dtype=np.float64)
            if len(w0) >= (n / 2):
                break
            WEmat = np.array([np.sqrt(we) for i in range(1, p + 1)], ndmin=1).T
            Xw = np.multiply(Xs, WEmat).astype("float64")
            yw = ys * np.sqrt(we)
            loops += 1
        if difference > self.maxit:
            print(
                "Warning: Method did not converge. The scaled difference between norms of the coefficient vectors is "
                + str(round(difference, 4))
            )
        plotprec = False
        if plotprec:
            print(str(loops - 1))
        w = we
        w[w0] = 0
        wx[w0] = 0
        wy = wye
        wy[w0] = 0
        Xrw = np.array(np.multiply(Xs, np.sqrt(WEmat)).astype("float64"))
        scaling.set_params(scale="None")
        Xrw = scaling.fit_transform(Xrw)
        b_rescaled = np.multiply(np.reshape(sy / sX, (p, 1)), b)
        yp_rescaled = np.matmul(X, b_rescaled).reshape(-1)
        if self.centre == "mean":
            intercept = np.mean(y - yp_rescaled)
        else:
            intercept = np.median(y - yp_rescaled)
        yfit = yp_rescaled + intercept
        if self.scale != "None":
            if self.centre == "mean":
                b0 = np.mean(ys.astype("float64") - np.matmul(Xs.astype("float64"), b))
            else:
                b0 = np.median(
                    np.array(ys.astype("float64") - np.matmul(Xs.astype("float64"), b))
                )
        else:
            if self.centre == "mean":
                ytil = np.array(np.matmul(X, b)).reshape(-1)
                intercept = np.mean(y - ytil)
            else:
                intercept = np.median(y - ytil)
            b0 = intercept
        r = y - yfit
        setattr(self, "coef_", b_rescaled)
        setattr(self, "intercept_", intercept)
        setattr(self, "coef_scaled_", b)
        setattr(self, "intercept_scaled_", b0)
        setattr(self, "residuals_", r)
        setattr(self, "fitted_", yfit)
        setattr(self, "x_caseweights_", wx)
        setattr(self, "y_caseweights_", wy)
        setattr(self, "caseweights_", w)
        setattr(self, "x_loc_", mX)
        setattr(self, "y_loc_", my)
        setattr(self, "x_sca_", sX)
        setattr(self, "y_sca_", sy)
        setattr(self, "scaling_", scaling)
        return self
        pass

    def predict(self, Xn):
        n, p, Xn = _predict_check_input(Xn)
        if p != self.X.shape[1]:
            raise (
                ValueError(
                    "New data must have seame number of columns as the ones the model has been trained with"
                )
            )
        return np.matmul(Xn, self.coef_) + self.intercept_
