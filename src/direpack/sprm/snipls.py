# Created on Fri Apr 26 19:27:52 2019

# @author: sven


from __future__ import absolute_import, division, print_function
from __future__ import unicode_literals
from sklearn.base import RegressorMixin, BaseEstimator, TransformerMixin
from sklearn.utils.metaestimators import _BaseComposition
import copy
import numpy as np
import pandas as ps
from ..preprocessing.robcent import VersatileScaler
from ..utils.utils import MyException, _predict_check_input, _check_input
from ..preprocessing._preproc_utilities import scale_data


class snipls(_BaseComposition, BaseEstimator, TransformerMixin, RegressorMixin):
    """
    SNIPLS Sparse Nipals Algorithm 
    
    Algorithm first outlined in: 
        Sparse and robust PLS for binary classification, 
        I. Hoffmann, P. Filzmoser, S. Serneels, K. Varmuza, 
        Journal of Chemometrics, 30 (2016), 153-162.
    
    Parameters
    -----------

    eta : float.
         Sparsity parameter in [0,1)

    n_components : int,
                     min 1. Note that if applied on data, n_components shall take a value <= min(x_data.shape)

    verbose: Boolean (def true)
                to print intermediate set of columns retained

    columns : Either boolean, list, numpy array or pandas Index (def false)
                if False, no column names supplied; if True, if X data are supplied as a pandas data frame, will extract column names from the frame throws an error for other data input types if a list, array or Index (will only take length x_data.shape[1]), the column names of the x_data supplied in this list, will be printed in verbose mode.

    centre : str, 
                type of centring (`'mean'` [recommended], `'median'` or `'l1median'`), 

    scale : str,
             type of scaling ('std','mad' or 'None')

    copy : (def True): boolean,
             whether to copy data.  Note : copy not yet aligned with sklearn def  - we always copy  
    
             
    Attributes
    ------------
    Attributes always provided:

        -  `x_weights_`: X block PLS weighting vectors (usually denoted W)
        -  `x_loadings_`: X block PLS loading vectors (usually denoted P)
        -  `C_`: vector of inner relationship between response and latent variablesblock re
        -  `x_scores_`: X block PLS score vectors (usually denoted T)
        -  `coef_`: vector of regression coefficients 
        -  `intercept_`: intercept
        -  `coef_scaled_`: vector of scaled regression coeeficients (when scaling option used)
        -  `intercept_scaled_`: scaled intercept
        -  `residuals_`: vector of regression residuals
        -  `x_ev_`: X block explained variance per component
        -  `y_ev_`: y block explained variance 
        -  `fitted_`: fitted response
        -  `x_Rweights_`: X block SIMPLS style weighting vectors (usually denoted R)
        -  `colret_`: names of variables retained in the sparse model
        -  `x_loc_`: X block location estimate 
        -  `y_loc_`: y location estimate
        -  `x_sca_`: X block scale estimate
        -  `y_sca_`: y scale estimate
        -  `centring_`: scaling object used internally (from `VersatileScaler`)
    
    """

    def __init__(
        self,
        eta=0.5,
        n_components=1,
        verbose=True,
        columns=False,
        centre="mean",
        scale="None",
        copy=True,
    ):
        self.eta = eta
        self.n_components = n_components
        self.verbose = verbose
        self.columns = columns
        self.centre = centre
        self.scale = scale
        self.copy = copy

    def fit(self, X, y):
        """
            Fit a  SNIPLS model. 
            
            Parameters
            ------------ 
                
                X : numpy array 
                    Input data.

                y :   vector or 1D matrix
                    Response data

        """
        if type(self.columns) is list:
            self.columns = np.array(self.columns)
        elif type(self.columns) is bool:
            if type(X) != ps.core.frame.DataFrame and self.columns:
                raise (
                    MyException(
                        "Columns set to true can only extract column names for data frame input"
                    )
                )
        if type(X) == ps.core.frame.DataFrame:
            if type(self.columns) is bool and self.columns:
                self.columns = X.columns
            X = X.to_numpy()
        (n, p) = X.shape
        if type(y) in [ps.core.frame.DataFrame, ps.core.series.Series]:
            y = y.to_numpy()
        X = _check_input(X)
        y = _check_input(y)
        ny = y.shape[0]
        if ny != n:
            if y.ndim == 2:
                y = y.T
            else:
                raise (MyException("Number of cases in X and y needs to agree"))
        y = y.astype("float64")
        if self.copy:
            X0 = copy.deepcopy(X)
            y0 = copy.deepcopy(y)
        else:
            X0 = X
            y0 = y
        self.X = X0
        self.y = y0
        X0 = X0.astype("float64")
        centring = VersatileScaler(center=self.centre, scale=self.scale)
        X0 = centring.fit_transform(X0).astype("float64")
        mX = centring.col_loc_
        sX = centring.col_sca_
        y0 = centring.fit_transform(y0).astype("float64")
        my = centring.col_loc_
        sy = centring.col_sca_
        T = np.empty((n, self.n_components), float)
        W = np.empty((p, self.n_components), float)
        P = np.empty((p, self.n_components), float)
        C = np.empty((self.n_components, 1), float)
        Xev = np.empty((self.n_components, 1), float)
        yev = np.empty((self.n_components, 1), float)
        B = np.empty((p, 1), float)
        oldgoodies = np.array([])
        Xi = X0
        yi = y0
        for i in range(1, self.n_components + 1):
            wh = np.dot(Xi.T, yi)
            wh = wh / np.linalg.norm(wh, "fro")
            # goodies = abs(wh)-llambda/2 lambda definition
            goodies = abs(wh) - self.eta * max(abs(wh))
            wh = np.multiply(goodies, np.sign(wh))
            goodies = np.where((goodies > 0))[0]
            goodies = np.union1d(oldgoodies, goodies)
            oldgoodies = goodies
            if len(goodies) == 0:
                print(
                    "No variables retained at"
                    + str(i)
                    + "latent variables"
                    + "and lambda = "
                    + str(self.eta)
                    + ", try lower lambda"
                )
                break
            elimvars = np.setdiff1d(range(0, p), goodies)
            wh[elimvars] = 0
            th = np.dot(Xi, wh)
            nth = np.linalg.norm(th, "fro")
            ch = np.dot(yi.T, th) / (nth ** 2)
            ph = np.dot(Xi.T, np.dot(Xi, wh)) / (nth ** 2)
            Xi = Xi - np.dot(th, ph.T)
            yi = yi - np.dot(th, ch)
            ph[elimvars] = 0
            W[:, i - 1] = np.reshape(wh, p)
            P[:, i - 1] = np.reshape(ph, p)
            C[i - 1] = ch
            T[:, i - 1] = np.reshape(th, n)
            Xev[i - 1] = (
                (nth ** 2 * np.linalg.norm(ph, "fro") ** 2)
                / np.sum(np.square(X0))
                * 100
            )
            yev[i - 1] = np.sum(nth ** 2 * (ch ** 2)) / np.sum(np.power(y0, 2)) * 100
            if type(self.columns) == bool:
                colret = goodies
            else:
                colret = self.columns[np.setdiff1d(range(0, p), elimvars)]
            if self.verbose:
                print(
                    "Variables retained for "
                    + str(i)
                    + " latent variable(s):"
                    + "\n"
                    + str(colret)
                    + ".\n"
                )
        if len(goodies) > 0:
            R = np.matmul(
                W[:, range(0, i)],
                np.linalg.inv(np.matmul(P[:, range(0, i)].T, W[:, range(0, i)])),
            )
            B = np.matmul(
                W[:, range(0, i)],
                np.matmul(
                    np.linalg.inv(
                        np.matmul(
                            np.matmul(W[:, range(0, i)].T, np.matmul(X0.T, X0)),
                            W[:, range(0, i)],
                        )
                    ),
                    np.matmul(np.matmul(W[:, range(0, i)].T, X0.T), y0),
                ),
            )
        else:
            B = np.empty((p, 1))
            B.fill(0)
            R = B
            T = np.empty((n, self.n_components))
            T.fill(0)
        B_rescaled = np.multiply(np.array(sy / sX).reshape((p, 1)), B)
        yp_rescaled = np.dot(X, B_rescaled)
        if self.centre == "mean":
            intercept = np.mean(y - yp_rescaled)
        else:
            intercept = np.median(y - yp_rescaled)
        yfit = yp_rescaled + intercept
        yfit = yfit.reshape(-1)
        r = y - yfit
        setattr(self, "x_weights_", W)
        setattr(self, "x_loadings_", P)
        setattr(self, "C_", C)
        setattr(self, "x_scores_", T)
        setattr(self, "coef_", B_rescaled)
        setattr(self, "coef_scaled_", B)
        setattr(self, "intercept_", intercept)
        setattr(self, "x_ev_", Xev)
        setattr(self, "y_ev_", yev)
        setattr(self, "fitted_", yfit)
        setattr(self, "residuals_", r)
        setattr(self, "x_Rweights_", R)
        setattr(self, "colret_", colret)
        setattr(self, "x_loc_", mX)
        setattr(self, "y_loc_", my)
        setattr(self, "x_sca_", sX)
        setattr(self, "y_sca_", sy)
        setattr(self, "centring_", centring)
        return self

    def predict(self, Xn):
        """
        Predict using a  SNIPLS model. 
        
        Parameters
        ------------ 
            
            Xn : numpy array or data frame 
                Input data.

        """
        n, p, Xn = _predict_check_input(Xn)
        if p != self.X.shape[1]:
            raise (
                ValueError(
                    "New data must have same number of columns as the ones the model has been trained with"
                )
            )
        return np.matmul(Xn, self.coef_) + self.intercept_

    def transform(self, Xn):
        """
        Transform input data. 
        
        
        Parameters
        ------------ 
            
            Xn : numpy array or data frame 
                Input data.

        """
        n, p, Xn = _predict_check_input(Xn)
        if p != self.X.shape[1]:
            raise (
                ValueError(
                    "New data must have seame number of columns as the ones the model has been trained with"
                )
            )
        Xnc = scale_data(Xn, self.x_loc_, self.x_sca_)
        return Xnc * self.x_Rweights_
