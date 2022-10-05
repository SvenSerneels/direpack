#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun May 12 10:03:05 2019

@author: Sven Serneels, Ponalytics.
"""

from sklearn.base import BaseEstimator, defaultdict
from sklearn.utils.metaestimators import _BaseComposition
from collections import defaultdict
import inspect
from ..dicomo.dicomo import dicomo
from ..dicomo._dicomo_utils import *


class capi(_BaseComposition, BaseEstimator):

    """
    CAPI Co-moment analysis projection index
    
    The CAPI projection index to estimate generalized betas was first introduced
    in: 
    
    S. Serneels, Projection pursuit based generalized betas accounting for 
    higher order co-moment effects in financial market analysis,  in: 
    JSM Proceedings, Business and Economic Statistics Section. 
    Alexandria, VA: American Statistical Association, 2019, 3009-3035.
    
    Class arguments 
    
    max_degree, int: maxmimal degree of co-moments to be used. In [2,3,4]. 
    
    projection_index, class object: class used to calculate co-moments. 
        Written to work with dicomo class yet other plugins could be written.
    
    pi_arguments, dict: dict of arguments to pass on to projection_index
    
    weights, list of float: weights to used in linear combination of co-moments. 
    
    centring, bool
    
    scaling, bool whether to calculate CAPI based on scaled higher co-moments 
        (co-skewness, co-kurtosis) or raw higher co-moments
    
    options, either a list of co-moment options to be included, or 'all' (e.g.
    option=i calculates M3,i and M4,i etc.)
    
    After intializing the object, call object.fit(x,y,**kwargs) to evaluate. 
    CAPI takes no direct kwargs, yet passes all kwargs on to the fit method of
    the projection index. 
    
    """

    def __init__(
        self,
        max_degree=2,
        projection_index=dicomo,
        pi_arguments={},
        weights=[1, 1, 1, -1, -1, -1],
        centring=False,
        scaling=True,
        options="all",
    ):
        self.max_degree = max_degree
        self.projection_index = projection_index
        self.pi_arguments = pi_arguments
        self.weights = weights
        self.most = self.projection_index(**self.pi_arguments)
        self.scaling = scaling
        self.options = options
        self.capi_index_ = None
        if self.max_degree > 4:
            raise (ValueError("Maximal degree is 4."))

    def fit(self, x, y, **kwargs):

        if self.scaling:
            order_kwargs = ["cov", "cos", "cok"]
        else:
            order_kwargs = ["com", "com", "com"]

        if self.max_degree < 2:
            raise (ValueError("capi not meaningful for max_degree < 2"))
        if self.options == "all":
            options = np.arange(1, 4)
        else:
            options = np.array(self.options, ndmin=1)
        moments = np.zeros(6)
        fit_arguments = {"order": 0, "y": y}
        fit_arguments = {**kwargs, **fit_arguments}
        init_moment_calc = 2
        k = 0
        for i in range(init_moment_calc, self.max_degree + 1):
            fit_arguments["order"] = i
            self.most.set_params(mode=order_kwargs[i - 2])
            l = min(i - 1, len(options))
            for j in options[np.arange(0, l)]:
                fit_arguments["option"] = j
                moments[i - 3 + j + k] = self.most.fit(x, **fit_arguments)
            if i == 3:
                k += 1
        capi_index_ = np.dot(self.weights, moments)
        self.capi_index_ = capi_index_
        self.moments_ = moments
        return capi_index_

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
