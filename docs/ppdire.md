Projection Pursuit Dimension Reduction
======================================

A `scikit-learn` compatible Python 3 package for Projection Pursuit Dimension Reduction. 
This class implements a very general framweork for projection pursuit, giving access to 
methods ranging from PP-PCA to CAPI generalized betas.  

Description
-----------

Projection pursuit (PP) provides a very general framework for dimension reduction and regression. The
`ppdire` class provides a framework to calculate PP estimates based on a wide variety of projection 
indices. 

While the class will also work with user-defined projection indices, a set of projection indices are 
included into the `direpack` package as two ancillary classes: 
- `dicomo` class for (co-)moment statistics (separate folder), cf the [dicomo Documentation file](https://github.com/SvenSerneels/direpack/blob/master/docs/dicomo.md)
- `capi` specifically for analyzing financial market returns based on a linear combination of co-moments \[2\] 

When using the `dicomo` class as a plugin, several well-known multivariate dimension reduction techniques 
are accessible, as well as robust alternatives thereto. For more details, have a look at the [ppdire examples notebook](https://github.com/SvenSerneels/direpack/blob/master/examples/ppdire_example.ipynb). 

The `ppdire` class allows for calculation of the projection pursuit optimization either 
through `scipy.optimize` or through the native grid\[1\] algorithm. Optimization through 
`scipy.optimize` is much more efficient, yet it will only provide correct results 
for classical projection indices. The native grid algorithm should be used when 
the projection index involves order statistics of any kind, such as ranks, trimming, 
winsorizing, or empirical quantiles.

Remarks: 
- all the methods contained in this package have been designed for continuous data. They do not work correctly for categorical or textual data.
- this package focuses on projection pursuit dimension reduction. Regression methods that involve a dimension reduction step can be accessed through it 
  (e.g. PCR, PLS, RCR, ...), yet the package does not provide an implementation for projection pursuit regression (PPR). To access PPR, we refer to 
  the `projection-pursuit` package, also distributed through PIP.    
        
The code is aligned to `scikit-learn`, such that modules such as `GridSearchCV` can flawlessly be applied to it. 

The `ppdire` folder contains
- The estimator (`ppdire.py`) 
- A class for the co-moment analysis projection index (`capi.py`)
- Ancillary functions for projection pursuit (`_ppdire_utils.py`)

The ppdire class
================

Dependencies
------------
- From `sklearn.base`: `BaseEstimator`,`TransformerMixin`,`RegressorMixin`
- From `sklearn.utils`: `_BaseComposition`
- `copy`
- `scipy.stats`
- From `scipy.linalg`: `pinv2`
- From `scipy.optimize`: `minimize`
- `numpy` 
- From `statsmodels.regression.quantile_regression`: `QuantReg`
- From `sklearn.utils.extmath`: `svd_flip`


Parameters
----------
- `projection_index`, function or class. `dicomo` and `capi` supplied in this
            package can both be used, but user defined projection indices can 
            be processed 
- `pi_arguments`, dict. Dict of arguments to be passed on to `projection index` 
- `n_components`, int. number of components to be estimated 
- `trimming`, float. trimming percentage for projection index, to be entered as pct/100 
- `alpha`, float. Continuum coefficient. Only relevant if `ppdire` is used to 
            estimate (classical or robust) continuum regression. 
- `optimizer`: str. Presently: either `'grid'` (native optimizer) or 
            any of the options in `scipy-optimize` (e.g. `'SLSQP'`)
- `optimizer_options`: dict with options to pass on to the optimizer. 
            If `optimizer == 'grid'`,
   * `ndir`: int: Number of directions to calculate per iteration.
   * `maxiter`: int. Maximal number of iterations. 
- `optimizer_constraints`: dict or list of dicts, further constraints to be
            passed on to the optimizer function.
- `regopt`, str. Regression option for regression step y~T. Can be set
                to `'OLS'` (default), `'robust'` (will run `sprm.rm`) or `'quantile'` 
                (`statsmodels.regression.quantreg`). 
- `center`, str. How to center the data. options accepted are options from
            `direpack`'s `VersatileScaler`. 
- `center_data`, bool. 
- `scale_data`, bool. Note: if set to `False`, convergence to correct optimum 
            is not a given. Will throw a warning. 
- `whiten_data`, bool. Typically used for ICA (kurtosis as PI)
- `square_pi`, bool. Whether to square the projection index upon evaluation. 
- `compression`, bool. If `True`, an internal SVD compression step is used for 
            flat data tables (p > n). Speds up the calculations. 
- `copy`, bool. Whether to make a deep copy of the input data or not. 
- `verbose`, bool. Set to `True` prints the iteration number. 
- `return_scaling_object`, bool.
Note: several interesting parameters can also be passed to the `fit` method.   

Attributes
----------
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
-  `coef_scaled_`: vector of scaled regression coeeficients (when scaling option used)
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


Methods
--------
- `fit(X, *args, **kwargs)`: fit model 
- `predict(X)`: make predictions based on fit 
- `transform(X)`: project X onto latent space 
- `getattr()`: get list of attributes
- `setattr(*kwargs)`: set individual attribute of sprm object 

The `fit` function takes several optional input arguments. These are flags that 
typically would not need to be cross-validated. They are: 
-   `y`, numpy vector or 1D matrix, either as `arg` directly or as `kwarg`
-   `h`, int. Overrides `n_components` for an individual call to `fit`. Use with caution. 
-   `dmetric`, str. Distance metric used internally. Defaults to `'euclidean'`
-   `mixing`, bool. Return mixing matrix? 
-   Further parameters to the regression methods can be passed on here 
    as additional `kwargs`. 
  

Ancillary functions 
-------------------
- `dicomo` (class):  (co-)moments 
- `capi` (class): co-moment analysis projection index 

        
References
----------
1. [Robust Multivariate Methods: The Projection Pursuit Approach](https://link.springer.com/chapter/10.1007/3-540-31314-1_32), Peter Filzmoser, Sven Serneels, Christophe Croux and Pierre J. Van Espen, in: From Data and Information Analysis to Knowledge Engineering,
        Spiliopoulou, M., Kruse, R., Borgelt, C., Nuernberger, A. and Gaul, W., eds., 
        Springer Verlag, Berlin, Germany,
        2006, pages 270--277.
2. [Projection pursuit based generalized betas accounting for higher order co-moment effects in financial market analysis](https://arxiv.org/pdf/1908.00141.pdf), Sven Serneels, in: 
        JSM Proceedings, Business and Economic Statistics Section. Alexandria, VA: American Statistical Association, 2019, 3009-3035.
3. Robust principal components and dispersion matrices via projection pursuit, Chen, Z. and Li, G., Research Report, Department of Statistics, Harvard University, 1981.
4. [Robust Continuum Regression](https://www.sciencedirect.com/science/article/abs/pii/S0169743904002667), Sven Serneels, Peter Filzmoser, Christophe Croux, Pierre J. Van Espen, Chemometrics and Intelligent Laboratory Systems, 76 (2005), 197-204.
