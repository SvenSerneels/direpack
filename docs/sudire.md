Sufficient Dimension Reduction
======================================

A `scikit-learn` compatible Python 3 package for Sufficient Dimension Reduction. 
This class implements a set of methods to perform Sufficient Dimension Reduction .  

Description
-----------

Sufficient Dimension Reduction(SDR) is a general framework which aims to capture all the relevant information in high dimensional data. This capture of information is based on the notion that a  combination  of the  predictors provides all the relevant information on the response, so that the rest of the predictors can be ignored. 

The different SDR methods   implemented in this class are :  
- `dcov-sdr` : Sufficient Dimension Reduction via Distance Covariance
- `mdd-sdr ` : Sufficient Dimension Reduction via Martingale Difference Divergence
- `sir` : Sliced Inverse Regression
- `save`: Sliced Average Variance Estimation
- ` dr` :  Directional Regression
- ` phd ` : Principal Hessian Directions
- `iht` : Iterative Hessian Transformation 

User defined functions can also be maximised by the method explained in  \[1\]. For more details on how to use the implemented SDR methods and how to use user defined functions, have a look at the [sudire example notebook]() 

The `sudire` class also allows for estimation of the central subspace by optimizing an objective function . The optimization is performed using the Interior Point Optimizer (IPOPT) which is part of the [COIN-OR project](https://coin-or.github.io/Ipopt/) 

Remarks: 
- all the methods contained in this package have been designed for continuous data.  Categorical or textual data first needs to be one hot encoded  or embedded. 
        
The code is aligned to `scikit-learn`, such that modules such as `GridSearchCV` can flawlessly be applied to it. 

The `sudire` folder contains
- The estimator (`sudire.py`) 
- Plotting functions for fitted sudire objects  (`sudire_plot.py`)
- Ancillary functions for sufficient dimension reduction (`_sudire_utils.py`)

The sudire class
================

Dependencies
------------
- From `sklearn.base`: `BaseEstimator`,`TransformerMixin`,`RegressorMixin`
- From `sklearn.utils`: `_BaseComposition`
- `copy`
- From `scipy.stats` : `trim_mean`
- From `scipy.linalg`: `inv`, `sqrtm`
- `cython`
- From  `ipopt` : `minimize_ipopt`
- `numpy` 
- From `statsmodels.regression.linear_model`: `OLS`
- `statsmodels.robust`



Parameters
----------
- `sufdirmeth`, function or string.  one of the elements in the list of implemented SDR methods
                                  or user defined function.
- `n_components`, int.  dimension of the central subspace.
- `trimming`, float. trimming percentage for projection index, to be entered as pct/100 
- `optimizer_options`: dict with options to pass on to the ipopt optimizer. 
   * `tol`: int: relative convergence tolerance.
   * `max_iter`: int. Maximal number of iterations. 
   * `constr_viol_tol` : Desired threshold for the constraint violation.
- `optimizer_constraints`: dict or list of dicts, further constraints to be
            passed on to the optimizer function.
- `center`, str. How to center the data. options accepted are options from
            `direpack`'s `VersatileScaler`. 
- `center_data`, bool. 
- `scale_data`, bool. Note: if set to `False`, convergence to correct optimum 
            is not a given. Will throw a warning. 
- `whiten_data`, bool. 
- `compression`, bool. If `True`, an internal SVD compression step is used for 
            flat data tables (p > n). Speds up the calculations. 
- `copy`, bool. Whether to make a deep copy of the input data or not. 
- `verbose`, bool. Set to `True` prints the iteration number. 
- `return_scaling_object`, bool.
Note:  parameters concerning the data can also be passed to the `fit` method.   

Attributes
----------
Attributes always provided 
-  `x_loadings_`: Estimated basis of the central subsapce 
-  `x_scores_`: The projected X data. 
-  `x_loc_`:  location estimate for X 
-  `x_sca_`:  scale estimate for X
- ` ols_obj` : fitted OLS objected
-  `y_loc_`: y location estimate
-  `y_sca_`: y scale estimate

Attributes created only when corresponding input flags are `True`:
-   `whitening_`: whitened data matrix (usually denoted K)
-   `scaling_object_`: scaling object from `VersatileScaler`


Methods
--------
- `fit(X, *args, **kwargs)`: fit model 
- `predict(X)`: make predictions based on fit 
- `transform(X)`: project X onto latent space 
- `getattr()`: get list of attributes
- `setattr(*kwargs)`: set individual attribute of sprm object 

The `fit` function takes several optional input arguments for user defined objective functions. 
  



        
References
----------
1. [Sufficient Dimension Reduction via Distance Covariance](https://doi.org/10.1080/10618600.2015.1026601), Wenhui Sheng and Xiangrong Yin in: Journal of Computational and Graphical Statistics (2016),  25, issue 1, pages 91-104.
2. [A martingale-difference-divergence-based estimation of central mean subspace](https://dx.doi.org/10.4310/19-SII562), Yu Zhang, Jicai Liu, Yuesong Wu and Xiangzhong Fang, in: Statistics and Its Interface (2019),  12, number 3, pages 489-501.
3. [Sliced Inverse Regression for Dimension Reduction](https://www.tandfonline.com/doi/abs/10.1080/01621459.1991.10475035) Li K-C,  Journal of the American Statistical Association (1991), 86, 316-327.
4. [Sliced Inverse Regression for Dimension Reduction: Comment](https://www.jstor.org/stable/2290564?seq=1#metadata_info_tab_contents),  R.D. Cook, and Sanford Weisberg, Journal of the American Statistical Association (1991), 86, 328-332.
5. [On directional regression for dimension reduction](https://doi.org/10.1198/016214507000000536) ,  B. Li and S.Wang, Journal of the American Statistical Association (2007), 102:997–1008.
6. [On principal hessian directions for data visualization and dimension reduction:Another application of stein’s lemma](https://www.tandfonline.com/doi/abs/10.1080/01621459.1992.10476258), K.-C. Li. , Journal of the American Statistical Association(1992)., 87,1025–1039.
7. [Dimension Reduction for Conditional Mean in Regression](https://pdfs.semanticscholar.org/fd99/4f0cd554790eb8e0449440a59dcd47cf3396.pdf), R. D. Cook and B. Li.,  The Annals of Statistics(2002)30(2):455–474.
8. [Robust Sufficient Dimension Reduction Via Ball Covariance](https://www.sciencedirect.com/science/article/pii/S0167947319301380) Jia Zhang and Xin Chen, Computational Statistics and Data Analysis 140 (2019) 144–154
