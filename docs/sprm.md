Sparse partial robust M regression
==================================

Description
-----------

The `sprm` module in `direpack` comprises code for Sparse Partial Robust M-regeression, as 
well as a few closely related estimators: the Sparse NIPALS estimator (a non-robust option
for sparse PLS) and the Robust M-regression estoimator (multiple regression based on the same
re-weighting priciple as SPRM, yet without dimension reduction). 

The SPRM method performs four tasks at the same time in a single, consistent estimate: 
- *regression*: yields regression coefficients and predicts responses
- *dimension reduction*: calculates interpretable PLS-like components maximizing covariance to the predictand in a robust way 
- *variable selection*: depending on the paramter settings, can yield highly sparse regression coefficients that contain exact zero elements 
- *outlier detection and compensation*: yields a set of case weights in \[0,1\]. The lower the weight, the more outlying a case is. The estimate itself is outlier robust. 

Note: all the methods contained in this package have been designed for continuous data. They do not work correctly for caetgorical or textual data. 
        
The code is aligned to ScikitLearn, such that modules such as `GridSearchCV` can flawlessly be applied to it. 

The repository contains
- The estimator (`sprm.py`) 

- Options for data pre-processing (`robcent.py`)
- The Sparse NIPALS (SNIPLS) estimator \[3\](`snipls.py`)
- Robust M regression estimator (`rm.py`)
- Ancillary functions for M-estimation (`_m_support_functions.py`)

Note that the `plot` folder contains some plotting functionality specific to SPRM (`sprm_plot.py`). 



1\. The SPRM estimator
======================

The main SPRM implementation yields a class with the following structure:

1\.1\. Dependencies
-----------------
- From `<sklearn.base>`: `BaseEstimator, TransformerMixin, RegressorMixin`
- From `<sklearn.utils>`: `_BaseComposition`
- `copy`
- From `<scipy.stats>`: `norm, chi2`
- `numpy` 
- From `<matplotlib>`: `pyplot`. 
- From `<statsmodels>`: `robust`. 

1\.2\. Parameters
---------------
- `eta`: float. Sparsity parameter in \[0,1). Note that `eta=0` returns the non-sparse, yet robust, partial robust M-regression (PRM) \[2\]. 
- `n_components`: int > 1. Note that if applied on data, `n_components` shall take a value <= min(x_data.shape)
- `fun`: str, downweighting function. `'Hampel'` (recommended), `'Fair'` or `'Huber'`
- `probp1`: float, probability cutoff for start of downweighting (e.g. 0.95)
- `probp2`: float, probability cutoff for start of steep downweighting (e.g. 0.975, only relevant if `fun='Hampel'`)
- `probp3`: float, probability cutoff for start of outlier omission (e.g. 0.999, only relevant if `fun='Hampel'`)
- `centring`: str, type of centring (`'mean'`, `'median'`,`'l1median'` or `'kstepLTS'`)
- `scaling`: str, type of scaling (`'std'`,`'mad'`, `'scaleTau2'`, the latter recommended, or `'None'`)
- `verbose`: boolean, specifying verbose mode
- `maxit`: int, maximal number of iterations in M algorithm
- `tol`: float, tolerance for convergence in M algorithm 
- `start_cutoff_mode`: str, value `'specific'` will set starting value cutoffs specific to X and y (preferred); any other value will set X and y stating cutoffs identically. The non-specific setting yields identical results to the SPRM R implementation available from [CRAN](https://cran.r-project.org/web/packages/sprm/index.html).
- `start_X_init`: str, values `'pcapp'` will include a PCA/broken stick projection to calculate the initial predictor block caseweights; any other value will just calculate initial predictor block case weights based on Euclidian distances within that block. The is less stable for very flat data (p >> n). 
- `colums` (def `False`): Either bool, list, numpy array or pandas Index
        if `False`, no column names supplied
        if `True`, 
            if X data are supplied as a pandas DataFrame, will extract column 
                names from the frame
            else throws an error
        if a list, array or Index (will only take length x_data.shape[1]), 
            the column names of the x_data supplied in this list, 
            will be printed in verbose mode
- `copy` (def `True`): boolean, whether to create deep copy of the data in the calculation process 

1\.3\. Attributes
---------------
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
-  `x_caseweights_`: X block case weights
-  `y_caseweights_`: y block case weights
-  `caseweights_`: combined case weights
-  `colret_`: names of variables retained in the sparse model
-  `x_loc_`: X block location estimate 
-  `y_loc_`: y location estimate
-  `x_sca_`: X block scale estimate
-  `y_sca_`: y scale estimate
-  `non_zero_scale_vars_`: indicator vector of variables in X with nonzero scale

1\.4\. Methods
------------
- `fit(X,y)`: fit model 
- `predict(X)`: make predictions based on fit 
- `transform(X)`: project X onto latent space 
- `weightnewx(X)`: calculate X case weights
- `getattr()`: get list of attributes
- `setattr(**kwargs)`: set individual attribute of sprm object 
- `valscore(X,y,scoring)`: option to use weighted scoring function in cross-validation if scoring=weighted 

1\.5\. Ancillary functions 
------------------------
- `snipls` (class): sparse NIPALS regression (first described in: \[3\]) 
- `Hampel`: Hampel weight function 
- `Huber`: Huber weight function 
- `Fair`: Fair weight function 
- `brokenstick`: broken stick rule to estimate number of relevant principal components  
- `VersatileScaler` (class): centring and scaling data, with several robust options beyond `sklearn`'s `RobustScaler` 
- `sprm_plot` (class): plotting SPRM results 
- `sprm_plot_cv` (class): plotting SPRM cross-validation results
        
        
2\. The Robust M (RM) estimator
==============================

RM has been implemented to be consistent with SPRM. It takes the same arguments, except for `eta`, `n_components` and `columns`, 
because it does not perform dimension reduction nor variable selection. For the same reasons, the outputs are limited to regression
outputs. Therefore, dimension reduction outputs like `x_scores_`, `x_loadings_`, etc. are not provided. For R adepts, note that a
[cellwise robust](https://github.com/SebastiaanHoppner/CRM) version of RM has recently been introduced. 
        
        
3\. The Sparse NIPALS (SNIPLS) estimator
=======================================

SNIPLS is the non-robust sparse univariate PLS algorithm \[3\]. SNIPLS has been implemented to be consistent with SPRM. It takes the same arguments, except for `'fun'` and `'probp1'` through `'probp3'`, since these are robustness parameters. For the same reasons, the outputs are limited to sparse dimension reduction and regression outputs. Robustness related outputs like `x_caseweights_` cannot be provided.
        
 
4\. Plotting functionality
=========================

The file `sprm_plot.py` contains a set of plot functions based on Matplotlib. The class sprm_plot contains plots for sprm objects, wheras the class sprm_plot_cv contains a plot for cross-validation. 

4\.1\. Dependencies
-----------------
- `pandas`
- `numpy`
- `matplotlib.pyplot`
- for plotting cross-validation results: `sklearn.model_selection.GridSearchCV`

4\.2\. Paramaters
---------------
- `res_sprm`, sprm. An sprm class object that has been fit.  
- `colors`, list of str entries. Only mandatory input. Elements determine colors as: 
    - \[0\]: borders of pane 
    - \[1\]: plot background
    - \[2\]: marker fill
    - \[3\]: diagonal line 
    - \[4\]: marker contour, if different from fill
    - \[5\]: marker color for new cases, if applicable
    - \[6\]: marker color for harsh calibration outliers
    - \[7\]: marker color for harsh prediction outliers
- `markers`, a list of str entries. Elements determkine markers for: 
    - \[0\]: regular cases 
    - \[1\]: moderate outliers 
    - \[2\]: harsh outliers 
    
4\.3\. Methods
------------
- `plot_coeffs(entity="coef_",truncation=0,columns=[],title=[])`: Plot regression coefficients, loadings, etc. with the option only to plot the x% smallest and largets coefficients (truncation) 
- `plot_yyp(ytruev=[],Xn=[],label=[],names=[],namesv=[],title=[],legend_pos='lower right',onlyval=False)`: Plot y vs y predicted. 
- `plot_projections(Xn=[],label=[],components = [0,1],names=[],namesv=[],title=[],legend_pos='lower right',onlyval=False)`: Plot score space. 
- `plot_caseweights(Xn=[],label=[],names=[],namesv=[],title=[],legend_pos='lower right',onlyval=False,mode='overall')`: Plot caseweights, with the option to plot `'x'`, `'y'` or `'overall'` case weights for cases used to train the model. For new cases, only `'x'` weights can be plotted. 

4\.4\. Remark
-----------
The latter 3 methods will work both for cases that the models has been trained with (no additional input) or new cases (requires Xn and in case of plot_ypp, ytruev), with the option to plot only the latter (option onlyval = True). All three functions have the option to plot case names if supplied as list.       

4\.5\. Ancillary classes
---------------------- 
- `sprm_plot_cv` has method `eta_ncomp_contour(title)` to plot sklearn GridSearchCV results 


References
==========
1. [Sparse partial robust M regression](https://www.sciencedirect.com/science/article/abs/pii/S0169743915002440), Irene Hoffmann, Sven Serneels, Peter Filzmoser, Christophe Croux, Chemometrics and Intelligent Laboratory Systems, 149 (2015), 50-59.
2. [Partial robust M regression](https://doi.org/10.1016/j.chemolab.2005.04.007), Sven Serneels, Christophe Croux, Peter Filzmoser, Pierre J. Van Espen, Chemometrics and Intelligent Laboratory Systems, 79 (2005), 55-64.
3. [Sparse and robust PLS for binary classification](https://onlinelibrary.wiley.com/doi/abs/10.1002/cem.2775), I. Hoffmann, P. Filzmoser, S. Serneels, K. Varmuza, Journal of Chemometrics, 30 (2016), 153-162.
        