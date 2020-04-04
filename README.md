Sparse partial robust M regression
==================================

A `scikit-learn` compatible Python 3 package for robust multivariate regression and dimension reduction statistics, including:
- Sparse Partial Robust M regresion (SPRM)\[1\],  a sparse and robust version of univariate partial least squares (PLS1). 
- Sparse NIPALS Regression (SNIPLS)
- Robust M regression
- Robust centring and scaling   

 ![AIG sprm score space](https://github.com/SvenSerneels/sprm/blob/master/img/AIG_T12.png "AIG SPRM score space")


Description
-----------

The SPRM method performs four tasks at the same time in a single, consistent estimate: 
- *regression*: yields regression coefficients and predicts responses
- *dimension reduction*: calculates interpretable PLS-like components maximizing covariance to the predictand in a robust way 
- *variable selection*: depending on the paramter settings, can yield highly sparse regression coefficients that contain exact zero elements 
- *outlier detection and compensation*: yields a set of case weights in \[0,1\]. The lower the weight, the more outlying a case is. The estimate itself is outlier robust. 

Note: all the methods contained in this package have been designed for continuous data. They do not work correctly for caetgorical or textual data. 
        
The code is aligned to ScikitLearn, such that modules such as `GridSearchCV` can flawlessly be applied to it. 

The repository contains
- The estimator (`sprm.py`) 
- Plotting functionality based on Matplotlib (`sprm_plot.py`)
- Options for data pre-processing (`robcent.py`)
- The Sparse NIPALS (SNIPLS) estimator \[3\](`snipls.py`)
- Robust M regression estimator (`rm.py`)
- Ancillary functions for plotting (`_plot_internals.py`)
- Ancillary functions for M-estimation (`_m_support_functions.py`)
- Ancillary functions for preprocessing (`_preproc_utilities.py`)

How to install
--------------
The package is distributed through PyPI, so install through: 
        
        pip install sprm 
        
Documentation
=============
Detailed documentation on how to use the classes is provided in the [Documentation file](https://github.com/SvenSerneels/sprm/blob/master/docs/sprm.md).


Examples
========
For examples, please have a look at the [SPRM Examples Notebook](https://github.com/SvenSerneels/sprm/blob/master/examples/sprm_example.ipynb).
  
        
References
==========
1. [Sparse partial robust M regression](https://www.sciencedirect.com/science/article/abs/pii/S0169743915002440), Irene Hoffmann, Sven Serneels, Peter Filzmoser, Christophe Croux, Chemometrics and Intelligent Laboratory Systems, 149 (2015), 50-59.
2. [Partial robust M regression](https://doi.org/10.1016/j.chemolab.2005.04.007), Sven Serneels, Christophe Croux, Peter Filzmoser, Pierre J. Van Espen, Chemometrics and Intelligent Laboratory Systems, 79 (2005), 55-64.
3. [Sparse and robust PLS for binary classification](https://onlinelibrary.wiley.com/doi/abs/10.1002/cem.2775), I. Hoffmann, P. Filzmoser, S. Serneels, K. Varmuza, Journal of Chemometrics, 30 (2016), 153-162.
        
[Release Notes](https://github.com/SvenSerneels/sprm/blob/master/SPRM_Release_Notes.md) can be checked out in the repository.  

[A list of possible topics for further development](https://github.com/SvenSerneels/sprm/blob/master/SPRM_Future_Dev.md) is provided as well. Additions and comments are welcome!