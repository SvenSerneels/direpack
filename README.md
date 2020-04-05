`direpack`: a Python 3 library for state-of-the-art statistical dimension reduction techniques
==============================================================================================

This package delivers a `scikit-learn` compatible Python 3 package for some state-of-the art multivariate statistical methods, with 
a focus on dimension reduction. 

The categories of methods delivered in this package, are: 
- Projection pursuit dimension reduction (`ppdire` folder; cf. [ppdire Documentation file](https://github.com/SvenSerneels/direpack/blob/master/docs/ppdire.md) and [ppdire Examples Notebook](https://github.com/SvenSerneels/direpack/blob/master/examples/ppdire_example.ipynb))
- Robust M-estimators for dimension reduction (`sprm` folder; cf. [SPRM Documentation file](https://github.com/SvenSerneels/direpack/blob/master/docs/sprm.md) and [SPRM Examples Notebook](https://github.com/SvenSerneels/direpack/blob/master/examples/sprm_example.ipynb))

The package also contains a set of tools for pre- and postprocessing: 
- The `preprocessing` folder provides classical and robust centring and scaling, as well as spatial sign transforms \[4\]
- The `dicomo` folder contains a versatile class to access a wide variety of moment and co-moment statistics, and statistics derived from those. Check out the [dicomo Examples Notebook](https://github.com/SvenSerneels/direpack/blob/master/examples/dicomo_example.ipynb).
- Plotting utilities in the `plot` folder 
- Cross-validation utilities in the `cross-validation` folder  

 ![AIG sprm score space](https://github.com/SvenSerneels/direpack/blob/master/img/AIG_T12.png "AIG SPRM score space")


Methods in the `sprm` folder
----------------------------
- The estimator (`sprm.py`) \[1\]
- The Sparse NIPALS (SNIPLS) estimator \[3\](`snipls.py`)
- Robust M regression estimator (`rm.py`)
- Ancillary functions for M-estimation (`_m_support_functions.py`)

Methods in the `ppdire` folder
------------------------------
The `ppdire` class will give access to a wide range of projection pursuit dimension reduction techniques.
These include slower approximate estimates for well-established methods such as PCA, PLS and continuum regression. 
However, the class provides unique access to a set of robust options, such as robust continuum regression (RCR) \[5\], through its native `grid` optimization algorithm, first 
published for RCR as well \[6\]. Moreover, `ppdire` is also a great gateway to calculate generalized betas, using the CAPI projection index \[7\]. 

The code is orghanized in 
- `ppdire.py` - the main PP dimension reduction class 
- `capi.py` - the co-moment analysis projection index.      

How to install
--------------
The package is distributed through PyPI, so install through: 
        
        pip install direpack
        
Documentation
=============
Detailed documentation on how to use the classes is provided in the `docs` folder per class.


Examples
========
Jupyter Notebooks with Examples are provided for each of the classes in the `examples` folder.
  
        
References
==========
1. [Sparse partial robust M regression](https://www.sciencedirect.com/science/article/abs/pii/S0169743915002440), Irene Hoffmann, Sven Serneels, Peter Filzmoser, Christophe Croux, Chemometrics and Intelligent Laboratory Systems, 149 (2015), 50-59.
2. [Partial robust M regression](https://doi.org/10.1016/j.chemolab.2005.04.007), Sven Serneels, Christophe Croux, Peter Filzmoser, Pierre J. Van Espen, Chemometrics and Intelligent Laboratory Systems, 79 (2005), 55-64.
3. [Sparse and robust PLS for binary classification](https://onlinelibrary.wiley.com/doi/abs/10.1002/cem.2775), I. Hoffmann, P. Filzmoser, S. Serneels, K. Varmuza, Journal of Chemometrics, 30 (2016), 153-162.
4. [Spatial Sign Preprocessing:  A Simple Way To Impart Moderate Robustness to Multivariate Estimators](https://pubs.acs.org/doi/abs/10.1021/ci050498u), Sven Serneels, Evert De Nolf, Pierre J. Van Espen, Journal of Chemical Information and Modeling, 46 (2006), 1402-1409.
5. [Robust Continuum Regression](https://www.sciencedirect.com/science/article/abs/pii/S0169743904002667), Sven Serneels, Peter Filzmoser, Christophe Croux, Pierre J. Van Espen, Chemometrics and Intelligent Laboratory Systems, 76 (2005), 197-204.
6. [Robust Multivariate Methods: The Projection Pursuit Approach](https://link.springer.com/chapter/10.1007/3-540-31314-1_32), Peter Filzmoser, Sven Serneels, Christophe Croux and Pierre J. Van Espen, in: From Data and Information Analysis to Knowledge Engineering, Spiliopoulou, M., Kruse, R., Borgelt, C., Nuernberger, A. and Gaul, W., eds., Springer Verlag, Berlin, Germany, 2006, pages 270--277.
7. [Projection pursuit based generalized betas accounting for higher order co-moment effects in financial market analysis](https://arxiv.org/pdf/1908.00141.pdf), Sven Serneels, in: JSM Proceedings, Business and Economic Statistics Section. Alexandria, VA: American Statistical Association, 2019, 3009-3035.
        
[Release Notes](https://github.com/SvenSerneels/direpack/blob/master/direpack_Release_Notes.md) can be checked out in the repository.  

[A list of possible topics for further development](https://github.com/SvenSerneels/direpack/blob/master/direpack_Future_Dev.md) is provided as well. Additions and comments are welcome!