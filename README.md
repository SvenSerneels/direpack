`direpack`: a Python 3 library for state-of-the-art statistical dimension reduction techniques
==============================================================================================

This package delivers a `scikit-learn` compatible Python 3 package for some state-of-the art multivariate statistical methods, with 
a focus on dimension reduction. 

The categories of methods delivered in this package, are: 
- Projection pursuit dimension reduction (`ppdire` folder; cf. [SPRM Documentation file](https://github.com/SvenSerneels/direpack/blob/master/docs/sprm.md) and [SPRM Examples Notebook](https://github.com/SvenSerneels/direpack/blob/master/examples/sprm_example.ipynb))
- Robust M-estimators for dimension reduction (`sprm` folder; cf. docs and examples)

The package also contains a set of tools for pre- and postprocessing: 
- The `preprocessing` folder provides classical and robust centring and scaling, as well as spatial sign transforms
- Plotting utilities in the `plot` folder 
- Cross-validation utilities in the `cross-validation` folder  

 ![AIG sprm score space](https://github.com/SvenSerneels/direpack/blob/master/img/AIG_T12.png "AIG SPRM score space")


Methods in the `sprm` folder
----------------------------
- The estimator (`sprm.py`) \[1\]
- The Sparse NIPALS (SNIPLS) estimator \[3\](`snipls.py`)
- Robust M regression estimator (`rm.py`)
- Ancillary functions for M-estimation (`_m_support_functions.py`)

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
        
[Release Notes](https://github.com/SvenSerneels/direpack/blob/master/direpack_Release_Notes.md) can be checked out in the repository.  

[A list of possible topics for further development](https://github.com/SvenSerneels/direpack/blob/master/direpack_Future_Dev.md) is provided as well. Additions and comments are welcome!