`direpack`: a Python 3 library for state-of-the-art statistical dimension reduction techniques
==============================================================================================

This package delivers a `scikit-learn` compatible Python 3 package for sundry state-of-the art multivariate statistical methods, with 
a focus on dimension reduction. 

The categories of methods delivered in this package, are: 
- Projection pursuit dimension reduction (`ppdire`) 
- Sufficient dimension reduction (`sudire`)
- Robust M-estimators for dimension reduction (`sprm`)
each of which are presented as `scikit-learn` compatible objects in the corresponding folders.

We hope that this package leads to scientific success. If it does so, we kindly ask to cite the `direpack` vignette \[0\], as well as the original publication of the corresponding method.  

The package also contains a set of tools for pre- and postprocessing: 
- The `preprocessing` folder provides classical and robust centring and scaling, as well as spatial sign transforms \[4\]
- The `dicomo` folder contains a versatile class to access a wide variety of moment and co-moment statistics, and statistics derived from those. Check out the [dicomo Documentation file](https://github.com/SvenSerneels/direpack/blob/master/docs/dicomo.md) and the [dicomo Examples Notebook](https://github.com/SvenSerneels/direpack/blob/master/examples/dicomo_example.ipynb).
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

Methods in the `sudire` folder
------------------------------
The `sudire` folder gives access to an extensive set of methods that resort under the umbrella of sufficient dimension reduction. 
These range from meanwhile long-standing, well-accepted approaches, such as sliced inverse regression (SIR) and the closely related SAVE \[8,9\], 
through methods such as directional regression \[10\] and principal Hessian directions \[11\], and more. However, the package also contains some 
of the most recently developed, state-of-the-art sufficient dimension reduction techniques, that require no distributional assumptions. 
The options provided in this category are based on energy statistics (distance covariance \[12\] or martingale difference divergence \[13\]) and 
ball statistics (ball covariance) \[14\]. All of these options can be called by setting the corresponding parameters in the `sudire` class, cf. [the docs](https://github.com/SvenSerneels/direpack/blob/master/docs/sudire.md). 
Note: the ball covariance option will require some lines to be uncommented as indicated. We decided not to make that option generally available, 
since it depends on the `Ball` package that seems to be difficult to install on certain architectures. 

How to install
--------------
The package is distributed through PyPI, so install through: 
        
        pip install direpack
        
Note that some of the key methods in the `sudire` subpackage rely on the IPOPT 
optimization package, which according to their recommendation, can best be installed
directly as: 

        conda install -c conda-forge cyipopt
        
Documentation
=============

- Detailed documentation can be found in the [ReadTheDocs page](https://direpack.readthedocs.io/en/latest/index.html). 
- A more extensive description on the background is presented in the [`direpack` vignette](https://arxiv.org/abs/2006.01635). 
- Examples on how to use each of the `dicomo`, `ppdire`, `sprm` and `sudire` classes are presented as Jupyter notebooks in the [examples](https://github.com/SvenSerneels/direpack/blob/master/examples) folder
- Furthemore, the [docs](https://github.com/SvenSerneels/direpack/blob/master/docs) folder contains a few markdown files on usage of the classes. 

  
        
References
==========
0. [`direpack`: A Python 3 package for state-of-the-art statistical dimension reduction methods](https://arxiv.org/abs/2006.01635)
1. [Sparse partial robust M regression](https://www.sciencedirect.com/science/article/abs/pii/S0169743915002440), Irene Hoffmann, Sven Serneels, Peter Filzmoser, Christophe Croux, Chemometrics and Intelligent Laboratory Systems, 149 (2015), 50-59.
2. [Partial robust M regression](https://doi.org/10.1016/j.chemolab.2005.04.007), Sven Serneels, Christophe Croux, Peter Filzmoser, Pierre J. Van Espen, Chemometrics and Intelligent Laboratory Systems, 79 (2005), 55-64.
3. [Sparse and robust PLS for binary classification](https://onlinelibrary.wiley.com/doi/abs/10.1002/cem.2775), I. Hoffmann, P. Filzmoser, S. Serneels, K. Varmuza, Journal of Chemometrics, 30 (2016), 153-162.
4. [Spatial Sign Preprocessing:  A Simple Way To Impart Moderate Robustness to Multivariate Estimators](https://pubs.acs.org/doi/abs/10.1021/ci050498u), Sven Serneels, Evert De Nolf, Pierre J. Van Espen, Journal of Chemical Information and Modeling, 46 (2006), 1402-1409.
5. [Robust Continuum Regression](https://www.sciencedirect.com/science/article/abs/pii/S0169743904002667), Sven Serneels, Peter Filzmoser, Christophe Croux, Pierre J. Van Espen, Chemometrics and Intelligent Laboratory Systems, 76 (2005), 197-204.
6. [Robust Multivariate Methods: The Projection Pursuit Approach](https://link.springer.com/chapter/10.1007/3-540-31314-1_32), Peter Filzmoser, Sven Serneels, Christophe Croux and Pierre J. Van Espen, in: From Data and Information Analysis to Knowledge Engineering, Spiliopoulou, M., Kruse, R., Borgelt, C., Nuernberger, A. and Gaul, W., eds., Springer Verlag, Berlin, Germany, 2006, pages 270--277.
7. [Projection pursuit based generalized betas accounting for higher order co-moment effects in financial market analysis](https://arxiv.org/pdf/1908.00141.pdf), Sven Serneels, in: JSM Proceedings, Business and Economic Statistics Section. Alexandria, VA: American Statistical Association, 2019, 3009-3035.
8. [Sliced Inverse Regression for Dimension Reduction](https://www.tandfonline.com/doi/abs/10.1080/01621459.1991.10475035) Li K-C,  Journal of the American Statistical Association (1991), 86, 316-327.
9. [Sliced Inverse Regression for Dimension Reduction: Comment](https://www.jstor.org/stable/2290564?seq=1#metadata_info_tab_contents),  R.D. Cook, and Sanford Weisberg, Journal of the American Statistical Association (1991), 86, 328-332.
10. [On directional regression for dimension reduction](https://doi.org/10.1198/016214507000000536) ,  B. Li and S.Wang, Journal of the American Statistical Association (2007), 102:997–1008.
11. [On principal hessian directions for data visualization and dimension reduction:Another application of stein’s lemma](https://www.tandfonline.com/doi/abs/10.1080/01621459.1992.10476258), K.-C. Li. , Journal of the American Statistical Association(1992)., 87,1025–1039.
12. [Sufficient Dimension Reduction via Distance Covariance](https://doi.org/10.1080/10618600.2015.1026601), Wenhui Sheng and Xiangrong Yin in: Journal of Computational and Graphical Statistics (2016),  25, issue 1, pages 91-104.
13. [A martingale-difference-divergence-based estimation of central mean subspace](https://dx.doi.org/10.4310/19-SII562), Yu Zhang, Jicai Liu, Yuesong Wu and Xiangzhong Fang, in: Statistics and Its Interface (2019),  12, number 3, pages 489-501.
14. [Robust Sufficient Dimension Reduction Via Ball Covariance](https://www.sciencedirect.com/science/article/pii/S0167947319301380) Jia Zhang and Xin Chen, Computational Statistics and Data Analysis 140 (2019) 144–154 
 
 
        
[Release Notes](https://github.com/SvenSerneels/direpack/blob/master/direpack_Release_Notes.md) can be checked out in the repository.  

[A list of possible topics for further development](https://github.com/SvenSerneels/direpack/blob/master/direpack_Future_Dev.md) is provided as well. Additions and comments are welcome!