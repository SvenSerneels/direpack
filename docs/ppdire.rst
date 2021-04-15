.. _ppdire:


################
ppdire
################

Beyond discussion, the class of dimension reduction with the longest standing history accessible through direpack, is projection pursuit (PP) dimension reduction.
Let $\mathbf{X}$ be a data matrix that is a sample of $n$ cases of a $p$ variate random variable and $\mathbf{y}$ be a sample of a corresponding depending variable, when applicable. 
The set of projection pursuit scores $\mathbf{t}_i$ that span the columns of $\mathbf{T}$ are defined as linear combinations of the original variables: $\mathbf{T} = \mathbf{X}\mathbf{W}$, where the $\mathbf{w}_i$ are
the solution to the optimization problem: 

.. math::
   :label: optim_ppdire
   :nowrap:
    
    \begin{equation*}
    \begin{aligned}
    & \underset{\mathbf{a}}{\text{maximise}} & & \mathfrak{P}\left(\mathbb{S}\left(\mathbf{a}^T\mathbf{X}\right)\right) \\
    & \text{subject to} & & \mathbf{w}_i^T\mathbf{X}^T\mathbf{X}\mathbf{w}_j = 0 \mbox{ and } \parallel \mathbf{w}_i\parallel_2 = 1,\\
    \end{aligned}
    \end{equation*}
    
     

where $i,j \in [1,\min(n,p)]$, $j > i$  and the set $\mathbb{S} = \{\mathbf{X},\mathbf{y}\}$ if data for a dependent variable $Y$ exist and is a singleton containing $\mathbf{X}$ otherwise.
Maximization of this criterion is very flexible and the properties of the dimension reduction accomplished according to it can vary widely, mainly dependent on the presence or absence of dependent 
variable data, as well as on $\mathfrak{P}$, which in the PP literature is referred to as the  projection index. 

dicomo
===========

The projection index determines which method is being calculated. 
In direpack, projection pursuit can be called through the ppdire subpackge and class object, which allows the user to pass any function of appropriate dimensionality as a projection index. 
However, a set of popular projection indices deriving from (co-)moments, are provided as well through the dicomo subpackage. For several of these, plugging them in leads to well-established methods. They comprise: 

     * Moment statistics: variance (PCA), higher order moments 
     * Co-moment statistics: covariance (PLS), higher order co-moments 
     * Standardized moments: skewness (ICA), kurtosis (ICA)
     * Standardized co-moments: correlation coefficient (CCA), co-skewness, co-kurtosis
     * Linear combinations of (standardized co-) moments. Here, the capi.py file in the ppdire subpackage delivers to co-moment analysis projection index (Serneels2019). 
     * Products of (co-)moments. Particularly the continuum association measure has been provided, which is given by $\mathop{\mbox{cont}}(\mathbf{X},\mathbf{y}) = \mathop{\mbox{cov}}(\mathbf{X},\mathbf{y})\mathop{\mbox{var}}(\mathbf{X})^{\alpha-1}$. Using this continuum measure produces continuum regression  (CR, Stone and Brooks (1990)). CR is equivalent to PLS for $\alpha = 1$ and approaches PCA as $\alpha \rightarrow\infty$.   



pp optimizers
==============

Early ideas behind PP was the ability to scan all directions maximizing the projection index as denoted in  :eq:`optim_ppdire`. This essentially corresponds to a brute force optimization technique, which can be computationally very demanding.
For instance, both PCA and PLS, can be solved analytically, leading to efficient algorithms that do not directly optimize :eq:`optim_ppdire`. Whenever the projection index plugged in, leads to a convex optimization problem, it is advisable to apply an efficient numerical optimization technique.  For that purpose,ppdire has the option to use scipy.optimize’s sequential least squares quadratic programming optimization (SLSQP). However, for projection indices based on ordering or ranking data, such as medians or trimmed (co-)moments, the problem is no longer convex and cannot be solved through SLSQP. 
For those purposes, the grid algorithm is included, which was originally developed to compute RCR (Filzmoser, Serneels, Croux, andVan Espen 2006). 

Regularized regression
=======================

While the main focus of direpack is dimension reduction, all dimension reduction techniques offer a bridge to regularized regression. 
This can be achieved by regressing the dependent variable onto the estimated dimension reduced space. The latter provides regularization of the covariance matrix,
due to the constraints in :eq:`optim_ppdire`, and allow to perform regression for an undersampled $\mathbf{X}$. The classical estimate is to predict $\mathbf{y}$ through least squares regression: 

.. math::
   :nowrap:

    \begin{equation*}
    \hat{\mathbf{y}}  = \hat{\mathbf{T}} \hat{\mathbf{T}}^T\mathbf{y}
    \end{equation*}

which again leads to well-established methods such as principal component regression (PCR), PLS regression, etc.



Usage
===========

.. currentmodule:: direpack

.. autosummary::
    :toctree: generated/
        
    ppdire

.. autosummary::
    :toctree: generated/
        
    dicomo




Dependencies
================

- From `sklearn.base`: `BaseEstimator`,`TransformerMixin`,`RegressorMixin`
- From `sklearn.utils`: `_BaseComposition`
- `copy`
- `scipy.stats`
- From `scipy.linalg`: `pinv2`
- From `scipy.optimize`: `minimize`
- `numpy` 
- From `statsmodels.regression.quantile_regression`: `QuantReg`
- From `sklearn.utils.extmath`: `svd_flip`


References
==========
1. Peter Filzmoser, Sven Serneels, Christophe Croux and Pierre J. Van Espen, Robust Multivariate Methods: The Projection Pursuit Approach,  in: From Data and Information Analysis to Knowledge Engineering,Spiliopoulou, M., Kruse, R., Borgelt, C., Nuernberger, A. and Gaul, W., eds.,  Springer Verlag, Berlin, Germany, 2006, pages 270--277.

2. Sven Serneels, Projection pursuit based generalized betas accounting for higher order co-moment effects in financial market analysis,  in: JSM Proceedings, Business and Economic Statistics Section. Alexandria, VA: American Statistical Association, 2019, 3009-3035.

3. Chen, Z. and Li, G., Robust principal components and dispersion matrices via projection pursuit,  Research Report, Department of Statistics, Harvard University, 1981.

4. Peter Filzmoser, Christophe Croux, Pierre J. Van Espen, Robust Continuum Regression, Sven Serneels,  Chemometrics and Intelligent Laboratory Systems, 76 (2005), 197-204.

5. Stone  M,  Brooks  RJ  (1990).   “Continuum  Regression:   Cross-Validated  Sequentially  Constructed Prediction Embracing Ordinary Least Squares, Partial Least Squares and PrincipalComponents Regression.”Journal of the Royal Statistical Society. Series B (Methodological),52, 237–269.