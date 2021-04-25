.. _sprm:

################
sprm
################

Sparse partial robust M regression (SPRM) is a sparse and robust alternative to PLS that can be calculated efficiently (Hoffmann, Serneels, Filzmoser,and Croux 2015). 
The subpackage is organized slightly differently from the other two mainsubpackages.  Because SPRM combines the virtues of robust regression with sparse dimension reduction, 
besides  the  SPRM  estimators  itself,  each  of  these  building  blocks  are  provided themselves as class objects that can be deployed in sklearn pipelines. 
The class objects rm, snipls and sprm are sourced by default when importing direpack. 

Robust M regression
=====================

M regression is a generalization of least squares regression in the sense that it minimizes a more general objective that allows to tune the estimator's robustness.
In M regression, the vector of regression coefficients is defined as: 

.. math::
   :label: optim_rm
   :nowrap:

    \begin{equation*}
    \hat{\boldsymbol{\beta}} = \mathop{\mbox{argmin}}_{\boldsymbol{\beta}}\sum_i \rho\left(\frac{r_i(\boldsymbol{\beta})}{\hat{\sigma}}\right) 
    \end{equation*}

where $r_i$ are the casewise regression residuals and $\hat{\sigma}$ is a robust scale estimator thereof. The $\rho$ function defines the properties of the estimator.
Identity to the least squares estimator is obtained if $\rho(r) = r^2$, but robustness can be introduced by taking a different function,
for instance a function that is approximately quadratic for small (absolute) $r$, but increases more slowly than $r^2$ for larger values of $r$. 
Objective :eq:`optim_rm` can be solved numerically, but it is well known that its solution can equivalently be obtained through an iteratively reweighting least squares (IRLS),
which is how it is implemented in sprm. In the package, the Fair, Huber or Hampel reweighting functions can be picked, which will lead to different robustness properties.



Sparse NIPALS
=====================

A second building block in the package is the SNIPLS algorithm. It is a sparse version of the NIPALS algorithm for PLS and as such, essentially a computationally efficient implementation of univariate sparse PLS.
Again, the SNIPLS components are linear combinations of the original variables through a set of weighting vectors $\mathbf{w}_i$ that maximize: 

.. math::
   :label: optim_snipls
   :nowrap:
    
    \begin{equation*}
    \begin{aligned}
    & 	\mathbf{w}_i &= \argmax_{\mathbf{a}} \mathop{\mbox{cov}^2}\left(\mathbf{a}^T\mathbf{X},\mathbf{y}\right) + \lambda \parallel\mathbf{a}\parallel_1 \\
    & \text{subject to} & \mathbf{w}_i^T\mathbf{X}^T\mathbf{X}\mathbf{w}_j = 0 \mbox{ and } \parallel \mathbf{w}_i\parallel_2 = 1\\
    \end{aligned}
    \end{equation*}


which in sparse PLS is typically maximized through a surrogate formulation.  However,  in this  case,  the  exact  solution  to  Criterion  :eq:`optim_snipls`  can  be  obtained,
which  is  what  the  SNIPLS algorithm  builds  upon.   For  details  on  the  algorithm,  the  reader  is  referred  to  Hoffmann, Filzmoser, Serneels, and Varmuza (2016).  
At this point, remark that the SNIPLS algorithm has also become a key building block to analyze outlyingness (Debruyne, Höppner, Serneels,and Verdonck 2019).





Sparse partial robust M
=========================

Sparse  partial  robust  M  dimension  reduction  unites  the  benefits  of  SNIPLS  and  robust  M estimation:  it yields an efficient sparse PLS dimension reduction, while at the same time,
it is robust against both leverage points and virtual outliers through robust M estimation. It is defined similarly as in :eq:`optim_snipls` but instead maximizing a weighted covariance, with case weights that depend on the data. 
Consistent with robust M estimation, it can be calculated through iteratively reweighting SNIPLS. SPRM improves upon the original reweighted PLS proposal by (i) yielding a sparse estimate, (ii) having a reweighting scheme as well as starting values that weight both in the score and residual spaces and (iii) by allowing different weight functions, the most tuneable one being the Hampel function.

Usage
===========

.. currentmodule:: direpack.sprm.sprm

.. autosummary::
    :toctree: generated/
    :caption: SPRM
        
    sprm

       
.. currentmodule:: direpack.sprm.snipls

.. autosummary::
    :toctree: generated/
    :caption: SNIPLS

    snipls
    
        
    






Dependencies
================

- `pandas`
- `numpy`
- `matplotlib.pyplot`
- for plotting cross-validation results: `sklearn.model_selection.GridSearchCV`


References
================

1. Irene Hoffmann, Sven Serneels, Peter Filzmoser, Christophe Croux, Sparse partial robust M regression, Chemometrics and Intelligent Laboratory Systems, 149 (2015), 50-59.
2. Sven Serneels, Christophe Croux, Peter Filzmoser, Pierre J. Van Espen, Partial robust M regression,  Chemometrics and Intelligent Laboratory Systems, 79 (2005), 55-64.
3. Hoffmann I., P. Filzmoser, S. Serneels, K. Varmuza, Sparse and robust PLS for binary classification,  Journal of Chemometrics, 30 (2016), 153-162.
4. Filzmoser P, Höppner S, Ortner I, Serneels S, Verdonck T. Cellwise robust M regression.  Computational Statistics and Data Analysis,147 (2020).
