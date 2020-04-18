Diverse (co-)moment statistics 
==============================

This class implements (co)-moment statistics, covering both clasical product-moment 
statistics, as well as more recently developed energy statistics. 
The `dicomo` class also serves as a plug-in into `capi` and  `ppdire`. It has been written consistently with `ppdire` such that it provides a wide range of 
projection indices based on (co-)moments.    

Description
-----------

The `dicomo` folder contains
- The class object (`dicomo.py`) 
- Ancillary functions for (co-)moment estimation (`_dicomo_utils.py`)

The `dicomo` class
==================

Parameters
----------
- `est`, str: mode of estimation. The set of options are `'arithmetic'` (product-moment) or `'distance'` (energy statistics)
- `mode`, str: type of moment. Options are: 
    * `'mom'`: moment 
    * `'var'`: variance 
    * `'std'`: standard deviation 
    * `'skew'`: skewness 
    * `'kurt'`: kurtosis
    * `'com'`: co-moment 
    * `'M3'`: shortcut for third order co-moment
    * `'cov'`: covariance 
    * `'cos'`: co-skewness
    * `'cok'`: co-kurtosis 
    * `'corr'`: correlation, 
    * `'continuum'`: continuum association 
    * `'mdd'`: martingale difference divergence (requires `est = 'distance'`)
    * `'mdc'`: martingale difference correlation (requires `est = 'distance'`)
    * `'ballcov'`: ball covariance (requires installing `Ball` and uncommenting the `import` statement)
- `center`: internal centring used in calculation. Options are `mean` or `median`.  

Attributes
----------
Attributes always provided 
-  `moment_`: The resulting (co-)moment

Depending on the options picked, intermediate results are stored as well, as `x_moment_`, `y_moment_` or `co_moment_`


Methods
--------
- `fit(X, *args, **kwargs)`: fit model 

The `fit` function takes several optional input arguments. These are options that 
apply to individual settings: 
-   `biascorr`, Bool, when `True`, correct for bias. For classical product-moment statistics, this 
    is the small sample correction. For energy statistics, this leads to the estimates that are unbiased in high dimension
    (but not preferred in low dimension). 
-   `alpha`, float, parameter for continuum association. Has no effect for other options.  
-   `option`, int, determines which higher order co-moment to calculate, e.g. for co-skewness, `option=1` calciulates CoS(x,x,y)
-   `order`, int, which order (co-)moment to calculate. Can be overruled by `mode`, e.g. if `mode='var'`, `order` is set to 2. 
-   `calcmode`, str, to use the efficient or naive algorithm to calculate distance statistics. Defaults to `fast` when available. 

Examples 
--------
Check out the [dicomo examples notebook](https://github.com/SvenSerneels/direpack/blob/master/examples/dicomo_example.ipynb)