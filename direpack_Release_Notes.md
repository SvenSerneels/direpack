`sprm` Release notes (versions 0.0 through 0.7)
====================

Version 0.2.1
-------------
- sprm now takes both numeric (n,1) np matrices and (n,) np.arrays as input 


Version 0.2.0
-------------
Changes compared to version 0.1: 
- All functionalities can now be loaded in modular way, e.g. to use plotting functions, now source the plot function separately:
        
        from sprm import sprm_plot 
        
- The package now includes a robust M regression estimator (rm.py), which is a multiple regression only variant of sprm. 
  It is based on the same iterative re-weighting scheme, buit does not perform dimension reduction, nor variable selection.
- The robust preprocessing routine (robcent.py) has been re-written so as to be more consistent with sklearn.

Version 0.3
-----------
All three estimators provided as separate classes in module:

        from sprm import sprm 
        from sprm import snipls
        from sprm import rm
        
Also, sprm now includes a check for zero scales. It will remove zero scale variables from the input data, and only use 
columns corresponding to nonzero predictor scales in new data. This check has not yet been built in for snipls or rm 
separately. 
        
Plus some minor changes to make it consistent with the latest numpy and matplotlib versions. 

Version 0.4
-----------
The preprocesing routine `robcent` has been refactored. Functionality has been 
added to centre the data nonparametrically by the L1 median. The ancillary functions
for `robcent` have been moved into `_preproc_utilities.py`. 

Furthermore, `sprm`, `snipls` and `rm` have all three been modified such that
they accept matrix, array or data frame input for both X and y. Also, the option
to provide column names has been extended to automatic extraction from data frame
input, or direct input as list, array or pandas Index. 

The license has been changed from GPL3 to MIT. 

0.4.2. `'kstepLTS'` location estimator included.


Version 0.5 
-----------
Pre-processing functions further refactored so as to be compatible with `sklearn` pipelines. 
Class now named `VersatileScaler`; the old `robcent` name still works, but will be sunset. 

Version 0.6
-----------
Preprocessing files moved into separate folder. More preprocessing options. 
Examples moved into Jupyter notebook in separate examples section.

`direpack` release notes (since version 0.8)
========================

Version 0.8
-----------
`ppdire` merges in

Version 0.9
-----------
- `preprocessing` widely extended 
- `plot` functions adapted 
- documentation improved 

Version 1.0
-----------
- `sudire` joins in
- `plot` functions adapted 
- documentation provided for `dicomo` 
- 1.0.2: link to `direpack` publication added
- 1.0.3: fixed rare division by zero in `l1median`
- 1.0.4: unit tests included
- 1.0.5: `sudire` notebook adapted
- 1.0.9: function to calculate the martingale difference divergence matrix (MDDM) added in `_dicomo_utils.py` 
- 1.0.11: documentation updated to accommodate for go-live of readthedocs page




 

