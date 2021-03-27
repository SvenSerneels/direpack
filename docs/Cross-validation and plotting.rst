.. _Cross-validation and plotting:

#############################
Cross-validation and plotting
#############################

Each of the sudire, ppdire and sprm subpackages in direpackare wrappers around a broad class of dimension reduction methods.  
Each of these methods will have at least one tune-able  hyperparameter;  some  have  many  more. The  user  will  want  to  be  able  to  find  the optimal hyperparameters for the data at hand, which can be done through cross-validation or bayesian optimization.
It is not the aim of direpack to provide its own hyperparameter tuning algorithms,  as ample cross-validation utilities are available in scikit-learn’s model selection subpackage and the direpack estimations have been written consistently with the scikit-learn API,
such that these model selection tools from scikit-learn can directly be applied to them. However, some caution should be taken when training the robust methods.  While all classical (non-robust) methods could just use scikit-learn’s default settings, when tuning a robust model, 
outliers  are  expected  to  be  in  the  data,  such  that  it  becomes  preferable  to  apply a robust  cross-validation  metric  as  well. Thereunto,  it  is  possible  to  use scikit-learn’s median_absolute_error, which is an MAE (L1) scorer that is less affected by extreme values than the default mean_squared_error. 
However, particularly in the case of robust M estimators, a more model consistent approach can be pursued.  The robust M estimators provide a set of case weights,  and these can be used to construct a weighted evaluation metric for cross-validation.  Exactly this is provided in the robust_loss function that is a part of the direpack cross-validation utilities.

Similar to hyperparameter tuning, direpack's mission is not to deliver a broad set of plotting utilities, but rather focus on the dimension reduction statistics. However, some plots many users would like to have in this context, are provided for each of the methods. These are : 

* Projection plots. These plots visualize the scores $\mathbf{t}_i$ and a distinction can be made in the plots between cases that the model had been trained with, and test set cases. 
* Parity plots. For the regularized regressions based on the estimated scores, these visualize the predicted versus actual responses, with the same distinction as for the scores. 

For the special case of SPRM, the plots have enhanced functionality. Since SPRM provides case weights, which can also be calculated for new cases, the SPRM plots can flag outliers. In the sprm_plot function, this is set up with two cut-offs, based on the caseweight values,and visualized asregular  cases,moderate  outliersorharsh  outliers.
For SPRM, there is anoption as well to visualize the case weights themselves.


Examples of direpack's plotting functionalities are available in the example notebooks of `ppdire <https://github.com/SvenSerneels/direpack/blob/master/examples/ppdire_example.ipynb>`_,  `sprm <https://github.com/SvenSerneels/direpack/blob/master/examples/sprm_example.ipynb>`_ and `sudire <https://github.com/SvenSerneels/direpack/blob/master/examples/sudire_example.ipynb>`_ . 

