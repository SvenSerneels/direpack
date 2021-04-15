

Welcome to direpack's documentation!
====================================
The direpack package aims to establish a set of modern statistical dimension reduction techniques into the Python universe as a single, consistent package.
The dimension reduction methods included resort into three categories: projection pursuit based dimension reduction, sufficient dimension reduction, and robust M estimators for dimension reduction. 
As a corollary, regularized regression estimators based on these reduced dimension spaces are provided as well, ranging from classical principal component regression up to sparse partial robust M regression.
The package also contains a set of classical and robust pre-processing utilities, including generalized spatial signs,  as well as dedicated plotting functionality and cross-validation utilities. 
Finally, direpack has been written consistent with the scikit-learn API, such that the estimators can flawlessly be included into (statistical and/or machine) learning pipelines in that framework.



Installation
============
The package is distributed through PyPI, so use:: 

      pip install direpack

Examples
===============
Example notebooks have been produced to showcase the use of direpack for statistical dimension reduction. These notebooks contain a  `ppdire example <https://github.com/SvenSerneels/direpack/blob/master/examples/ppdire_example.ipynb>`_ ,  `sprm example <https://github.com/SvenSerneels/direpack/blob/master/examples/sprm_example.ipynb>`_  and a `sudire example <https://github.com/SvenSerneels/direpack/blob/master/examples/sudire_example.ipynb>`_ . 




Contents
========

.. toctree::
   :maxdepth: 2

   ppdire
   sudire
   sprm
   Pre-processing
   Cross-validation and plotting


.. toctree::
   :maxdepth: 1
   :caption: Other information 

   Contributing
   



Indices and tables
==================

* :ref:`genindex`
* :ref:`search`
