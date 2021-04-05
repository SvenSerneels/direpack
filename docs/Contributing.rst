.. _Contributing:

################
Contributing
################

No package is complete and the authors would like to see direpack extend its functionality in the future. Some possible additions could be : 

-  Cellwise robust dimension reduction methods : For instance, a cellwise robust version of the robust M regression method, included in sprm, has recently been published (Filzmoseret  al.2020), and could be included in direpack.
-  Uncertainty quantification : The methods provided through direpack provide point estimates. In the future, the package could, e.g. be augmented with appropriate bootstrapping techniques, as was done for a related dimension reduction context
-  GPU flexibility : There are many matrix manipulations in direpack, which can possiblybe  sped  up  by  allowing  a  GPU  compatibility,  which  could  be  achieved  by  providing a TensorFlowor PyTorch back-end. However, this would be a major effort, since thepresent back-end integrally builds upon numpy.
-  More (and better) unit tests. 

Guidelines
============

Testing
-------
Contributions should be accompanied by unit tests similar to those already available. Contrbutors can use the datasets presented in the example notebooks. 

Documentation
-------------
We have followed `PEP8 <https://www.python.org/dev/peps/pep-0008/>`_ style  when building this project and ask that contributors do so,
for ease of maintainability. 

Article
================
An article with further information on the package is available. Menvouta, E. J., S. Serneels, T. Verdonck (2020). direpack: A Python 3 pack-age for state-of-the-art statistical dimension reduction methods.arXiv  e-prints, arXiv:2006.01635.

Contacts
================

* Dr Sven Serneels is Sr. Director of Data Science at  Aspen Technology and can be contacted at svenserneel (at) gmail.com.

* Emmanuel Jordy Menvouta is a PhD researcher in Statistics and Data Science at KU Leuven and can be contacted at emmanueljordy.menvoutankpwele (at) kuleuven.be. 

* Prof Tim Verdonck is Professor of Statistics and Data Science at University of Antwerp and KU Leuven. He can be reached at tim.verdonck (at) uantwerp.be.