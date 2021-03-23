# -*- coding: utf-8 -*-
"""
Created on Tue Jun 30 13:17:46 2020

@author: Emmanuel Jordy Menvouta
"""

import unittest
import pandas as ps
import numpy as np
from ..preprocessing.robcent import VersatileScaler
import sklearn.decomposition as skd
from ..dicomo.dicomo import dicomo
from ..ppdire.ppdire import ppdire
import sklearn.cross_decomposition as skc

class Testppdire(unittest.TestCase):
    """ Test some methods in the ppdire class"""
    
    @classmethod
    def setUpClass(self):
        print('setupClass')
        
    @classmethod
    def tearDownClass(self):
        print('teardownClass')
        
        
    def setUp(self):
        self.data=ps.read_csv("./data/Returns_shares.csv")
        self.datav = np.matrix(self.data.values[:,2:8].astype('float64'))
        self.x = self.datav[:,1:5]
        self.y = self.datav[:,0]
        self.n=self.data.shape[0]
        self.p = self.data.shape[1]
        self.centring = VersatileScaler()
        self.Xs = self.centring.fit_transform(self.x)
        
        
        
        
        
        
        
    def tearDown(self):
        del self.x
        del self.y
        del self.n
        del self.p
        del self.Xs
        del self.centring
        
        
    
    def test_pca(self):
        """ tests the exactness of ppdire's pca"""
        
        pppca = ppdire(projection_index = dicomo, pi_arguments = {'mode' : 'var'}, n_components=4, optimizer='SLSQP')
        pppca.fit(self.x)
        skpca = skd.PCA(n_components=4)
        skpca.fit(self.Xs)
        np.testing.assert_almost_equal(np.abs(pppca.x_loadings_),np.abs(skpca.components_.T),decimal=3)
        
    def test_pls(self):
        """ tests the exactness of ppdire's pls"""
        
        skpls = skc.PLSRegression(n_components=4)
        skpls.fit(self.Xs,(self.y-np.mean(self.y))/np.std(self.y))
        pppls = ppdire(projection_index = dicomo, pi_arguments = {'mode' : 'cov'}, n_components=4, square_pi=True, optimizer='SLSQP', optimizer_options={'maxiter':500})
        pppls.fit(self.x,self.y)
        np.testing.assert_almost_equal(np.abs(np.matmul(self.Xs,skpls.coef_)*np.std(self.y) + np.mean(self.y)),np.abs(pppls.fitted_),decimal=3)
        
#    def test_robust(self):
#        lcpca = ppdire(projection_index = dicomo, pi_arguments = {'mode' : 'var', 'center': 'median'}, n_components=4, optimizer='grid',optimizer_options={'ndir':1000,'maxiter':10})
#        lcpca.fit(self.x)
#        test_ans=np.array([[ 0.6324543 , -0.00651997, -0.35820225,  0.6438448 ],
#                           [ 0.44750274, -0.67228343,  0.4950862 , -0.21806968],
#                           [ 0.53378114,  0.28794634, -0.46650197, -0.72699245],
#                           [ 0.35432068,  0.68524337,  0.64350842,  0.09692107]])
#        np.testing.assert_almost_equal(np.abs(test_ans),np.abs(lcpca.x_loadings_),decimal=3)
        
    

    
if __name__ =='__main__':
    unittest.main()
 

