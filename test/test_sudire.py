# -*- coding: utf-8 -*-
"""
Created on Tue Jun 30 13:17:46 2020

@author: Emmanuel Jordy Menvouta
"""

import unittest
import numpy as np
import pandas as pd
from ..src.direpack.sudire.sudire import sudire, estimate_structural_dim
from sklearn.model_selection import train_test_split

class Testsudire(unittest.TestCase):
    """ Test some methods in the sudire class"""
    
    @classmethod
    def setUpClass(cls):
        print('setupClass')
        
    @classmethod
    def tearDownClass(cls):
        print('teardownClass')
        
        
    def setUp(self):
        self.data=pd.read_csv('./data/auto-mpg.csv', index_col='car_name')
        self.data = self.data[self.data.horsepower != '?']
        self.data.horsepower = self.data.horsepower.astype('float')
        self.x = self.data
        self.y = self.x['mpg']
        self.x.drop('mpg', axis=1, inplace=True)
        self.x.drop('origin', axis = 1, inplace = True)            
        self.n=self.x.shape[0]
        self.p = self.x.shape[1]
        self.struct_dim = 2
        self.x_train, self.x_test, self.y_train, self.y_test = train_test_split(
        self.x, self.y, test_size=0.3, random_state=42)
        
    def tearDown(self):
        del self.x
        del self.y
        del self.n
        del self.p
        del self.data
        del self.struct_dim
        del self.x_train
        del self.x_test
        del self.y_train
        del self.y_test
        
    def test_estimdim(self):
        """ Tests the estimation of the central subspace via Bootstrap """
        
        central_dim, diff_vec = estimate_structural_dim('dr',self.x_train.values,self.y_train.values , B=100, n_slices=4)
        np.testing.assert_equal(central_dim,6)
        
    def test_sir(self):
        """ Tests Sliced Inverse Regression"""
        
        mod_auto = sudire('sir', center_data= True, scale_data=True,n_components=self.struct_dim)
        mod_auto.fit(self.x_train.values, self.y_train.values)
        test_ans = 1.4142135623730947
        np.testing.assert_almost_equal(np.linalg.norm(mod_auto.x_loadings_),test_ans,decimal=14)
        
    def test_save(self):
        """ Tests Sliced Average Variance Estimation """
        
        mod_auto = sudire('save', center_data= True, scale_data=True,n_components=self.struct_dim)
        mod_auto.fit(self.x_train.values, self.y_train.values)
        test_ans = 1.4142135623730943
        np.testing.assert_almost_equal(np.linalg.norm(mod_auto.x_loadings_),test_ans,decimal=14)
        
    def test_dr(self):
        """ Tests Directional Regression """
        
        mod_auto = sudire('dr', center_data= True, scale_data=True,n_components=self.struct_dim)
        mod_auto.fit(self.x_train.values, self.y_train.values)
        test_ans = 1.4142135623730934
        np.testing.assert_almost_equal(np.linalg.norm(mod_auto.x_loadings_),test_ans,decimal=14)
        
    def test_iht(self):
        """ Tests Iterative Hessian Transformations """
        
        mod_auto = sudire('iht', center_data= True, scale_data=True,n_components=self.struct_dim)
        mod_auto.fit(self.x_train.values, self.y_train.values)
        test_ans = 1.4142135623730934
        np.testing.assert_almost_equal(np.linalg.norm(mod_auto.x_loadings_),test_ans,decimal=14)
        
    def test_phd(self):
        """ Tests Principal Hessian Directions """
        
        mod_auto = sudire('phd', center_data= True, scale_data=True,n_components=self.struct_dim)
        mod_auto.fit(self.x_train.values, self.y_train.values)
        test_ans = 1.4142135623730894
        np.testing.assert_almost_equal(np.linalg.norm(mod_auto.x_loadings_),test_ans,decimal=14)
        
# Tests for DCOV and MDD based SDR can be run as below. However, they require IPOPT to be installed 
# independently of the Python packages, which is hard to ascertain in an online GitHub workflow. 
# Moreover, the result may slightly differ numberically depending on the solver used internally in 
# IPOPT.
        
#    def test_dcov(self):  
#        """ Test DCOV based SDR"""
#        
#        mod_auto = sudire('dcov-sdr', center_data= True, scale_data=True,n_components=self.struct_dim)
#        mod_auto.fit(self.x_train.values, self.y_train.values)
#        test_ans = 1.1985000652583924
#        np.testing.assert_almost_equal(np.linalg.norm(mod_auto.x_loadings_),test_ans,decimal=5)
#        
#        
#    def test_mdd(self):
#        
#        """ Test MDD based SDR"""
#        mod_auto = sudire('mdd-sdr', center_data= True, scale_data=True,n_components=self.struct_dim)
#        mod_auto.fit(self.x_train.values, self.y_train.values)
#        test_ans = 0.3793912951554523
#        np.testing.assert_almost_equal(np.linalg.norm(mod_auto.x_loadings_),test_ans,decimal=5)
    
        
    
        
        
        
        
        
        
        
        
        
        
if __name__ =='__main__':
    unittest.main()
        
 
        
        
