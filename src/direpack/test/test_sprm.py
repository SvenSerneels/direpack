# -*- coding: utf-8 -*-
"""
Created on Tue Jun 30 13:17:46 2020

@author: Emmanuel Jordy Menvouta
"""

import unittest
import pandas as ps
import numpy as np
from ..sprm.sprm import sprm
from ..sprm.snipls import snipls
from ..sprm.rm import rm



class Testsprm(unittest.TestCase):
    """ Test some methods in the sprm class"""
    
    @classmethod
    def setUpClass(cls):
        print('setupClass')
        
    @classmethod
    def tearDownClass(cls):
        print('teardownClass')
        
        
    def setUp(self):
        self.data=ps.read_csv("./data/Returns_shares.csv")
        self.datav = np.matrix(self.data.values[:,2:8].astype('float64'))
        self.x = self.datav[:,0:5]
        self.y = self.datav[:,5]
        self.n=self.data.shape[0]
        self.p = self.data.shape[1]
        self.x0 = self.x.astype('float')
        self.y0 = self.y.astype('float')
        self.columns = self.data.columns[2:8]
        
        
        
        
        
        
        
    def tearDown(self):
        del self.x
        del self.y
        del self.n
        del self.p
        del self.x0
        del self.y0
        del self.data
        del self.datav
        
        
        
    
    def test_sprm(self):
        """ Test the functioning of the sprm object"""
        
        res_sprm = sprm(2,.8,'Hampel',.95,.975,.999,'kstepLTS','scaleTau2',True,100,.01,'ally','xonly',self.columns,True)
        res_sprm.fit(self.x0[:2666],self.y0[:2666])
        test_ans = 28.40453479240838
        np.testing.assert_almost_equal(np.linalg.norm(res_sprm.weightnewx(self.x0[2666:])),test_ans,decimal=4)
    
        
    
    
        
        
    def test_snipls(self):
        """ Test the functioning of the snipls object"""
        res_snipls = snipls(n_components=4, eta=.5)
        res_snipls.fit(self.x0[:2666],self.y0[:2666])
        test_ans = 38.6183244001568
        np.testing.assert_almost_equal(np.linalg.norm(res_snipls.predict(self.x0[2666:])),test_ans,decimal=4)
        
        
    def test_rm(self):
        """ Test the functioning of the rm object"""
        
        
        res_rm = rm('Hampel',.95,.975,.999,'median','mad','specific',True,100,.01,True)
        res_rm.fit(self.x0[:2666],self.y0[:2666])
        test_ans = 28.62510008113666
        np.testing.assert_almost_equal(np.linalg.norm(res_rm.predict(self.x0[2666:])),test_ans,decimal=4)
        
        
        
        

        
        
                
    
        
    

    
if __name__ =='__main__':
    unittest.main()
 
