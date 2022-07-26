# -*- coding: utf-8 -*-
"""
Created on Tue Jun 30 13:17:46 2020

@author: Emmanuel Jordy Menvouta
"""

import unittest
from ..dicomo.dicomo import dicomo
import pandas as ps
import numpy as np
import statsmodels.robust as srs
import scipy.stats as sps
import dcor as dc

class Testdicomo(unittest.TestCase):
    """ Test  methods in the dicomo class"""
    
    @classmethod
    def setUpClass(cls):
        print('...setupClass')
        
        
        
    @classmethod
    def tearDownClass(cls):
        print('...teardownClass')
        
        
    @classmethod    
    def setUp(self):
        self.data=ps.read_csv("./data/Returns_shares.csv")
        self.datav = np.matrix(self.data.values[:,2:8].astype('float64'))
        self.est = dicomo()
        self.x = self.datav[:,1]
        self.y = self.datav[:,0]
        self.n=self.data.shape[0]
        self.p = self.data.shape[1]
        
        
        
    @classmethod    
    def tearDown(self):
        del self.est
        del self.x
        del self.y
        del self.n
        del self.p
        
        
    
    def test_mom(self):
        """ Tests functions to compute moments"""
        
        self.assertAlmostEquals(self.est.fit(self.x,biascorr=False),np.var(self.x))# biased var
        self.assertAlmostEquals(self.est.fit(self.x,biascorr=True),np.var(self.x)*self.n/(self.n-1))#unbiased var
        self.est.set_params(center='median')
        self.assertAlmostEquals(self.est.fit(self.x),srs.mad(self.x)[0],places=4)
        self.est.set_params(center='mean')
        self.assertAlmostEquals(self.est.fit(self.x,biascorr=False,order=3),sps.moment(self.x,3)[0])#third moment
        self.est.set_params(mode='skew')
        self.assertAlmostEquals(self.est.fit(self.x,biascorr=False),sps.skew(self.x)[0])# skew without small sample corr
        self.assertAlmostEquals(self.est.fit(self.x,biascorr=True),sps.skew(self.x,bias=False)[0])
        
        
        
        
        
        
    def test_como(self):
        """ Tests function to compute comomennts"""
        
        self.est.set_params(mode='com')
        self.assertAlmostEquals(self.est.fit(self.x,y=self.y,biascorr=True),self.data.iloc[:,2:4].cov().values[0,1])#covariance
        self.assertAlmostEquals(self.est.fit(self.x,y=self.y,biascorr=True,option=1,order=3),0.39009,places=4)#third order comoment
        self.est.set_params(mode='corr')
        self.assertAlmostEquals(self.est.fit(self.x,y=self.y),self.data.iloc[:,2:4].corr().values[0,1])#correlation
        self.est.set_params(mode='continuum')
        self.assertAlmostEquals(np.sqrt(self.est.fit(self.x,y=self.y,alpha=1,biascorr=True)),self.data.iloc[:,2:4].cov().values[0,1])#continuum
        
        
    def test_energy(self):
        """ Tests function  to compute energy statistics"""
        
        self.est.set_params(est='distance',mode='var')
        self.assertAlmostEquals(self.est.fit(self.x,biascorr=False),dc.distance_stats(self.x,self.x).covariance_xy)
        self.assertAlmostEquals(self.est.fit(self.x,biascorr=True),np.sqrt(dc.u_distance_stats_sqr(self.x,self.x).covariance_xy))      
        self.est.set_params(mode='com')
        self.assertAlmostEquals(self.est.fit(self.x,y=self.y,biascorr=False),dc.distance_covariance(self.x,self.y))
        self.est.set_params(mode='mdd')
        self.assertAlmostEquals(self.est.fit(self.x,y=self.y,biascorr=False),0.352427150086)
        
        
        
        
        
        

        
        
        

        
        
        
        
        
        
        
    
    
if __name__ =='__main__':
    unittest.main()
        
    
