# -*- coding: utf-8 -*-
"""
Created on Tue Jun 30 13:17:46 2020

@author: Emmanuel Jordy Menvouta
"""

import unittest
import numpy as np
import pandas as pd
from direpack.sudire._sudire_utils import *
from direpack import sudire
from sklearn.model_selection import train_test_split


class Testsudire(unittest.TestCase):
    """Test some methods in the sudire class"""

    @classmethod
    def setUpClass(cls):
        print("setupClass")

    @classmethod
    def tearDownClass(cls):
        print("teardownClass")

    def setUp(self):
        self.data = pd.read_csv("./data/boston_housing.csv")
        self.x = self.data
        self.y = self.x["MEDV"]
        self.x.drop("MEDV", axis=1, inplace=True)
        self.n = self.x.shape[0]
        self.p = self.x.shape[1]
        self.struct_dim = 2

    def tearDown(self):
        del self.x
        del self.y
        del self.n
        del self.p
        del self.data
        del self.struct_dim

    #    def test_estimdim(self):
    #        """ Tests the estimation of the central subspace via Bootstrap """
    #
    #        central_dim, diff_vec = estimate_structural_dim('dr',self.x_train.values,self.y_train.values , B=100, n_slices=4)
    #        np.testing.assert_equal(central_dim,6)

    def test_sir(self):
        """Tests Sliced Inverse Regression"""

        # mod_auto = sudire('sir', center_data= True, scale_data=True,n_components=self.struct_dim)
        # mod_auto.fit(self.x_train.values, self.y_train.values)
        res_sir = SIR(
            self.x.values,
            self.y.values,
            6,
            self.struct_dim,
            "continuous",
            True,
            True,
        )
        test_ans = 3.9759408796493894
        np.testing.assert_almost_equal(
            np.linalg.norm(res_sir), test_ans, decimal=8
        )

    def test_save(self):
        """Tests Sliced Average Variance Estimation"""

        # mod_auto = sudire('save', center_data= True, scale_data=True,n_components=self.struct_dim)
        # mod_auto.fit(self.x_train.values, self.y_train.values)
        res_save = SAVE(
            self.x.values,
            self.y.values,
            6,
            self.struct_dim,
            "continuous",
            True,
            True,
        )
        test_ans = 6.347895320407837
        np.testing.assert_almost_equal(
            np.linalg.norm(res_save), test_ans, decimal=8
        )

    def test_dr(self):
        """Tests Directional Regression"""

        # mod_auto = sudire('dr', center_data= True, scale_data=True,n_components=self.struct_dim)
        # mod_auto.fit(self.x_train.values, self.y_train.values)
        res_dr = DR(
            self.x.values,
            self.y.values,
            6,
            self.struct_dim,
            "continuous",
            True,
            True,
        )
        test_ans = 4.013789664544885
        np.testing.assert_almost_equal(
            np.linalg.norm(res_dr), test_ans, decimal=8
        )

    def test_iht(self):
        """Tests Iterative Hessian Transformations"""

        # mod_auto = sudire('iht', center_data= True, scale_data=True,n_components=self.struct_dim)
        # mod_auto.fit(self.x_train.values, self.y_train.values)
        res_iht = IHT(
            self.x.values, self.y.values, self.struct_dim, True, True
        )
        # local linux -- resolve platform sensitivity!!
        # test_ans = 0.22443355
        test_ans = 1.68656340
        np.testing.assert_almost_equal(
            np.linalg.norm(res_iht), test_ans, decimal=8
        )

    def test_phd(self):
        """Tests Principal Hessian Directions"""

        # mod_auto = sudire('phd', center_data= True, scale_data=True,n_components=self.struct_dim)
        # mod_auto.fit(self.x_train.values, self.y_train.values)
        res_phd = PHD(
            self.x.values, self.y.values, self.struct_dim, True, True
        )
        test_ans = 3.2904239864118763
        np.testing.assert_almost_equal(
            np.linalg.norm(res_phd), test_ans, decimal=8
        )

    def test_dcov(self):
        """Test DCOV based SDR"""

        mod_auto = sudire(
            "dcov-sdr",
            center_data=True,
            scale_data=True,
            n_components=self.struct_dim,
        )
        mod_auto.fit(self.x.values, self.y.values)
        test_ans = 1.4628980331787338
        np.testing.assert_almost_equal(
            np.linalg.norm(mod_auto.x_loadings_), test_ans, decimal=5
        )

    def test_mdd(self):

        """Test MDD based SDR"""
        mod_auto = sudire(
            "mdd-sdr",
            center_data=True,
            scale_data=True,
            n_components=self.struct_dim,
        )
        mod_auto.fit(self.x.values, self.y.values)
        test_ans = 3.5752717342726803
        np.testing.assert_almost_equal(
            np.linalg.norm(mod_auto.x_loadings_), test_ans, decimal=5
        )


if __name__ == "__main__":
    unittest.main()
