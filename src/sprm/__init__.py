#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jul 22 12:17:17 2018

@author: Sven Serneels, Ponalytics
"""

__name__ = "sprm"
__author__ = "Sven Serneels"
__license__ = "MIT"
__version__ = "0.7.0"
__date__ = "2020-04-03"

from .preprocessing.robcent import VersatileScaler, versatile_scale
from .preprocessing.gsspp import GenSpatialSignPrePprocessor, gen_ss_pp, gen_ss_covmat
from .sprm.sprm import sprm
from .sprm.snipls import snipls
from .sprm.rm import rm
from .cross_validation._cv_support_functions import robust_loss
from .plot.sprm_plot import sprm_plot,sprm_plot_cv





