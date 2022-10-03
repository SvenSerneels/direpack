"""
Created on Fri Jan 25 18:22:27 2019

Functions called internally in M-estimation 

@author: Sven Serneels, Ponalytics
"""

import numpy as np
import pandas as ps


def Fair(x, probct, *args):
    return 1 / (1 + abs(x / (probct * 2))) ** 2


def Huber(x, probct, *args):
    x[np.where(x <= probct)[0]] = 1
    x[np.where(x > probct)] = probct / abs(x[np.where(x > probct)])
    return x


def Hampel(x, probct, hampelb, hampelr):
    wx = x
    wx[np.where(x <= probct)[0]] = 1
    wx[np.where((x > probct) & (x <= hampelb))[0]] = probct / abs(
        x[np.where((x > probct) & (x <= hampelb))[0]]
    )
    wx[np.where((x > hampelb) & (x <= hampelr))[0]] = np.divide(
        probct * (hampelr - (x[np.where((x > hampelb) & (x <= hampelr))[0]])),
        (hampelr - hampelb) * abs(x[np.where((x > hampelb) & (x <= hampelr))[0]]),
    )
    wx[np.where(x > hampelr)[0]] = 0
    return wx


def brokenstick(n_components):
    q = np.triu(np.ones((n_components, n_components)))
    r = np.empty((n_components, 1), float)
    r[0:n_components, 0] = range(1, n_components + 1)
    q = np.matmul(q, 1 / r)
    q /= n_components
    return q
