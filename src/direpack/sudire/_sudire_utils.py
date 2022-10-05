# -*- coding: utf-8 -*-
"""
Created on Sat Apr 11 17:00:13 2020

@author: Emmanuel Jordy Menvouta
"""


import numpy as np
from ..dicomo._dicomo_utils import *
import scipy.spatial as spp
import pandas as pd
import sympy
from ..utils.utils import MyException


# import Ball
# uncomment this line if you want to use


def mdd_trim(beta, *args):
    """
    This is the objective function to be optimised when using MDD-SDR
    args is expected to be a tuple With the following elements : 
        X : input Data as matrix
        Y : input Data as 1d matrix 
        h, int: Dimension of the central subspace
        N2 : inverse of square root of the covariance matrix of X 
        is_distance_matrix, bool : if true, X and Y represent distance matrices
        trimming, float : trimming fraction to be applied for MDD
        center, bool :  if true, X and Y will be centered when computing MDD
        dmetric : 'the distance metric to be used in the computation of MDD
        biascorr, bool : if True, an unbiased estimator of MDD is computed.
    """
    X = args[0]
    Y = args[1]
    h = args[2]
    N2 = args[3]
    is_data_matrix = args[4]
    trimming = args[5]
    center = args[6]
    dmetric = args[7]
    biascorr = args[8]
    beta = np.reshape(beta, (-1, h), order="F")
    if is_data_matrix:
        X_coord = give_coords(X)
        # Y_coord = give_coords(Y)
        X_dat = np.matmul(X_coord, beta)
        res = np.sqrt(
            difference_divergence(
                X_dat, Y, center=center, trimming=trimming, biascorr=biascorr
            )
        )
        return -10 * res
    else:
        X_dat = np.matmul(X, beta)
        res = np.sqrt(
            difference_divergence(
                X_dat, Y, center=center, trimming=trimming, biascorr=biascorr
            )
        )
        return -10 * res


def dcov_trim(beta, *args):
    """
    This is the objective function to be optimised when using DCOV-SDR
    args is expected to be a tuple With the following elements : 
        X : input Data as matrix
        Y : input Data as 1d matrix 
        h : Dimension of the central subspace
        N2 : inverse of square root of the covariance matrix of X 
        is_distance_matrix, bool : if true, X and Y represent distance matrices
        center, bool :  if true, X and Y will be centered when computing MDD
        dmetric : 'the distance metric to be used in the computation of MDD
        biascorr, bool : if True, an unbiased estimator of MDD is computed.
    """
    X = args[0]
    Y = args[1]
    h = args[2]
    N2 = args[3]
    is_distance_matrix = args[4]
    trimming = args[5]
    center = args[6]
    dmetric = args[7]
    beta = np.reshape(beta, (-1, h), order="F")
    if is_distance_matrix:
        X_coord = give_coords(X)
        Y_coord = give_coords(Y)
        X_dat = np.matmul(X_coord, beta)
        dmx, n1 = distance_matrix_centered(
            X_dat, trimming=trimming, center=center, dmetric=dmetric
        )
        dmy, n2 = distance_matrix_centered(
            Y_coord, trimming=trimming, center=center, dmetric=dmetric
        )
        res = distance_moment(
            dmx, dmy, n1=n1, center=center, trimming=trimming, order=2
        )
        return -10 * res
    else:
        Y = Y.reshape(-1, 1)
        X_dat = np.matmul(X, beta)
        dmx, n1 = distance_matrix_centered(
            X_dat, trimming=trimming, center=center, dmetric=dmetric
        )
        dmy, n2 = distance_matrix_centered(
            Y, trimming=trimming, center=center, dmetric=dmetric
        )
        res = distance_moment(
            dmx, dmy, n1=n1, center=center, trimming=trimming, order=2
        )
        return -10 * res


def give_coords(distances):
    """give coordinates of points for which distances given

    coordinates are given relatively. 1st point on origin, 2nd on x-axis, 3rd 
    x-y plane and so on. Maximum n-1 dimentions for which n is the number
    of points
     Args:
        distanes (list): is a n x n, 2d array where distances[i][j] gives the distance 
            from i to j assumed distances[i][j] == distances[j][i]

     Returns:
        numpy.ndarray: cordinates in list form n dim
    """
    distances = np.array(distances)

    n = len(distances)
    X = sympy.symarray("x", (n, n - 1))

    for row in range(n):
        X[row, row:] = [0] * (n - 1 - row)

    for point2 in range(1, n):

        expressions = []

        for point1 in range(point2):
            expression = np.sum((X[point1] - X[point2]) ** 2)
            expression -= distances[point1, point2] ** 2
            expressions.append(expression)

        X[point2, :point2] = sympy.solve(expressions, list(X[point2, :point2]))[1]

    return X


def matpower(A, alpha):
    """
    computes the A to the power alpha using the eigen decomposition of A.
    """
    A = (A + A.T) / 2
    eig_vals, eig_vecs = np.linalg.eigh(A)
    res = np.matmul(np.matmul(eig_vecs, np.diag(eig_vals ** alpha)), eig_vecs.T)
    return res


def discretize(y, h):
    """
    This function is used to discretize a continuous variable y into h classes
    that are ordered.
    input :
        y : input data as vector or 1D matrix.
        h : the number of classes 
        
    output : 
         vector containing the class to which each element of y belongs. 
    """
    n = len(y)
    m = np.floor(n / h)
    yord = np.sort(y)
    divpt = []
    for i in range(1, h):
        divpt.append(yord[int((i * m))])
    y1 = np.repeat(0, n)
    y1[y < divpt[0]] = 1
    y1[y >= divpt[h - 2]] = h
    for i in range(1, h - 1):
        y1[(y >= divpt[i - 1]) & (y < divpt[i])] = i + 1
    return y1


def SIR(x, y, n_slices, d, ytype="continuous", center_data=True, scale_data=True):
    """
    computes  the Sliced Inverse Regression  estimator of the central subspace.
    The algortihm is inspired from :
    "Bing, Li .(2018). Sufficient Dimension Reduction: Methods and Applications with R.
    Chapman & Hall/CRC Monographs on Statistics and Applied Probability."
    
    Input : 
        X : input Data as Matrix 
        Y : input data as vector or 1d matrix.
        n_slices, int : number of slices 
        d, int : dimension of the central subspace.
        ytype, str : either discrete or continuous
        center_data, bool, if true, the data is centered before SIR
        scale_data, bool, if true, the data is sclaed before SIR
        
    Output :
         The estimated basis of the central subspace.
        
    """
    y = np.asarray(y).flatten()
    n = x.shape[0]  ## zxception if d> p or x.shape[0] != y.shape[0]
    signsqrt = matpower(np.cov(x, rowvar=0), -0.5)
    #     xc = x - x.mean(axis=0)
    #     xstd= np.matmul(xc,signsqrt)

    if center_data:
        x = x - x.mean(axis=0)
    if scale_data:
        x = np.matmul(x, signsqrt)
    xstd = x
    if ytype == "continuous":
        ydis = discretize(y, n_slices)
    else:
        ydis = y
    yless = ydis
    ylabel = []
    for i in range(n):
        if np.var(yless) != 0:
            ylabel.append(yless[0])
            yless = yless[yless != yless[0]]
    ylabel.append(yless[0])
    prob = []
    exy = []

    for i in range(n_slices):
        prob.append(len(ydis[ydis == ylabel[i]]) / n)
        xres = np.apply_along_axis(np.mean, 0, xstd[ydis == ylabel[i], :])
        exy.append(xres)

    exy = np.vstack(exy)
    sirmat = np.matmul(np.matmul(exy.T, np.diag(np.array(prob))), exy)
    eig_vals, eig_vecs = np.linalg.eigh(sirmat)
    idx = eig_vals.argsort()[::-1]
    eig_vals = eig_vals[idx]
    eig_vecs = eig_vecs[:, idx]
    if scale_data:
        return np.matmul(signsqrt, eig_vecs[:, 0:d])
    else:
        return eig_vecs[:, 0:d]


def SAVE(x, y, n_slices, d, ytype="continuous", center_data=True, scale_data=True):
    """
    computes  the Sliced Average Variance Estimator of the central subspace.
    The algortihm is inspired from :
    "Bing, Li .(2018). Sufficient Dimension Reduction: Methods and Applications with R.
    Chapman & Hall/CRC Monographs on Statistics and Applied Probability."
    
    Input : 
        X : input Data as Matrix 
        Y : input data as vector or 1d matrix.
        n_slices, int : number of slices 
        d, int : dimension of the central subspace.
        ytype, str : either discrete or continuous
        center_data, bool, if true, the data is centered before SIR
        scale_data, bool, if true, the data is sclaed before SIR
        
    Output :
         The estimated basis of the central subspace.
        
    """
    y = np.asarray(y).flatten()
    p = x.shape[1]
    n = x.shape[0]  ## zxception if d> p or x.shape[0] != y.shape[0]
    signsqrt = matpower(np.cov(x, rowvar=0), -0.5)
    #     xc = x - x.mean(axis=0)
    #     xstd= np.matmul(xc,signsqrt)
    if center_data:
        x = x - x.mean(axis=0)
    if scale_data:
        x = np.matmul(x, signsqrt)

    xstd = x

    if ytype == "continuous":
        ydis = discretize(y, n_slices)
    else:
        ydis = y
    yless = ydis
    ylabel = []
    for i in range(n):
        if np.var(yless) != 0:
            ylabel.append(yless[0])
            yless = yless[yless != yless[0]]
    ylabel.append(yless[0])
    prob = []
    for i in range(n_slices):
        prob.append(len(ydis[ydis == ylabel[i]]) / n)

    vxy = np.zeros((n_slices, p, p))
    for i in range(n_slices):
        vxy[i, :, :] = np.cov(xstd[ydis == ylabel[i], :], rowvar=0)

    savemat = np.zeros((p, p))
    for i in range(n_slices):
        savemat = savemat + prob[i] * np.matmul(
            (vxy[i, :, :] - np.identity(p)), (vxy[i, :, :] - np.identity(p))
        )

    eig_vals, eig_vecs = np.linalg.eigh(savemat)
    idx = eig_vals.argsort()[::-1]
    eig_vals = eig_vals[idx]
    eig_vecs = eig_vecs[:, idx]
    if scale_data:
        return np.matmul(signsqrt, eig_vecs[:, 0:d])
    else:
        return eig_vecs[:, 0:d]


def DR(x, y, n_slices, d, ytype="continuous", center_data=True, scale_data=True):
    """
    computes  the Directional Regression Estimator of the central subspace.
    The algortihm is inspired from :
    "Bing, Li .(2018). Sufficient Dimension Reduction: Methods and Applications with R.
    Chapman & Hall/CRC Monographs on Statistics and Applied Probability."
    
    Input : 
        X : input Data as Matrix 
        Y : input data as vector or 1d matrix.
        n_slices, int : number of slices 
        d, int : dimension of the central subspace.
        ytype, str : either discrete or continuous
        center_data, bool, if true, the data is centered before SIR
        scale_data, bool, if true, the data is sclaed before SIR
        
    Output :
         The estimated basis of the central subspace.
        
    """
    y = np.asarray(y).flatten()
    p = x.shape[1]
    n = x.shape[0]  ## zxception if d> p or x.shape[0] != y.shape[0]
    signsqrt = matpower(np.cov(x, rowvar=0), -0.5)

    if center_data:
        x = x - x.mean(axis=0)
    if scale_data:
        x = np.matmul(x, signsqrt)

    xstd = x
    if ytype == "continuous":
        ydis = discretize(y, n_slices)
    else:
        ydis = y
    yless = ydis
    ylabel = []
    for i in range(n):
        if np.var(yless) != 0:
            ylabel.append(yless[0])
            yless = yless[yless != yless[0]]
    ylabel.append(yless[0])
    prob = []
    exy = []

    for i in range(n_slices):
        prob.append(len(ydis[ydis == ylabel[i]]) / n)

    vxy = np.zeros((n_slices, p, p,))
    exy = []
    for i in range(n_slices):
        vxy[i, :, :,] = np.cov(xstd[ydis == ylabel[i], :], rowvar=0)
        xres = np.apply_along_axis(np.mean, 0, xstd[ydis == ylabel[i], :])
        exy.append(xres)

    exy = np.vstack(exy)
    mat1 = np.zeros((p, p))
    mat2 = np.zeros((p, p))

    for i in range(n_slices):
        mat1 = mat1 + prob[i] * np.matmul(
            (
                vxy[i, :, :]
                + np.matmul(exy[i, :].reshape((-1, 1)), exy[i, :].reshape((-1, 1)).T)
            ),
            (
                vxy[i, :, :]
                + np.matmul(exy[i, :].reshape((-1, 1)), exy[i, :].reshape((-1, 1)).T)
            ),
        )
        mat2 = mat2 + prob[i] * (
            np.matmul(exy[i, :].reshape((-1, 1)), exy[i, :].reshape((-1, 1)).T)
        )

    out = (
        2 * mat1
        + 2 * np.matmul(mat2, mat2)
        + 2 * np.sum(np.diag(mat2)) * mat2
        - 2 * np.identity(p)
    )
    eig_vals, eig_vecs = np.linalg.eigh(out)
    idx = eig_vals.argsort()[::-1]
    eig_vals = eig_vals[idx]
    eig_vecs = eig_vecs[:, idx]
    if scale_data:
        return np.matmul(signsqrt, eig_vecs[:, 0:d])
    else:
        return eig_vecs[:, 0:d]


def PHD(x, y, d, center_data=True, scale_data=True):

    """
    computes  the Principal Hessian Dimension estimator of the central subspace.
    The algortihm is inspired from :
    "Bing, Li .(2018). Sufficient Dimension Reduction: Methods and Applications with R.
    Chapman & Hall/CRC Monographs on Statistics and Applied Probability."
    
    Input : 
        X : input Data as Matrix 
        Y : input data as vector or 1d matrix.
        d, int : dimension of the central subspace.
        center_data, bool, if true, the data is centered before SIR
        scale_data, bool, if true, the data is sclaed before SIR
        
    Output :
         The estimated basis of the central subspace.
    
    """
    n = x.shape[0]
    if len(y.shape) < 2:
        y = y.reshape((-1, 1))
    signsqrt = matpower(np.cov(x, rowvar=0), -0.5)
    if center_data:
        x = x - x.mean(axis=0)
    if scale_data:
        x = np.matmul(x, signsqrt)

    yc = y - np.mean(y, axis=0)
    out = np.matmul(np.multiply(x, yc).T, x) / n
    eig_vals, eig_vecs = np.linalg.eigh(out)
    idx = eig_vals.argsort()[::-1]
    eig_vals = eig_vals[idx]
    eig_vecs = eig_vecs[:, idx]

    if scale_data:
        return np.matmul(signsqrt, eig_vecs[:, 0:d])
    else:
        return eig_vecs[:, 0:d]


def IHT(x, y, d, center_data=True, scale_data=True):
    """
    computes  the Iterative Hessian Transformation estimator of the central subspace.
    The algortihm is inspired from :
    "Bing, Li .(2018). Sufficient Dimension Reduction: Methods and Applications with R.
    Chapman & Hall/CRC Monographs on Statistics and Applied Probability."
    
    Input : 
        X : input Data as Matrix 
        Y : input data as vector or 1d matrix.
        d, int : dimension of the central subspace.
        center_data, bool, if true, the data is centered before SIR
        scale_data, bool, if true, the data is sclaed before SIR
        
    Output :
         The estimated basis of the central subspace.
        
    """

    p = x.shape[1]
    ## zxception if d> p or x.shape[0] != y.shape[0]
    if len(y.shape) < 2:
        y = y.reshape((-1, 1))
    signsqrt = matpower(np.cov(x, rowvar=0), -0.5)

    if center_data:
        x = x - x.mean(axis=0)
    if scale_data:
        x = np.matmul(x, signsqrt)

    covxy = np.cov(x, y, rowvar=0)[-1, :-1]
    covx = np.cov(x, rowvar=0)
    mat = covxy
    for i in range(1, p):
        mat = np.column_stack((mat, np.matmul(matpower(covx, i), covxy)))

    out = np.matmul(mat, mat.T)
    eig_vals, eig_vecs = np.linalg.eigh(out)
    idx = eig_vals.argsort()[::-1]
    eig_vals = eig_vals[idx]
    eig_vecs = eig_vecs[:, idx]
    if scale_data:
        return np.matmul(signsqrt, eig_vecs[:, 0:d])
    else:
        return eig_vecs[:, 0:d]


# def ballcov_func(beta, *args):
#     """
#     Objective function  for BCOV-SDR.
#     Will only work after uncommenting the import Ball statement above
#     """

#     X= args[0]
#     Y= args[1]
#     h=args[2]
#     beta = np.reshape(beta,(-1,h),order = 'F')
#     X_dat = np.matmul(X, beta)
#     res = Ball.bcov_test(X_dat,Y,num_permutations=0)[0]
#     return(-10*res)
