#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jan 27 11:41:35 2019

SPRM Plotting options (based on Matplotlib)
---------------------

In class sprm_plot
    Y vs Y predicted plot with outlier flagging
    Projection to score space with outlier flagging 
    Barchart of regression coefficients with range selection 
    Barchart of caseweights 
        All of these plots work both with or without new test datya beoing s
        supplied and have the optoion to only plot the test data.
        
In class sprm_plot_cv
    3D contour for SPRM Grid Search CV results 
        Requires an sklearn GridSearchCV object that trained an SPRM model 
        Provides a 3d contour plot across the (eta, n_components) space

Version 0.2: Ancillary functions: have been moved to ._plot_internals

@author: Sven Serneels, Ponalytics. 
"""

from __future__ import absolute_import, division, print_function
from __future__ import unicode_literals

from ..sprm.sprm import sprm
from ..utils.utils import MyException, convert_X_input, convert_y_input
from ..cross_validation._cv_support_functions import cv_score_table

import matplotlib.pyplot as pp 
import numpy as np
from sklearn.model_selection import GridSearchCV

class sprm_plot(sprm):
    
    def __init__(self,res_sprm,colors,markers=['o','d','v'],*args):
        """
        Initialize with 
        res_sprm, an sprm class object
        
        Only mandatory input is colors, a list of colors for 
            [0] borders of pane 
            [1] plot background
            [2] marker fill
            [3] diagonal line 
            [4] marker contour, if different from fill
            [5] marker color for new cases, if applicable
            [6] marker color for harsh calibration outliers
            [7] marker color for harsh prediction outliers
        
        Optional input markers, a list:
            [0] marker for regular cases
            [1] marker for moderate outliers (caseweight in (0,1))
            [2] marker for harsh outliers (caseweight = 0)
        
        """
        if not(isinstance(res_sprm,sprm)):
            raise(MyException("Object supplied to sprmplot needs to be an sprm object"))
        self.res_sprm = res_sprm
        self.colors = colors
        self.markers = markers

    def plot_yyp(self,ytruev=[],Xn=[],label=[],names=[],namesv=[],title=[],legend_pos='lower right',onlyval=False):
        """
        plot_yyp will plot y vs y predicted for SPRM opbjects
        Optional inputs: 
            ytruev: array (new_cases,) of predictands
            Xn: array (new_cases,variables) of predictors 
            If these arguments are supplied, SPRM predictions for ytrue will be 
                made from Xn through res_sprm.predict()
            label: string: name of variable to be plotted. Will show in legend.
            names: list or tuple of strings, casenames from training set
            namesv: list or tuple of strings, casenames from test set
            title: String containing plot title
            legend_pos: string containing legend position
            onlyval: boolean: only plot validation cases
        """
        
        if len(label)==0:
            label = 'none'
        fig = pp.figure()
        fig.set_facecolor(self.colors[0])
        pp.rcParams['axes.facecolor'] = self.colors[1]
        ax1 = fig.add_subplot(111)
        if (not(onlyval)):
            ytruec = self.res_sprm.y
            if len(ytruec.shape) >1:
                ytruec = np.array(ytruec).reshape(-1).astype('float64')
            ypredc = np.array(self.res_sprm.fitted_).T.reshape(-1)
            labelcr = label[0] + ' Training' + ' Regular'
            labelcm = label[0] + ' Training' + ' Moderate'
            labelch = label[0] + ' Training' + ' Harsh'
            reg_cases = np.where(self.res_sprm.caseweights_ == 1)
            mod_outliers = np.where((self.res_sprm.caseweights_ > 0) & (self.res_sprm.caseweights_ < 1))
            harsh_outliers = np.where(self.res_sprm.caseweights_ == 0)
            ax1.scatter(ytruec[reg_cases], ypredc[reg_cases], c=self.colors[2],label=labelcr, 
                    zorder=1,edgecolors=self.colors[4],marker=self.markers[0])
            if len(mod_outliers[0]>0):
                ax1.scatter(ytruec[mod_outliers], ypredc[mod_outliers], c=self.colors[2],label=labelcm, 
                    zorder=1,edgecolors=self.colors[4],marker=self.markers[1])
            if len(harsh_outliers[0]>0):
                ax1.scatter(ytruec[harsh_outliers], ypredc[harsh_outliers], c=self.colors[6],label=labelch, 
                    zorder=1,edgecolors=self.colors[4],marker=self.markers[2])
        else:
            if (len(Xn)==0):
                ValueError('In onlyval=True mode, new cases Xn need to be provided')
        if not(len(Xn)==0):
            if len(ytruev.shape) >1:
                ytruev = np.array(ytruev).reshape(-1).astype('float64')
            ypredv = self.res_sprm.predict(Xn)
            ypredv = np.array(ypredv).reshape(-1).astype('float64')
            wv = self.res_sprm.weightnewx(Xn)
            labelvr = label[0] + ' Test' + ' Regular'
            labelvm = label[0] + ' Test' + ' Moderate'
            labelvh = label[0] + ' Test' + ' Harsh'
            reg_cases = np.where(wv == 1)[0]
            mod_outliers = np.where((wv > 0) & (wv < 1))[0]
            harsh_outliers = np.where(wv == 0)[0]
            if len(reg_cases>0):
                ax1.scatter(ytruev[reg_cases],ypredv[reg_cases],c=self.colors[5],label=labelvr,
                        zorder=1,edgecolors=self.colors[4],marker=self.markers[0])
            if len(mod_outliers>0):
                ax1.scatter(ytruev[mod_outliers],ypredv[mod_outliers],c=self.colors[5],label=labelvm,
                        zorder=1,edgecolors=self.colors[4],marker=self.markers[1])
            if len(harsh_outliers>0):
                ax1.scatter(ytruev[harsh_outliers],ypredv[harsh_outliers],c=self.colors[7],label=labelvh,
                        zorder=1,edgecolors=self.colors[4],marker=self.markers[2])
        x_abline = np.array(ax1.get_xbound())
        ax1.add_line(pp.Line2D(x_abline,x_abline,color=self.colors[3]))
        if len(label)==0:
            ax1.legend_.remove()
        else:
            pp.legend(loc=legend_pos)
        if len(names)>0:
            if not(onlyval):
                for i in range(0,len(names)-1):
                    ax1.annotate(names[i], (ytruec[i],ypredc[i]))
        if len(namesv)>0:
            for i in range(0,len(namesv)-1):
                ax1.annotate(namesv[i], (ytruev[i],ypredv[i]))
        if len(title)>0:
            pp.title(title)
        pp.show()
        
    def plot_coeffs(self,entity="coef_",truncation=0,columns=[],title=[]):
        """
        plot_coeffs will plot estimated model parameters, with option to 
        truncate (useful if highly multivariate) 
        Optional Inputs: 
            entity, str, exact name of attribute from sprm object to be plotted
            truncation, float in [0,1), Percentage of smallest and largest 
                coefficients to be plotted
            columns, array, tuple or list of str, variable names
            title, str, plot title
        """
        fig = pp.figure()
        fig.set_facecolor(self.colors[0])
        pp.rcParams['axes.facecolor'] = self.colors[1]
        ax1 = fig.add_subplot(111)
        p = len(self.res_sprm.non_zero_scale_vars_)
        if len(columns) > 0:
            x_plot = columns[self.res_sprm.non_zero_scale_vars_]
            x_labels = columns[self.res_sprm.non_zero_scale_vars_]
        else:
            x_plot = np.arange(0,p)
       
        b_orig = np.array(getattr(self.res_sprm,entity)).reshape(-1)
        if truncation > 0:
            ind_sort = np.argsort(np.argsort(b_orig))
            b_sort = np.sort(b_orig)
            left = np.ceil(p*(truncation/2)).astype(int)
            right = (p-np.ceil(p*(truncation/2)).astype(int))
            b_plot = b_sort[np.union1d(np.arange(0,left),
                                       np.arange(right,p))]
            x_labels = np.union1d(x_plot[ind_sort[0:left]],
                                x_plot[ind_sort[right:p]])
            x_plot = np.arange(0,len(x_labels))
            ax1.bar(x_plot,b_plot,color=self.colors[2],edgecolor=self.colors[4],tick_label=x_labels)
        else:
            b_plot = b_orig 
            ax1.bar(x_plot,b_plot,color=self.colors[2],edgecolor=self.colors[4])  
        if ((truncation > 0) | len(columns) > 0): 
            pp.xticks(x_plot, x_labels, rotation='vertical')
            pp.margins(.4)
        if len(title)>0:
            pp.title(title)
        pp.show()
        
        
    def plot_projections(self,Xn=[],label=[],components = [0,1],names=[],namesv=[],title=[],legend_pos='lower right',onlyval=False):
        
        """
        plot_projections will plot the score space  
        Optional inputs: 
            Xn: array (new_cases,variables) of predictors 
            If supplied, SPRM projections for new cases will be 
                made from Xn through res_sprm.transform()
            label: string: name of variable to be plotted. Will show in legend.
            names: list or tuple of strings, casenames from training set
            namesv: list or tuple of strings, casenames from test set
            title: String containing plot title
            legend_pos: string containing legend position
            onlyval: boolean: only plot validation cases
        """
        
        if len(label)==0:
            label = 'none'
        fig = pp.figure()
        fig.set_facecolor(self.colors[0])
        pp.rcParams['axes.facecolor'] = self.colors[1]
        ax1 = fig.add_subplot(111)
        if (not(onlyval)):
            Tc = np.array(self.res_sprm.x_scores_)
            labelcr = label[0] + ' Training' + ' Regular'
            labelcm = label[0] + ' Training' + ' Moderate'
            labelch = label[0] + ' Training' + ' Harsh'
            reg_cases = np.where(self.res_sprm.caseweights_ == 1)
            mod_outliers = np.where((self.res_sprm.caseweights_ > 0) & (self.res_sprm.caseweights_ < 1))
            harsh_outliers = np.where(self.res_sprm.caseweights_ == 0)
            ax1.scatter(Tc[reg_cases,components[0]], Tc[reg_cases,components[1]], c=self.colors[2],label=labelcr, 
                    zorder=1,edgecolors=self.colors[4],marker=self.markers[0])
            if len(mod_outliers[0])>0:
                ax1.scatter(Tc[mod_outliers,components[0]], Tc[mod_outliers,components[1]], c=self.colors[2],label=labelcm, 
                    zorder=1,edgecolors=self.colors[4],marker=self.markers[1])
            if len(harsh_outliers[0])>0:
                ax1.scatter(Tc[harsh_outliers,components[0]], Tc[harsh_outliers,components[1]], c=self.colors[6],label=labelch, 
                    zorder=1,edgecolors=self.colors[4],marker=self.markers[2])
        else:
            if (len(Xn)==0):
                ValueError('In onlyval=True mode, new cases Xn need to be provided')
        if not(len(Xn)==0):
            Tv = np.array(self.res_sprm.transform(Xn))
            labelvr = label[0] + ' Test' + ' Regular'
            labelvm = label[0] + ' Test' + ' Moderate'
            labelvh = label[0] + ' Test' + ' Harsh'
            wv = self.res_sprm.weightnewx(Xn)
            reg_cases = np.where(wv == 1)
            mod_outliers = np.where((wv > 0) & (wv < 1))
            harsh_outliers = np.where(wv == 0)
            if len(reg_cases[0]>0):
                ax1.scatter(Tv[reg_cases,components[0]], Tv[reg_cases,components[1]],c=self.colors[5],label=labelvr,
                        zorder=1,edgecolors=self.colors[4],marker=self.markers[0])
            if len(mod_outliers[0])>0:
                ax1.scatter(Tv[mod_outliers,components[0]], Tv[mod_outliers,components[1]],c=self.colors[5],label=labelvm,
                        zorder=1,edgecolors=self.colors[4],marker=self.markers[1])
            if len(harsh_outliers[0])>0:
                ax1.scatter(Tv[harsh_outliers,components[0]], Tv[harsh_outliers,components[1]],c=self.colors[7],label=labelvh,
                        zorder=1,edgecolors=self.colors[4],marker=self.markers[2])
        if len(label)==0:
            ax1.legend_.remove()
        else:
            pp.legend(loc=legend_pos)
        if len(names)>0:
            if not(onlyval):
                for i in range(0,len(names)-1):
                    ax1.annotate(names[i], (Tc[i,components[0]], Tc[i,components[1]]))
        if len(namesv)>0:
            for i in range(0,len(namesv)-1):
                ax1.annotate(namesv[i], (Tv[i,components[0]], Tv[i,components[1]]))
        if len(title)>0:
            pp.title(title)
        pp.show()
        
    def plot_caseweights(self,Xn=[],label=[],names=[],namesv=[],title=[],legend_pos='lower right',onlyval=False,mode='overall'):
        
        """
        plot_caseweights will plot caseweights
        Optional inputs: 
            Xn: array (new_cases,variables) of predictors 
            If supplied, SPRM projections for new cases will be 
                made from Xn through res_sprm.weightnewx()
            label: string: name of variable to be plotted. Will show in legend.
            names: list or tuple of strings, casenames from training set
            namesv: list or tuple of strings, casenames from test set
            title: String containing plot title
            legend_pos: string containing legend position
            onlyval: boolean: only plot validation cases
            mode: str, which weights to plot for cases from training set, 
                - 'overall': combined caseweights
                - 'x': predictor block 
                - 'y': predictand block
                Since for validation cases y is unknown, 'x' caseweights are 
                plotted by default there. 
        """
        
        if len(label)==0:
            label = 'none'
        fig = pp.figure()
        fig.set_facecolor(self.colors[0])
        pp.rcParams['axes.facecolor'] = self.colors[1]
        ax1 = fig.add_subplot(111)
        if (not(onlyval)):
            if mode=='overall':
                wc = self.res_sprm.caseweights_
            elif mode == 'x':
                wc = self.res_sprm.x_caseweights_
            elif mode == 'y':
                wc = self.res_sprm.y_caseweights_
            else:
                ValueError('Options for mode are overall, x or y')
            labelc = label[0] + ' Training' 
        else:
            wc = []
            labelc=[]
            if (len(Xn)==0):
                ValueError('In onlyval=True mode, new cases Xn need to be provided')
        if not(len(Xn)==0):
            wv = self.res_sprm.weightnewx(Xn)
            labelv = label[0] + ' Test'
        else:
            wv=[]
            labelv = []
        name_indices = np.array(range(1,len(wc)+len(wv)+1)) 
        if (len(wc)>0):
            ax1.bar(name_indices,np.concatenate((wc,np.repeat(np.nan,len(wv)))),color=self.colors[2],label=labelc)
        if (len(wv)>0):
            ax1.bar(name_indices,np.concatenate((np.repeat(np.nan,len(wc)),wv)),color=self.colors[5],label=labelv)
        if len(label)==0:
            ax1.legend_.remove()
        else:
            pp.legend(loc=legend_pos)
        if len(names)>0:
            if not(onlyval):
                for i in range(0,len(names)-1):
                    ax1.annotate(names[i], (name_indices[i+1],wc[i]))
        if len(namesv)>0:
            for i in range(0,len(namesv)-1):
                ax1.annotate(namesv[i], (name_indices[len(wc)+i],wv[i]))
        if len(title)>0:
            pp.title(title)
        pp.show()
        


class sprm_plot_cv(GridSearchCV,sprm):
    
    def __init__(self,res_sprm_cv,colors,*args):
        
        """
        Initialize with 
        res_sprm_cv, an GridSearchCV cross-validated sprm object 
        
        Only mandatory input is colors, a list of colors for 
            [0] borders of pane 
            [1] plot background
            [2] marker fill
            [3] diagonal line 
            [4] marker contour, if different from fill
            [5] marker color for new cases, if applicable
            [6] marker color for harsh calibration outliers
            [7] marker color for harsh prediction outliers
        
        """
        
        self.res_sprm_cv = res_sprm_cv
        self.colors = colors
        
    def eta_ncomp_contour(self,title='SPRM Cross-Validation Contour Plot'):
        
        """
        Function to draw contour plot from cross-valation results. 
        Optional Input: 
        title, str. Plot title. 
        
        """
        
        if not(hasattr(self,'cv_score_table_')):
            cv_score_table_ = cv_score_table(self.res_sprm_cv) 
            setattr(self,'cv_score_table_',cv_score_table_)
        fig = pp.figure()
        fig.set_facecolor(self.colors[0])
        pp.rcParams['axes.facecolor'] = self.colors[1]
        ax1 = fig.add_subplot(111)
        ax1.tricontour(self.cv_score_table_.values[:,0],self.cv_score_table_.values[:,1],self.cv_score_table_.values[:,2])
        pp.title(title)
        pp.show()          