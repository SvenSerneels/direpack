# -*- coding: utf-8 -*-
"""
Created on Sat Apr 11 17:25:42 2020

@author: Emmanuel Jordy Menvouta
"""

from __future__ import absolute_import, division, print_function
from __future__ import unicode_literals



from ..sudire.sudire import sudire
from ..utils.utils import MyException
import matplotlib.pyplot as pp 
import numpy as np


class sudire_plot(sudire):
    
    def __init__(self,res_sudire,colors,markers=['o','d','v'],*args):
        """
        Initialize with 
        res_sudire, a sudire class object
        
        Only mandatory input is colors, a list of colors for 
            [0] borders of pane 
            [1] plot background
            [2] marker fill
            [3] diagonal line 
            [4] marker contour, if different from fill
            [5] marker color for new cases, if applicable
               
        """
        if not(isinstance(res_sudire,sudire)):
            raise(MyException("Object supplied to sudireplot needs to be a sudire object"))
        self.res_sudire = res_sudire
        self.colors = colors
        self.markers = markers
        
    def plot_yyp(self,ytruev=[],Xn=[],label=[],names=[],namesv=[],title=[],legend_pos='lower right',onlyval=False):
        """
        plot_yyp will plot y vs y predicted for sudire M opbjects
        Optional inputs: 
            ytruev: array (new_cases,) of predictands
            Xn: array (new_cases,variables) of predictors 
            If these arguments are supplied, sudire predictions for ytrue will be 
                made from Xn through res_sudire.predict()
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
            ytruec = self.res_sudire.y0
            if len(ytruec.shape) >1:
                ytruec = np.array(ytruec).reshape(-1).astype('float64')
            ypredc = np.array(self.res_sudire.ols_obj_.fittedvalues).T.reshape(-1)
            
            ax1.scatter(ytruec, ypredc, c=self.colors[2], label=label,
                    zorder=1,edgecolors=self.colors[4],marker=self.markers[0])
            pp.xlabel("y-true")
            pp.ylabel("y-pred")
            
        else:
            if (len(Xn)==0):
                ValueError('In onlyval=True mode, new cases Xn need to be provided')
        if not(len(Xn)==0):
            if len(ytruev.shape) >1:
                ytruev = np.array(ytruev).reshape(-1).astype('float64')
            ypredv = self.res_sudire.predict(Xn)
            ypredv = np.array(ypredv).reshape(-1).astype('float64')
                
            ax1.scatter(ytruev,ypredv,c=self.colors[5],label=label,
                        zorder=1,edgecolors=self.colors[4],marker=self.markers[0])
            pp.xlabel("y-true")
            pp.ylabel("y-pred")
           
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
        
    def plot_projections(self,Xn=[],label=[],components = [0,1],names=[],namesv=[],title=[],legend_pos='lower right',onlyval=False):
        
        """
        plot_projections will plot the score space  
        Optional inputs: 
            Xn: array (new_cases,variables) of predictors 
            If supplied, sudire projections for new cases will be 
                made from Xn through res_sudire.transform()
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
            Tc = np.array(self.res_sudire.x_scores_)
            ax1.scatter(Tc[:,components[0]], Tc[:,components[1]], c=self.colors[2], label=label, 
                    zorder=1,edgecolors=self.colors[4],marker=self.markers[0])
        else:
            if (len(Xn)==0):
                ValueError('In onlyval=True mode, new cases Xn need to be provided')
        if not(len(Xn)==0):
            Tv = np.array(self.res_sudire.transform(Xn))
            ax1.scatter(Tv[:,components[0]], Tv[:,components[1]],c=self.colors[5],label=label,
                        zorder=1,edgecolors=self.colors[4],marker=self.markers[0])
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
        
        
        

