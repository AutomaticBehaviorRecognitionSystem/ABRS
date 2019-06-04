# Copyright (c) 2019 Primoz Ravbar UCSB
# Licensed under BSD 2-Clause [see LICENSE for details]
# Written by Primoz Ravbar

#This file contains functions used with ABRS

import numpy as np
import scipy
from scipy import ndimage
from scipy import misc
import time
import matplotlib.pyplot as plt
import os

from ABRS_modules import etho2ethoAP

from matplotlib.colors import ListedColormap
from matplotlib.patches import Rectangle

def create_colorMat(colorByVect):
    
    
    shV = np.shape(colorByVect);
    cM=np.zeros((shV[1],3))
    cM_AP=np.zeros((shV[1],3)) 

    indB = np.where(colorByVect[0,:]  == 1);cM[indB[0]]=[0.9, 0.6, 0.3]
    indB = np.where(colorByVect[0,:]  == 2);cM[indB]=[0.4, 0, 0.9]
    indB = np.where(colorByVect[0,:]  == 3);cM[indB]=[0, 0.4, 0.0]
    indB = np.where(colorByVect[0,:]  == 4);cM[indB]=[0, 0.9, 0.0]
    indB = np.where(colorByVect[0,:]  == 5);cM[indB]=[0, 0.6, 1.0]
    indB = np.where(colorByVect[0,:]  == 6);cM[indB]=[0, 0.0, 0.0]
    indB = np.where(colorByVect[0,:]  == 7);cM[indB]=[1, 1.0, 1.0]

    indB = np.where(colorByVect[0,:]  == 1);cM_AP[indB[0]]=[1, 0.0, 0.0]
    indB = np.where(colorByVect[0,:]  == 2);cM_AP[indB]=[1, 0.0, 0.0]
    indB = np.where(colorByVect[0,:]  == 3);cM_AP[indB]=[0.0, 0, 1.0]
    indB = np.where(colorByVect[0,:]  == 4);cM_AP[indB]=[0.0, 0, 1.0]
    indB = np.where(colorByVect[0,:]  == 5);cM_AP[indB]=[0.0, 0, 1.0]
    indB = np.where(colorByVect[0,:]  == 6);cM_AP[indB]=[0, 0.0, 0.0]
    indB = np.where(colorByVect[0,:]  == 7);cM_AP[indB]=[1, 1.0, 1.0]    
        
    
    return cM, cM_AP

cmapG = ListedColormap([[0, 0.0, 0.0],[0.9, 0.6, 0.3], [0.4, 0, 0.9], [0, 0.4, 0.0], [0, 0.9, 0.0], [0, 0.6, 1.0], [0, 0.0, 0.0],[1, 1.0, 1.0]])
cmapAP = ListedColormap([[0, 0.0, 0.0],[1, 0.0, 0.0], [0.0, 0, 1.0], [0, 0.0, 0.0]])

cMatG = [[0, 0.0, 0.0],[0.9, 0.6, 0.3], [0.4, 0, 0.9], [0, 0.4, 0.0], [0, 0.9, 0.0], [0, 0.6, 1.0], [0, 0.0, 0.0],[1, 1.0, 1.0]]
cMatAP = [[0, 0.0, 0.0],[1, 0.0, 0.0], [0.0, 0, 1.0], [0, 0.0, 0.0]]



def plot_data_with_stats (rawData,means,stds,numbOfBoxes = 1):

    colList = ['r', 'g', 'b']
    
    numbNonZero = 0
    
    for b in range(0,numbOfBoxes):
        numbNonZeroNew = np.count_nonzero(rawData[:,b])
        if numbNonZeroNew > numbNonZero:
            numbNonZero = numbNonZeroNew

    xAxis = np.ones((1,numbNonZero))- 0.25 + np.random.rand(1,numbNonZero)*0.25

    colIndex = 0
    for b in range(0,numbOfBoxes):

        numbNonZero = np.count_nonzero(rawData[:,b])

        currentAxis = plt.gca()
        currentAxis.add_patch(Rectangle(((b+1 - 0.35), means[0,b]/30), 0.5, stds[0,b]/30, alpha=0.35,facecolor='k'))
        currentAxis.add_patch(Rectangle(((b+1 - 0.35), means[0,b]/30), 0.5, -stds[0,b]/30, alpha=0.35,facecolor='k'))

        currentAxis.add_patch(Rectangle(((b+1 - 0.35), means[0,b]/30), 0.5, (stds[0,b]/30)/np.sqrt(numbNonZero), alpha=0.5,facecolor='k'))
        currentAxis.add_patch(Rectangle(((b+1 - 0.35), means[0,b]/30), 0.5, -(stds[0,b]/30)/np.sqrt(numbNonZero), alpha=0.5,facecolor='k'))

        plt.plot([b+1 - 0.45 , b+1 + 0.25],[means[0,b]/30 , means[0,b]/30],c='r')

        plt.scatter(xAxis[0,0:numbNonZero]+b,rawData[0:numbNonZero,b]/30,s=5,c=colList[colIndex],alpha=0.25)

        colIndex = colIndex + 1
        if colIndex > 2: colIndex = 0    
