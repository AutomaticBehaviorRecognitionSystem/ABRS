# extract ST Features from ST images (dim reduction)

import numpy as np
import scipy
from scipy import ndimage
from scipy import misc
import pickle
import time
import matplotlib.pyplot as plt
import cv2
import os
import natsort

from ABRS_modules import discrete_radon_transform
from ABRS_modules import subplot_images

def project_to_basis_fun (dirPathInput,dirPathU,dirPathOutput,numbFiles):

        
    UDirPathFileName = dirPathU + '\\' + 'USVdictTrainingSet_ST_Gr33a_dust_th0_10_averSubT2_binVar';normalizeByMax = 0;


    with open(UDirPathFileName, "rb") as f:
         USVdict = pickle.load(f)

    U=USVdict['U'];
    
    fileList = natsort.natsorted(os.listdir(dirPathInput),reverse=False) #on 12/10/2018 #pip install natsort

    featureMat = np.zeros((30,numbFiles*50));
    posMat = np.zeros((2,numbFiles*50));
    maxMovementMat = np.zeros((1,numbFiles*50));
    
    ind = 0;

    for fl in range(0, numbFiles, 1):

        inputFileName = fileList[fl];
        print (inputFileName);

        fileDirPathInputName = dirPathInput + '\\' + inputFileName; #fir Win
        #fileDirPathInputName = dirPathInput + '/' + inputFileName; # for Mac

        with open(fileDirPathInputName, "rb") as f:
            dictST = pickle.load(f)
            
        sMRecSp = dictST["sMRecSp"]; 
        sMRec = sMRecSp.todense(); #from scipy
        sMRec = np.nan_to_num(sMRec);

        dictPosRec = dictST["dictPosRec"];
        xPosRec = dictPosRec["xPosRec"];
        yPosRec = dictPosRec["yPosRec"];

        maxMovementRec = dictST["maxMovementRec"];

        for i in range(0, sMRec.shape[0]):

            imST = np.reshape(sMRec[i,:],(80,80));
            imR = discrete_radon_transform(imST, 80);
            F = np.fft.fft(imR,axis = 0);
            FF = np.fft.fft(np.absolute(F),axis = 1);
            aFF = np.absolute(FF);
            if normalizeByMax == 1:
                naFF = aFF/np.max(np.max(aFF));
            if normalizeByMax == 0:
                naFF = aFF;
            naFFNZ = np.nan_to_num(naFF);

            vecFF = np.reshape(np.absolute(naFFNZ),(80*80,1));

            for dim in range(0,30):

                Udim = U[:,dim];

                prDim = np.dot(Udim,vecFF);prDim = prDim[0];
                
                featureMat[dim,ind] = prDim;

            posMat[0,ind] = xPosRec[i];
            posMat[1,ind] = yPosRec[i];

            maxMovementMat[0,ind] = maxMovementRec[i];

            ind = ind +1;
            
    STF_30_posXY_dict = {"featureMat" : featureMat, "posMat" : posMat, "maxMovementMat" : maxMovementMat};
    
    outputFileName = 'STF_30_posXY_dict' + inputFileName[5:];

    newPath = dirPathOutput + '\\' + outputFileName;
                
    with open(newPath, "wb") as f:
        pickle.dump(STF_30_posXY_dict, f)
        
    return STF_30_posXY_dict        


numbFiles=720; #number of sMRec files in ST image folders #crimson data in Data_demo

#pathToABRSfolder = 'INSERT PATH TO ABRS FOLDER HERE'
pathToABRSfolder = 'C:\\Users\\ravbar\\Desktop\\ABRS_GH_out'

firstFolder = 0; 

dirPathInput = pathToABRSfolder + '\\Data_demo\\ST' #ST image folder; contains subfolders with ST images

dirPathU = pathToABRSfolder + '\\Filters'; # path to Filters (basis from SVD training; U matrix)

dirPathOutput = pathToABRSfolder + '\\Data\\ST_features' # output folder where feature files will be written 

STFolderList = os.listdir(dirPathInput);
sz = np.shape(STFolderList);sizeSTFolder = sz[0];

for fld in range(firstFolder, sizeSTFolder):

    currentSTFolder = STFolderList[fld];

    dirPathInputSTfolder = dirPathInput + '\\' + currentSTFolder;

    checkIfFolder = os.path.isdir(dirPathInputSTfolder);

    if checkIfFolder == True:
        
        STF_30_posXY_dict = project_to_basis_fun (dirPathInputSTfolder,dirPathU,dirPathOutput,numbFiles)                

#imB=U[:,4].reshape(80,80);plt.matshow(imB[5:75,5:75], interpolation=None, aspect='auto');plt.show()
