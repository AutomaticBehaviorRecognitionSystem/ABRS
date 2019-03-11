

# Copyright (c) 2019 Primoz Ravbar UCSB
# Licensed under BSD 2-Clause [see LICENSE for details]
# Written by Primoz Ravbar

import numpy as np
import scipy
from scipy import ndimage
from scipy import misc
import pickle
import time
import matplotlib.pyplot as plt
import cv2
import os

from ABRS_modules import discrete_radon_transform
from ABRS_modules import create_ST_image
from ABRS_modules import smooth_2d
from ABRS_modules import center_of_gravity
from ABRS_modules import subplot_images
from ABRS_modules import getting_frame_record
from ABRS_modules import read_frames
from ABRS_modules import subtract_average



def video_clips_to_ST_image_fun (dirPathInput,dirPathOutput,fbList,clipStart,clipEnd,clipsNumber,bufferSize,windowST,OSplatform='winds'):

    #OSplatform = 'winds'

    clips = np.arange(clipFirst,clipsNumber,1);

    fileList = os.listdir(dirPathInput);

    clIndex = 0;

    for cl in clips:


        fileName = fileList[cl];

        ext = fileName[-3:];

        if (ext == 'avi' or ext == 'mov') == True:

            fileDirPathInputName = dirPathInput + '\\' + fileName;

            r = np.arange(0,clipEnd,bufferSize);


            for bf in r:

                    startFrame = bf;
                    endFrame = startFrame + bufferSize;

                        
                    frRec = read_frames(startFrame, endFrame, fileDirPathInputName, [400,400]);


                    if clIndex == clips[0] and startFrame == clipStart:
                        frRecRemain = np.zeros((windowST,160000));
                        frRec = np.concatenate((frRecRemain,frRec), axis=0);

                    if clIndex > clips[0] or startFrame > clipStart:
                        frRec = np.concatenate((frRecRemain,frRec), axis=0);  

                    for fb in fbList:  

                        for w in range(0,frRec.shape[0]-windowST):

                            startWin = w;
                            endWin = startWin + windowST;
                       
                            posDic, maxMovement, cfrVectRec, frameVectFloatRec = getting_frame_record(frRec, startWin, endWin,fb);
                            
                            cG=center_of_gravity(cfrVectRec); # produce basic ST image: cG                           
                            
                            averageSubtFrameVecRec = subtract_average(cfrVectRec,0);
                            
                            totVar = np.sum(np.absolute(averageSubtFrameVecRec),axis=0);
                            imVar = np.reshape(totVar,(80,80));
                            imVarNorm = imVar/np.max(np.max(imVar));
                            imVarBin = np.zeros((80,80));
                            imVarBin[imVarNorm > 0.30] = 1; #min var threshold at 0.30 
                            
                            I = np.reshape(cG,(80,80))*imVarBin;
                            I = np.nan_to_num(I);
                            

                            imSTsm = smooth_2d (I, 3);
                            sM = imSTsm.reshape((80*80));
                            sM = np.nan_to_num(sM);
                            
                            if np.max(sM) > 0:
                                sMNorm = sM/np.max(sM);
                            else:
                               sMNorm = sM;

                            xPos = posDic["xPos"];
                            yPos = posDic["yPos"]; 

                            if w == 0:
                            
                                   sMRec = sMNorm;
                                   xPosRec = xPos;
                                   yPosRec = yPos;
                                   maxMovementRec = maxMovement;
                                
                            if w > 0:
                                
                                   sMRec = np.vstack((sMRec,sMNorm));
                                   xPosRec = np.vstack((xPosRec,xPos));
                                   yPosRec = np.vstack((yPosRec,yPos));
                                   maxMovementRec = np.vstack((maxMovementRec,maxMovement));
                                   

                        sMRecTh = sMRec; sMRecTh[sMRec < 0.1] = 0; #   set threshold at 0.1                         
                        sMRecSp = scipy.sparse.csr_matrix(sMRecTh);
                       
                        dictPosRec = {"xPosRec" : xPosRec, "yPosRec" : yPosRec};

                        dictST = {"sMRecSp" : sMRecSp, "dictPosRec" : dictPosRec, "maxMovementRec" : maxMovementRec};
                        
                        nameSMRec = 'dictST_' + fileName[0:-4] + '_bf_' + str('%06.0f' % bf) + '_fb' + str(fb);
                       
                        if OSplatform == 'winds':           
                            #newPath = dirPathOutput + '\\' + fileName[0:-4] + '_fb' + str(fb)  #!!!!!!works with non-numbered clips (one clip per video recording)
                            newPath = dirPathOutput + '\\' + fileName[0:-6] + '_fb' + str(fb) #!!!!!!works with numbered clips (multiple clips of the same video; clip names end with index number)

                        if OSplatform == 'mac':           
                            #newPath = dirPathOutput + '/' + fileName[0:-4] + '_fb' + str(fb)  #!!!!!!works with non-numbered clips (one clip per video recording)
                            newPath = dirPathOutput + '/' + fileName[0:-6] + '_fb' + str(fb) #!!!!!!works with numbered clips (multiple clips of the same video; clip names end with index number)
 

                        if not os.path.exists(newPath):
                            os.mkdir(newPath);
                        if OSplatform == 'winds': 
                            fileDirPathOutputName = newPath + '\\' + nameSMRec;
                        if OSplatform == 'mac': 
                            fileDirPathOutputName = newPath + '/' + nameSMRec;    
                                   
                        with open(fileDirPathOutputName, "wb") as f:
                            pickle.dump(dictST, f)

                             
                    frRecSh = frRec.shape;
                    frRecRemain = frRec[bufferSize:frRecSh[0],:];
            clIndex = clIndex +1;
            print(clIndex)
                
OSplatform = 'winds';
#OSplatform = 'mac';

frameRate = 30;

windowST = 16;

clipStart = 0;

clipEnd = 1200*frameRate; #crimson movies #length of clip in seconds


clipsNumberMax = 51; #


fbList = [1,2,3,4]; # works for raw movies with 2x2 arenas (split the frames into 4)
#fbList = [1]; # one arena in the frame

clipFirst = 0;

bufferSize = 50;

firstFolder = 0; #

pathToABRSfolder = 'INSERT PATH TO ABRS MAIN FOLDER HERE'

if OSplatform == 'winds':
    # path to raw video data folder; video clips must be in subfolders at this path
    rawVidDirPath = pathToABRSfolder + '\\Data_demo\\RawVideo'

    # the output folder for ST images; subfolders will be created for each movie
    dirPathOutput = pathToABRSfolder + '\\Data\\ST';

if OSplatform == 'mac':
    # path to raw video data folder; video clips must be in subfolders at this path
    rawVidDirPath = pathToABRSfolder + '/Data_demo/RawVideo'

    # the output folder for ST images; subfolders will be created for each movie
    dirPathOutput = pathToABRSfolder + '/Data/ST'  

videoFolderList = sorted(os.listdir(rawVidDirPath));
#videoFolderList = os.listdir(rawVidDirPath);
sz = np.shape(videoFolderList);sizeVideoFolder = sz[0];

for fld in range(firstFolder, sizeVideoFolder):
    
    print(fld)

    currentVideoFolder = videoFolderList[fld];

    if currentVideoFolder[-5:] != 'Store' :

        dirPathInput = rawVidDirPath + '\\' + currentVideoFolder;
        clipList = sorted(os.listdir(dirPathInput));szClipList = np.shape(clipList);
        clipsNumber = szClipList[0];

        if clipsNumber > clipsNumberMax:
            clipsNumber = clipsNumberMax;

        video_clips_to_ST_image_fun (dirPathInput,dirPathOutput,fbList,clipStart,clipEnd,clipsNumber,bufferSize,windowST,OSplatform);


   
    
    
    
