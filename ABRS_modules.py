# Copyright (c) 2019 Primoz Ravbar UCSB
# Licensed under BSD 2-Clause [see LICENSE for details]
# Written by Primoz Ravbar

#This file contains functions used with ABRS
#Last updated: 01/06/2020

import numpy as np
import scipy
from scipy import ndimage
from scipy import misc #pip install pillow
import pickle
import time
import matplotlib.pyplot as plt
import cv2
import os

from scipy.signal import savgol_filter

def create_ST_image(cfrVectRec):

    shapeCfrVectRec = cfrVectRec.shape;
    rw = np.zeros((1,shapeCfrVectRec[1])); 
    M1 = np.concatenate((cfrVectRec,rw), axis=0);
    M2 = np.concatenate((rw,cfrVectRec), axis=0);

    dM = M1 - M2;
    dM = dM[1:16,:];

    dM1 = np.concatenate((dM,rw), axis=0);
    dM2 = np.concatenate((rw,dM), axis=0);
    dM1M2 = dM1*dM2;
    MST = dM1M2*0;
    MST[dM1M2<0] = dM1M2[dM1M2<0];
    

    sM = np.sum(np.absolute(MST), axis=0);
    


    I = np.reshape(sM,(80,80));

	 
    return I, sM, MST;

def center_of_gravity(cfrVectRec):

    sh = np.shape(cfrVectRec);

    F=np.absolute(np.fft.fft(cfrVectRec,axis=0))

    
    av = np.zeros((1,sh[0]));
    av[0,:] = np.arange(1,sh[0]+1);
    A = np.repeat(av,sh[1],axis=0);

    FA = F*np.transpose(A);

    sF = np.sum(F,axis=0);
    sFA = np.sum(FA,axis=0);

    cG = sFA/sF


    return cG

def center_of_gravity2(cfrVectRec):


    F=np.absolute(np.fft.fft(cfrVectRec,axis=0))

    sumF = np.sum(F,axis=0);
    Fnorm = F/sumF;
    F=Fnorm;
    
    shF = np.shape(F);
    halfFreq = int(np.ceil(shF[0]/2));

    F1 = F[0:halfFreq,:];
    
    shF1 = np.shape(F1);
    
    av = np.zeros((1,shF1[0]));
    av[0,:] = np.arange(1,shF1[0]+1);
    A = np.repeat(av,shF1[1],axis=0);

    FA = F1*np.transpose(A);

    sF = np.sum(F1,axis=0);
    sFA = np.sum(FA,axis=0);

    cG = sFA/sF

    return cG, F1

def center_of_gravity3(cfrVectRec):

    shX = np.shape(cfrVectRec);

    zeroPdd = 9;

    Xz=np.vstack((np.zeros((zeroPdd,shX[1])),cfrVectRec));
    Xz2 = np.vstack((Xz,np.zeros((zeroPdd,shX[1]))));

    F=np.absolute(np.fft.fft(Xz2,axis=0));

    shF = np.shape(F);
    halfFreq = int(np.ceil(shF[0]/2));
    hiCutOff = 0;

    F1 = F[0:halfFreq-hiCutOff,:];
    shF1 = np.shape(F1);

    sumF1 = np.sum(F1,axis=0);
    Fnorm = F1/sumF1;
    F1=Fnorm;
    
    av = np.zeros((1,shF1[0]));
    av[0,:] = np.arange(1,shF1[0]+1);
    A = np.repeat(av,shF1[1],axis=0);

    FA = F1*np.transpose(A);

    sF = np.sum(F1,axis=0);
    sFA = np.sum(FA,axis=0);

    cG = sFA/sF   

    return cG, F1

def create_3C_image (cfrVectRec):

    winSize = np.shape(cfrVectRec)

    cG=center_of_gravity(cfrVectRec)

    averageSubtFrameVecRec = subtract_average(cfrVectRec,0)

    imRaw = np.reshape(cfrVectRec[winSize[0]-1,:],(80,80))

    #imDiff = np.reshape(averageSubtFrameVecRec[1,:]-averageSubtFrameVecRec[0,:],(80,80))
    imDiff = np.reshape(cfrVectRec[winSize[0]-1,:]-cfrVectRec[winSize[0]-2,:],(80,80));
    #imDiff = np.reshape(cfrVectRec[1,:]-cfrVectRec[0,:],(80,80));
    imDiffAbs = np.absolute(imDiff)
    maxImDiffAbs = np.max(np.max(imDiffAbs))
    imDiffCl = np.zeros((80,80))
    imDiffCl[imDiffAbs > maxImDiffAbs/10] = imDiff[imDiffAbs > maxImDiffAbs/10]
    imDiffClNorm = imDiffCl/maxImDiffAbs
                                
    totVar = np.sum(np.absolute(averageSubtFrameVecRec),axis=0)
    imVar = np.reshape(totVar,(80,80))
    imVarNorm = imVar/np.max(np.max(imVar))
    imVarBin = np.zeros((80,80))
    imVarBin[imVarNorm > 0.10] = 1; #######!!!!!!!!!! Set signal threshold default = 0.10
                                
    I = np.reshape(cG,(80,80))*imVarBin;
    I = np.nan_to_num(I);
                                
    imSTsm = smooth_2d (I, 3)
    sM = imSTsm.reshape((80*80))
    sM = np.nan_to_num(sM)

    if np.max(sM) > 0:
        sMNorm = sM/np.max(sM)
    else:
        sMNorm = sM

    I_RS = np.reshape(sMNorm,(80,80))

    imDiffAbs = np.absolute(imDiffClNorm)

    imDiffClNeg = np.zeros((80,80))
    imDiffClPos = np.zeros((80,80))
    imDiffClNeg[imDiffClNorm<0] = np.absolute(imDiffClNorm[imDiffClNorm<0])
    imDiffClPos[imDiffClNorm>0] = imDiffClNorm[imDiffClNorm>0]

    imDiffClNormNeg = imDiffClNeg/np.max(np.max(imDiffClNeg))
    imDiffClNormPos = imDiffClPos/np.max(np.max(imDiffClPos))
    

    rgbArray = np.zeros((80,80,3), 'uint8')
    rgbArray[..., 0] = I_RS*255
    rgbArray[..., 1] = imDiffAbs*255
    rgbArray[..., 2] = imRaw*255

    im3C = rgbArray

    return im3C 
    

def subtract_average(frameVectRec,dim):

    shFrameVectRec = np.shape(frameVectRec);
    averageSubtFrameVecRec = np.zeros((shFrameVectRec[0],shFrameVectRec[1]));

    if dim == 0:
      averageVect = np.mean(frameVectRec,0);

    if dim == 1:
      averageVect = np.mean(frameVectRec,1);
      
    if dim == 0:
        for i in range(0,shFrameVectRec[0]):
           averageSubtFrameVecRec[i,:] = frameVectRec[i,:] - averageVect;

    if dim == 1:
        for i in range(0,shFrameVectRec[1]):
           averageSubtFrameVecRec[:,i] = frameVectRec[:,i] - averageVect;       

    return averageSubtFrameVecRec   

        

def read_frames(startFrame, endFrame, file_name, newSize):
   
    cap = cv2.VideoCapture(file_name)

    print(file_name)

    for i in range(startFrame, endFrame):

        cap.set(1,i);
        ret, frame = cap.read() #get frame


        if np.size(np.shape(frame)) != 0:            
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) #
        else:
            gray = np.zeros((1024, 1024))
        
        rs = cv2.resize(gray,(newSize[0],newSize[1]));
        frameVect = rs.reshape(1,newSize[0]*newSize[1]);
        frameVectFloat = frameVect.astype(float);    

        if i == startFrame:
            frRec = frameVectFloat;
        if i > startFrame:
            frRec = np.vstack((frRec,frameVectFloat));

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()    
        
    return frRec;


def read_frames2(startFrame, endFrame, file_name, newSize):

    
    frRec = np.zeros((endFrame,newSize[0]*newSize[1]))
    
    
    cap = cv2.VideoCapture(file_name)

    print(file_name)

    for i in range(startFrame, endFrame):

        cap.set(1,i);
        ret, frame = cap.read() #get frame
       
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) #convert frame to gray
        
        rs = cv2.resize(gray,(newSize[0],newSize[1]))
        frameVect = rs.reshape(1,newSize[0]*newSize[1])
        frameVectFloat = frameVect.astype(float)

        frRec[i,:] = frameVectFloat


        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()    
        
    return frRec

def read_frames3(startFrame, endFrame, cap, newSize):



    for i in range(startFrame, endFrame):

        cap.set(1,i);
        ret, frame = cap.read() #get frame

        if np.size(np.shape(frame)) != 0:            
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) #
        else:
            gray = np.zeros((1024, 1024))
        
        rs = cv2.resize(gray,(newSize[0],newSize[1]));
        frameVect = rs.reshape(1,newSize[0]*newSize[1]);
        frameVectFloat = frameVect.astype(float);    

        if i == startFrame:
            frRec = frameVectFloat;
        if i > startFrame:
            frRec = np.vstack((frRec,frameVectFloat));

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()    
        
    return frRec;


def getting_frame_record(frRec, startWin, endWin, fb):



    if fb>0:
        sideSize = 200
    if fb==0:
        sideSize = 400
            
    for i in range(startWin,endWin):

            frame = frRec[i,:]
          
            gray = frame.reshape(400,400);

            if fb == 1:
                rf = gray[0:200,0:200];
                
            if fb == 2:   
                rf = gray[0:200,200:400];
               
            if fb == 3:
                rf = gray[200:400,0:200];
               
            if fb == 4:    
                rf = gray[200:400,200:400];

            if fb == 0:    
                rf = gray    

  
            rs = rf   
            
            frameVect = rs.reshape(1,np.shape(rs)[0]*np.shape(rs)[1])
            frameVectFloat = frameVect.astype(float)

 
            if i == startWin:
               previousFrame = frameVectFloat;
               frameDiffComm = previousFrame*0;
               frameVectFloatRec = frameVectFloat;
            
            if i > startWin:
                frameDiffComm = frameDiffComm + np.absolute(frameVectFloat - previousFrame);
                frameVectFloatRec = np.vstack((frameVectFloatRec,frameVectFloat));
                previousFrame = frameVectFloat;


    indMaxDiff = np.argmax(frameDiffComm);
    
    rowMaxDiff = np.floor(indMaxDiff/sideSize)
    colMaxDiff = indMaxDiff - (rowMaxDiff*sideSize)

    rowMaxDiff = rowMaxDiff.astype(int);
    colMaxDiff = colMaxDiff.astype(int);

    maxMovement = np.max(frameDiffComm);

    posDic = {"xPos" : colMaxDiff, "yPos" : rowMaxDiff};
    

    for i in range(0,(endWin-startWin)):
        
           rs = frameVectFloatRec[i,:].reshape(sideSize,sideSize)

           bottomOvershot=0
           rightOvershot=0
           
           topEdge = rowMaxDiff-40;
           if topEdge < 0:
               topEdge=0;
           bottomEdge = rowMaxDiff+40;
           if bottomEdge > sideSize:
               bottomOvershot = bottomEdge-sideSize
               bottomEdge=sideSize
           leftEdge = colMaxDiff-40;
           if leftEdge < 0:
               leftEdge=0;
           rightEdge = colMaxDiff+40;
           if rightEdge > sideSize:
               rightOvershot = rightEdge-sideSize
               rightEdge=sideSize

           cfr = rs[topEdge:bottomEdge,leftEdge:rightEdge];
           shapeCfr = cfr.shape; 


           if topEdge == 0:
               rw = np.zeros((np.absolute(shapeCfr[0]-80),shapeCfr[1]))
               cfr = np.vstack((rw,cfr))
               shapeCfr = cfr.shape; 
           if bottomOvershot > 0:
               rw = np.zeros((np.absolute(shapeCfr[0]-80),shapeCfr[1]))
               cfr = np.vstack((cfr,rw))
               shapeCfr = cfr.shape; 
           if leftEdge == 0:
               col = np.zeros((shapeCfr[0], np.absolute(shapeCfr[1]-80)))
               cfr = np.hstack((col,cfr))
               shapeCfr = cfr.shape; 
           if rightOvershot > 0:
               col = np.zeros((shapeCfr[0], np.absolute(shapeCfr[1]-80)))
               cfr = np.hstack((cfr,col))
               shapeCfr = cfr.shape;

    

           cfrVect = cfr.reshape(1,80*80);


           if i == 0:
               cfrVectRec = cfrVect;
           if i > 0:
               cfrVectRec = np.vstack((cfrVectRec,cfrVect));
            

    return posDic, maxMovement, cfrVectRec, frameVectFloatRec;    

def find_movement_in_fb(rawFrRec, startWin, endWin, fb, newSize):

    sh = np.shape(rawFrRec);
    oldSize = [int(np.sqrt(sh[1])),int(np.sqrt(sh[1]))];

    ratioDev = oldSize[0]/newSize[0];
    print(ratioDev)

    for i in range(startWin,endWin):


            rawFrameVect = rawFrRec[i,:];            
            rawFrame = rawFrameVect.reshape(oldSize[0],oldSize[1]);
            resizedFrame = cv2.resize(rawFrame,(newSize[0],newSize[1]));
            
            

            if fb == 1:
                rf = resizedFrame[0:int(newSize[0]/2),0:int(newSize[0]/2)];
                
            if fb == 2:   
                rf = resizedFrame[0:200,200:400];
               
            if fb == 3:
                rf = resizedFrame[200:400,0:200];
               
            if fb == 4:    
                rf = resizedFrame[200:400,200:400];

            if fb == 0:    
                rf = resizedFrame     

            rs = rf

            
            frameVect = rs.reshape(1,int(newSize[0]/2)*int(newSize[1]/2));
            frameVectFloat = frameVect.astype(float);

            if i == startWin:
               previousFrame = frameVectFloat;
               frameDiffComm = previousFrame*0;
               frameVectFloatRec = frameVectFloat;
            
            if i > startWin:
                frameDiffComm = frameDiffComm + np.absolute(frameVectFloat - previousFrame);
                frameVectFloatRec = np.vstack((frameVectFloatRec,frameVectFloat));
                previousFrame = frameVectFloat;


    indMaxDiff = np.argmax(frameDiffComm);
    
    rowMaxDiff = np.floor(indMaxDiff/int(newSize[0]/2));
    colMaxDiff = indMaxDiff - (rowMaxDiff*int(newSize[0]/2));


    rowMaxDiff = int(rowMaxDiff);
    colMaxDiff = int(colMaxDiff);


    posDic = {"xPos" : int(colMaxDiff*ratioDev), "yPos" : int(rowMaxDiff*ratioDev)};

    
    for i in range(startWin,endWin):

            rawFrameVect = rawFrRec[i,:];            
            rawFrame = rawFrameVect.reshape(oldSize[0],oldSize[1]);

            if fb == 1:
                rawFBF = rawFrame[0:int(oldSize[0]/2),0:int(oldSize[0]/2)];
                
            if fb == 2:   
                rf = resizedFrame[0:200,200:400];
               
            if fb == 3:
                rf = resizedFrame[200:400,0:200];
               
            if fb == 4:    
                rf = resizedFrame[200:400,200:400];

            rowPos = posDic["yPos"]; colPos = posDic["xPos"];    

            topEdge = rowPos-40;
            if topEdge < 0:
                topEdge=0;
            bottomEdge = rowPos+40;
            if bottomEdge < 0:
               bottomEdge=0;
            leftEdge = colPos-40;
            if leftEdge < 0:
                leftEdge=0;
            rightEdge = colPos+40;
            if rightEdge < 0:
                rightEdge=0;

            zoomInFrame = rawFBF[topEdge:bottomEdge,leftEdge:rightEdge];
            
            shapeZoomInFrame = zoomInFrame.shape; 
            
            if shapeZoomInFrame[1] < 80:
                col = np.zeros((shapeZoomInFrame[0], np.absolute(shapeZoomInFrame[1]-80)))
                zoomInFrame = np.concatenate((col,zoomInFrame), axis=1);
                shapeZoomInFrame = zoomInFrame.shape;
            if shapeZoomInFrame[0] < 80:
                rw = np.zeros((np.absolute(shapeZoomInFrame[0]-80),shapeZoomInFrame[1]));
                zoomInFrame = np.concatenate((rw,zoomInFrame), axis=0);
                shapeZoomInFrame = zoomInFrame.shape;
    

            zoomInFrameVect = zoomInFrame.reshape(1,80*80);

            if i == 0:
                zoomInFrameVectRec = zoomInFrameVect;
            if i > 0:
                zoomInFrameVectRec = np.vstack((zoomInFrameVectRec,zoomInFrameVect));

    return posDic, zoomInFrameVectRec



def discrete_radon_transform(image, steps):
    
    R = np.zeros((steps, len(image)), dtype='float64')
    for s in range(steps):
        rotation = misc.imrotate(image, -s*180/steps).astype('float64') #pip install pillow
        R[:,s] = sum(rotation)
    return R

def smooth_1d (M, winSm, axis = 1):

    Msm = scipy.signal.savgol_filter(M, winSm, polyorder=0, deriv=0, axis=axis, mode='interp')
    return Msm

def smooth_2d (M, winSm):
    
    Msm = savgol_filter(M, window_length=winSm, polyorder=0);
    return Msm

def etho_mat_2_etho_vect(ethoMatrix):

    maxEtho = np.zeros((1,np.shape(ethoMatrix)[1]))
    maxEtho[0,:] = np.argmax(ethoMatrix[0:np.shape(ethoMatrix)[0],:],axis=0)

    ethoVect = maxEtho

    return ethoVect


def etho2ethoAP (idx):

    sh = np.shape(idx);
    idxAP = np.zeros((1,sh[1]))

    for i in range(0,sh[1]):

       if idx[0,i] == 1 or idx[0,i] == 2:           
           idxAP[0,i] = 1;

       if idx[0,i] == 3 or idx[0,i] == 4 or idx[0,i] == 5:
           idxAP[0,i] = 2;

       if idx[0,i] == 6:           
           idxAP[0,i] = 3;    

    return idxAP

def create_LDA_training_dataset (dirPathFeatures,numbFiles):



    fileList = sorted(os.listdir(dirPathFeatures));

    for fl in range(0, numbFiles, 1):

        inputFileName = fileList[fl];
        print (inputFileName);

        featureMatDirPathFileName = dirPathFeatures + '\\' + inputFileName;

        with open(featureMatDirPathFileName, "rb") as f:
             STF_30_posXY_dict = pickle.load(f);
             #featureMatCurrent = pickle.load(f)
        featureMatCurrent = STF_30_posXY_dict["featureMat"];
        posMatCurrent = STF_30_posXY_dict["posMat"];
        maxMovementMatCurrent = STF_30_posXY_dict["maxMovementMat"];

        if fl == 0:
            
            featureMat =  featureMatCurrent;
            posMat = posMatCurrent;
            maxMovementMat = maxMovementMatCurrent;

        if fl > 0:

            featureMat = np.hstack((featureMat,featureMatCurrent));
            posMat = np.hstack((posMat, posMatCurrent));
            maxMovementMat = np.hstack((maxMovementMat, maxMovementMatCurrent));

    return featureMat, posMat, maxMovementMat

def removeZeroLabelsFromTrainingData (label,data):


    shData = np.shape(data);
    shLabel = np.shape(label);

    newLabel = label*0;
    newData = data*0;

    ind = 0;

    for i in range(0,shData[1]):

        if label[0,i] != 0:    

            newLabel[0,ind] = label[0,i]; 
            newData[:,ind] = data[:,i];

            ind=ind+1;

    return newLabel,newData


def computeSpeedFromPosXY (posMat,halfWindow):

    sh = np.shape(posMat);

    speedMat = np.zeros((1,sh[1]));

    for i in range(halfWindow,sh[1]-halfWindow,1):

        x_sqr =(np.absolute(posMat[0,i+halfWindow]-posMat[0,i-halfWindow]))*(np.absolute(posMat[0,i+halfWindow]-posMat[0,i-halfWindow])); 
        y_sqr =(np.absolute(posMat[1,i+halfWindow]-posMat[1,i-halfWindow]))*(np.absolute(posMat[1,i+halfWindow]-posMat[1,i-halfWindow]));
        speedMat[0,i] = np.sqrt(x_sqr+y_sqr);

    return speedMat

                                                                                

def balance_labels2d(y,X,limitCol):

    shY = np.shape(y)
    shX = np.shape(X)

    yBal = np.zeros((1,1))
    yNew = np.zeros((1,1))

    XIm = np.zeros((1,shX[1],shX[2],shX[3]))
    XBal = np.zeros((shX[0],shX[1],shX[2],shX[3]))


    sumCol = np.zeros((10,1))

    ind=0;

    for i in range(0,shY[0]):

        behInd = int(y[i,0])
        sumCol[behInd]=sumCol[behInd]+1
        

        if sumCol[behInd] < limitCol[behInd]:
            
            ind=ind+1

            yNew[0,0] = behInd;
            XIm = X[i,:,:,:]

            yBal = np.vstack((yBal,yNew))
            XBal[ind,:,:,:] = XIm


    yBal = yBal[1:ind,:]        
    XBal = XBal[1:ind,:,:,:]     
            
    return  yBal,XBal    
    
