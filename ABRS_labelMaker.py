#ABRS_labelMaker

# Copyright (c) 2019 Primoz Ravbar UCSB
# Licensed under BSD 2-Clause [see LICENSE for details]
# Written by Primoz Ravbar

import numpy as np
import matplotlib.pyplot as plt
import cv2
import pickle
#import msvcrt


from scipy import misc #pip install pillow
import scipy
from scipy import ndimage

from PIL import Image


from ABRS_modules import getting_frame_record
from ABRS_modules import center_of_gravity
from ABRS_modules import subtract_average
from ABRS_modules import smooth_2d
from ABRS_modules import smooth_1d
from ABRS_modules import discrete_radon_transform
from ABRS_modules import computeSpeedFromPosXY
from ABRS_modules import create_3C_image

import tensorflow as tf

from tensorflow import keras
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten
from tensorflow.keras.layers import Conv2D, MaxPooling2D
from tensorflow.keras.callbacks import TensorBoard

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, LSTM



cap = cv2.VideoCapture('INSERT PATH TO RAW MOVIE.avi');fb=1


newSize = (400,400)
startFrame = 0  # set this to any frame in the movie clip
endFrame = 34900

kernelSize = 100
smoothingWindow = 89

windowSize = 100 #size of window for training -- ignore in this version

winST = 16

halfWindowSpeed = 15

movementThreshold = 150

ind = 0
beh = 0
indBehDur = 0

indDownSamp = 0


prevFrame = np.zeros((400,400))
frRec = np.zeros((16+1,newSize[0]*newSize[1]))

trainImRec = np.zeros((80*80,1000))
trainLabelRec = np.zeros((1,1000))

predictionsProbRec = np.zeros((10,endFrame))

etho = np.zeros((1,endFrame))
ethoCorr = np.zeros((1,np.shape(trainLabelRec)[1]))

pathToABRSfolder = 'INSERT PATH TO ABRS MAIN FOLDER HERE'

    

model = keras.models.load_model('modelConv2ABRS_3C_train_with_descendingcombinedwithothers_avi_10') #this is a CNN model
model.summary()

featureCol = np.zeros((30,1));
featureColAP = np.zeros((30,1));
posCol = np.zeros((2,1));
imCol = np.zeros((80*80,1));
behCol = np.zeros((1,1));

featureMat = np.zeros((30,kernelSize))
posMat = np.zeros((2,kernelSize))
imMat = np.zeros((80*80,windowSize))
behMat = np.zeros((1,windowSize))

im3Crec = np.zeros((1000,80,80,3))

kernelInd = 0
trainInd = windowSize
keyInd = 0
frameInd = 0


for frameInd in range(startFrame,endFrame,1):

    cap.set(1,frameInd)

    if indDownSamp == 0:
        ret, frame = cap.read() #
        
    print(np.shape(frame)) 

    if np.size(np.shape(frame)) != 0:            
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) # check if the frame is not corrupt
    else:
        gray = np.zeros((1024, 1024)) # size of the frame being used!

    rs = cv2.resize(gray,(newSize[0],newSize[1]))

    cropedSize = int(np.shape(gray)[0]/2)

    if fb == 1:
        grayCroped = gray[0:cropedSize,0:cropedSize]
    if fb == 2:
        grayCroped = gray[0:cropedSize,cropedSize:cropedSize*2]
    if fb == 3:
        grayCroped = gray[cropedSize:cropedSize*2,0:cropedSize]
    if fb == 4:
        grayCroped = gray[cropedSize:cropedSize*2,cropedSize:cropedSize*2]

    currentFrame = rs.astype(float)/1;
    diffFrame = currentFrame - prevFrame;
    prevFrame = currentFrame;

    diffFrameAbs = np.absolute(diffFrame)

    frameVect = currentFrame.reshape(1,newSize[0]*newSize[1]);
    frameVectFloat = frameVect.astype(float);

    frRecShort = np.delete(frRec, 0, 0);
    frRec = np.vstack((frRecShort,frameVectFloat));

    sumFrRec = np.sum(frRec,0);
    
    posDic, maxMovement, cfrVectRec, frameVectFloatRec = getting_frame_record(frRec, 0, winST,fb);
  
    im3CRaw = create_3C_image (cfrVectRec)
        
    if np.count_nonzero(im3CRaw[:,:,0])>6400:            
        im3CRaw[:,:,0] = np.zeros((80,80))
        
    if np.count_nonzero(im3CRaw[:,:,1])>800:            
        im3CRaw[:,:,1] = np.zeros((80,80))
        
    rgbArray = np.zeros((80,80,3), 'uint8')
    rgbArray[..., 0] = im3CRaw[:,:,0]
    rgbArray[..., 1] = im3CRaw[:,:,1]
    rgbArray[..., 2] = im3CRaw[:,:,2]
    im3C = Image.fromarray(rgbArray)

    X_rs = np.zeros((1,80,80,3))
        
    X_rs[0,:,:,:]=im3C

    storeFrameRec = 0
    if storeFrameRec == 1:
        im3Crec[frameInd,:,:,:]=im3C

    X = X_rs/256  # normalize


    predictionsProb = model.predict(X)

    predictionsProbRec[:,ind] = predictionsProb

    predictionLabel = np.zeros((1,np.shape(predictionsProb)[0]))
    predictionLabel[0,:] = np.argmax(predictionsProb,axis=1)
        
    behPr = beh
    beh = predictionLabel

    if maxMovement < movementThreshold: #this is to set the threshold for signal strenght
        beh=7
        
    etho[0,ind]=beh

########## denoise/correct for very short behaviors #######

    if behPr == beh:    
        indBehDur = indBehDur +1

    if behPr != beh:

        if indBehDur >= 5:    
            indBehDurNew = 1
        
        if indBehDur < 5:
            etho[0,ind-indBehDur:ind] = beh
            behPr = beh
            #print("---");print(indBehDur)
            #print(etho[0,ind-indBehDur:ind])
            indBehDurNew = indBehDur + 4

        indBehDur = indBehDurNew

##########


    if ind > 2:  
        print(etho[0,ind-1])

    ###### this part is being developed for online training and for semi-automatic ethogram production 
    
    trainKey = 'n'
    if keyInd == windowSize: 
        trainKey = input('train?')
    

    if trainKey == 't':

        trainLabelRec[0,trainInd-windowSize:trainInd] = beh
        trainImRec[:,trainInd-windowSize:trainInd] = imMat
        
        trainInd = trainInd +windowSize
        keyInd=0
        print(trainKey)

    if trainKey == 'f':
        beh = input('behavior?')
        trainLabelRec[0,trainInd-windowSize:trainInd] = beh
        trainImRec[:,trainInd-windowSize:trainInd] = imMat
        
        trainInd = trainInd +windowSize
        keyInd=0
        print(trainKey)    

    if trainKey != 't' and keyInd>windowSize:

        trainLabelRec[0,trainInd-windowSize:trainInd] = beh
        trainImRec[:,trainInd-windowSize:trainInd] = imMat

        trainInd = trainInd +windowSize
        
        keyInd=0
        print(trainKey)

    if trainKey == 'c':

        break

    keyInd = keyInd + 1


    ##################################################################

    
    cv2.imshow('im3CRaw',im3CRaw)
    #cv2.imshow('frame',gray)
    cv2.imshow('grayCroped',grayCroped)


    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

    if ind > endFrame-1:
        break

    ind=ind+1
        
cap.release()
cv2.destroyAllWindows()


for i in range(0,np.shape(trainLabelRec)[1]):

    if trainLabelRec[0,i] != etho[0,i]:
        ethoCorr[0,i] = trainLabelRec[0,i]
    if trainLabelRec[0,i] == etho[0,i] or trainLabelRec[0,i]==0:
        ethoCorr[0,i] = etho[0,i]    

#with open('label' + 'Live' + '_fb' + str(fb), "wb") as f:    
with open('label' + 'CantonS_decap_dusted_9' + '_fb' + str(fb), "wb") as f:
        pickle.dump(ethoCorr, f)

#plt.matshow(etho,interpolation=None, aspect='auto');plt.show()
