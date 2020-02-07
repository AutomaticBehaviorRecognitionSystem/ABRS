#ABRS_label_maker_GUI

# Copyright (c) 2019 Primoz Ravbar UCSB
# Licensed under BSD 2-Clause [see LICENSE for details]
# Written by Primoz Ravbar

import tkinter
from tkinter import filedialog as fd

import matplotlib
import matplotlib.pyplot as plt
matplotlib.use("TkAgg")
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure


import cv2
import PIL.Image, PIL.ImageTk
from PIL import Image
import time
import numpy as np
import pickle

from ABRS_modules import *

import tensorflow as tf

from tensorflow import keras
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten
from tensorflow.keras.layers import Conv2D, MaxPooling2D
from tensorflow.keras.callbacks import TensorBoard

from tensorflow.keras.models import Sequential
 
class LabelMakerGUI:

     
     def __init__(self, window, window_title, dictParam, boolFirstRun = True):
         
         self.window = window
         self.window.title(window_title)

         menu = tkinter.Menu(window)
         window.config(menu=menu) 
         filemenu = tkinter.Menu(menu)
         menu.add_cascade(label='File', menu=filemenu) 
         filemenu.add_command(label='New') 
         filemenu.add_command(label='Open', command=self.get_file_path_name)
         filemenu.add_command(label='Exit', command=window.quit) 

         self.dictParam = dictParam
         
         self.videoSource = self.dictParam["videoSource"]

         self.modelPathName = self.dictParam["modelPathName"]
         self.savedLabelsPath = self.dictParam["savedLabelsPath"]
         self.savedTrainingDataPath = self.dictParam["savedTrainingDataPath"]

         self.ethoLength = self.dictParam["zoomEthoLength"]
         self.posRatio = int(1000/self.ethoLength)

         self.startZoom = 0
         self.endZoom = self.ethoLength

         self.boolDoRun = False
         self.strLabel = "0"

         self.startFrame=5
         self.endFrame=self.startFrame+5


         self.ethoCurrent = np.zeros((25,self.ethoLength))

 
         # 
         #self.canvas = tkinter.Canvas(window, width = self.width, height = self.height)
         self.canvas = tkinter.Canvas(window, width = 1000, height = 500)
         self.canvas.place(relx=0.01, rely=0.01, relwidth=0.7, relheight=0.5)


         # etho canvas  
         self.canvasEtho = tkinter.Canvas(window, bg='green', bd=5)
         self.canvasEtho.place(relx=0.01, rely=0.45+0.05, relwidth=0.8, relheight=0.3)
         self.canvasEtho.bind("<Button-1>", self.etho_callback)

         self.figure1 = plt.Figure(figsize=(5,4), dpi=100)
         self.ax1 = self.figure1.add_subplot(111)

         # Entry correct it
         self.entryCorrectIt = tkinter.Entry(self.canvasEtho)
         self.entryCorrectIt.place(relx=0.95,rely=0.5, relheight=0.2, relwidth=0.05)

         # Button Correct it!
         self.btnCorrectIt=tkinter.Button(self.canvasEtho, text="Correct it!", width=50, command=self.correct_etho)
         self.btnCorrectIt.place(relx=0.9,rely=0.2, relheight=0.2, relwidth=0.1)

         # Button Save Labels 
         self.btnSaveLabels=tkinter.Button(self.canvasEtho, text="Save Labels", width=50, command=self.save_labels_and_images)
         self.btnSaveLabels.place(relx=0.9,rely=0.8, relheight=0.2, relwidth=0.1)
         
         
         # Frame controls      
         self.controlFrame = tkinter.Frame(window, bg='#80c1ff', bd=5)
         self.controlFrame.place(relx=0.01, rely=0.7+0.1, relwidth=0.8, relheight=0.15)

         # Scale frame position 
         self.framePosScale = tkinter.Scale(self.controlFrame, from_=0, to=self.dictParam["zoomEthoLength"], orient="horizontal")
         self.framePosScale.place(relx=0.01, rely=0.6,relwidth=0.8)
         

         # Button frame forward
         self.btnFrameForward=tkinter.Button(self.controlFrame, text="Forward", width=50, command=self.move_frame_forward)
         self.btnFrameForward.place(relx=0.4,rely=0.2, relheight=0.3, relwidth=0.3)

         # Button frame back
         self.btnFrameBack=tkinter.Button(self.controlFrame, text="Backward", width=50, command=self.move_frame_back)
         self.btnFrameBack.place(relx=0.1,rely=0.2, relheight=0.3, relwidth=0.3)

         # Button ZOOM
         self.btnFrameForward=tkinter.Button(self.controlFrame, text="ZOOM Select", width=50, command=self.select_zoom)
         self.btnFrameForward.place(relx=0.8,rely=0.2, relheight=0.3, relwidth=0.1)
         
         self.btnFrameForward=tkinter.Button(self.controlFrame, text="ZOOM Out", width=50, command=self.zoom_out)
         self.btnFrameForward.place(relx=0.9,rely=0.2, relheight=0.3, relwidth=0.1)

         
         # Frame param RUN 
         self.paramFrame = tkinter.Frame(window, bg='#80c1ff', bd=5)
         self.paramFrame.place(relx=0.65, rely=0.01, relwidth=0.1, relheight=0.45)

         self.labelEntryStartFrame = tkinter.Label(self.paramFrame, text="First frame") 
         self.labelEntryStartFrame.place(relx=0.1,rely=0.25, relheight=0.05, relwidth=0.7)


         self.entryStartFrame = tkinter.Entry(self.paramFrame)
         self.entryStartFrame.place(relx=0.1,rely=0.3, relheight=0.05, relwidth=0.7)

         self.labelEntryStartFrame = tkinter.Label(self.paramFrame, text="Last frame")
         self.labelEntryStartFrame.place(relx=0.1,rely=0.45, relheight=0.05, relwidth=0.7)
         
         self.entryEndFrame = tkinter.Entry(self.paramFrame)
         self.entryEndFrame.place(relx=0.1,rely=0.5, relheight=0.05, relwidth=0.7)

         self.btnSetStartEndFrame=tkinter.Button(self.paramFrame, text="ENTER", command=self.set_start_end_frames)
         self.btnSetStartEndFrame.place(relx=0.1,rely=0.6, relheight=0.05, relwidth=0.7)

         self.btnSetStartEndFrame=tkinter.Button(self.paramFrame, text="LOAD frames", command=self.create_buffer_obj)
         self.btnSetStartEndFrame.place(relx=0.1,rely=0.8, relheight=0.05, relwidth=0.7)

         self.labelTextBox = tkinter.Label(window)
         self.labelTextBox.place(relx=0.5, rely=0.2)


         self.frameInd = 5
         self.boolBottomPressed = False

         self.ethoMarker = np.zeros((1,2))
         self.boolEthoMark = True


         if boolFirstRun == True and self.boolDoRun == True:
              
              self.startFrame=5
              self.endFrame=self.startFrame+5
              self.vid = cv2.VideoCapture(self.videoSource)
              self.width = self.vid.get(cv2.CAP_PROP_FRAME_WIDTH)
              self.height = self.vid.get(cv2.CAP_PROP_FRAME_HEIGHT)

              self.ethoRaw = np.zeros((25,self.ethoLength))
              self.ethoRawExp = np.zeros((125,self.ethoLength))
              self.ethoVect = np.zeros((1,self.ethoLength))
              self.ethoCurrent = np.zeros((25,self.ethoLength))

              self.ethoMarker = np.zeros((1,2))
              self.boolEthoMark = True
         
              self.bufferObj = BufferRecord(self.ethoLength,self.modelPathName,self.videoSource,self.startFrame,self.endFrame)

              boolFirstRun = False


         self.delay = 15


         self.update()
 
         self.window.mainloop()

     #########################################################################################

     def create_buffer_obj(self):

         self.boolDoRun = True 

         self.vid = cv2.VideoCapture(self.videoSource)
         self.width = self.vid.get(cv2.CAP_PROP_FRAME_WIDTH)
         self.height = self.vid.get(cv2.CAP_PROP_FRAME_HEIGHT)

         self.ethoRaw = np.zeros((25,self.ethoLength))
         self.ethoRawExp = np.zeros((125,self.ethoLength))
         self.ethoVect = np.zeros((1,self.ethoLength))

         self.ethoMarker = np.zeros((1,2))
         self.boolEthoMark = True
         
         self.bufferObj = BufferRecord(self.ethoLength,self.modelPathName,self.videoSource,self.startFrame,self.endFrame)

         for i in range(0,self.endFrame-self.startFrame):
              
             predLabel, predProb = self.bufferObj.get_predictions(i)   
             self.ethoCurrent[int(predLabel),i+self.startFrame] = 250
             #pass

     def get_file_path_name(self):

          filename = fd.askopenfilename()
          print(filename)
          self.videoSource = filename

          self.ethoMarker = np.zeros((1,2))
          self.boolEthoMark = True


     def set_start_end_frames(self):

         self.startFrame = int(self.entryStartFrame.get())
         self.endFrame = int(self.entryEndFrame.get())

     def etho_callback(self,event):
          

         boolProceed = True

         if self.boolEthoMark == True and boolProceed == True:
              self.ethoMarker[0][0] = event.x
              print(str(self.ethoMarker))
              self.boolEthoMark = False
              boolProceed = False
              

         if self.boolEthoMark == False and boolProceed == True:
              self.ethoMarker[0][1] = event.x
              print(str(self.ethoMarker))
              self.boolEthoMark = True
 
     
     def move_frame_forward(self):

         self.frameInd = self.frameInd + 1
         self.framePosScale.set(self.frameInd)
         self.boolBottomPressed = True

         print(self.dictParam["zoomEthoLength"])

     def move_frame_back(self):

         self.frameInd = self.frameInd - 1
         self.framePosScale.set(self.frameInd)
         self.boolBottomPressed = True

     def select_zoom(self):

         self.startZoom = int(self.ethoMarker[0][0]/self.posRatio)
         self.endZoom = int(self.ethoMarker[0][1]/self.posRatio)
         self.dictParam = {"zoomEthoLength" : self.endZoom-self.startZoom}
         self.posRatio = int(1000/(self.endZoom-self.startZoom))

         self.framePosScale.configure(to=self.dictParam["zoomEthoLength"])


     def zoom_out(self):

         self.dictParam = {"zoomEthoLength" : 500}
         self.startZoom = 0
         self.endZoom = 500
         self.framePosScale.configure(to=self.dictParam["zoomEthoLength"])
           
         

     def correct_etho(self):


         
         self.correctedEthoRaw = self.ethoRaw 

         startCorr = int((self.ethoMarker[0][0]+0)/self.posRatio)+self.startZoom
         endCorr = int((self.ethoMarker[0][1]+0)/self.posRatio)+self.startZoom

         correctedValue = int(self.entryCorrectIt.get())

         self.ethoRaw[:,startCorr:endCorr] = 0
         self.ethoRaw[correctedValue*1,startCorr:endCorr] = 252
         
         self.ethoRawExp[:,startCorr:endCorr] = 0
         self.ethoRawExp[correctedValue*5+5,startCorr:endCorr] = 252
         
         self.correctedEthoRaw[:,startCorr:endCorr] = 0
         self.correctedEthoRaw[correctedValue*1,startCorr:endCorr] = 252

         self.ethoCurrent[:,startCorr+self.startFrame : endCorr+self.startFrame] = 0
         self.ethoCurrent[correctedValue*1,startCorr+self.startFrame : endCorr+self.startFrame] = 252

         self.ethoVectCorr = etho_mat_2_etho_vect(self.correctedEthoRaw)

         print(self.entryCorrectIt.get())

     def save_labels_and_images(self):

          STimRec = self.bufferObj.get_ST_image_record()

          frameNameStart = str(self.startFrame)
          frameNameEnd = str(self.endFrame)

          with open(self.savedLabelsPath + '\\' + 'ethoLabel' + frameNameStart + '_' + frameNameEnd, "wb") as f:
               pickle.dump(self.ethoVectCorr, f)

          with open(self.savedTrainingDataPath + '\\' + 'STimRec' + frameNameStart + '_' + frameNameEnd, "wb") as f:
               pickle.dump(STimRec, f)     

                                 
 
     def update(self):
         #
         if self.boolBottomPressed == False:

             scalePosInd = self.framePosScale.get()

             if scalePosInd < (self.endFrame - self.startFrame):
              
                 self.frameInd = scalePosInd

         noError = True

         if self.boolDoRun == True:

              self.ethoLength = self.dictParam["zoomEthoLength"]
              self.posRatio = int(1000/self.ethoLength)

              rawFrame = self.bufferObj.get_raw_frame(self.frameInd + self.startZoom)
              rawFrame = cv2.resize(rawFrame,(400,400))

              imST = self.bufferObj.get_ST_image(self.frameInd + self.startZoom)
              imST = cv2.resize(imST,(400,400))

              predLabel, predProb = self.bufferObj.get_predictions(self.frameInd + self.startZoom)
              strLabel = str(predLabel)
              self.labelTextBox['text'] = strLabel
              
              self.ethoVect[0,self.frameInd] = int(predLabel)
              self.ethoRaw[int(predLabel)*1,self.frameInd + self.startZoom] = 150
              self.ethoRawExp[int(predLabel)*5+5 : int(predLabel)*5+8,self.frameInd + self.startZoom] = 150


              self.ethoZoomedExp = self.ethoRawExp[:,self.startZoom:self.endZoom]
              

              self.ethoRawRs = cv2.resize(self.ethoZoomedExp,(1000,100))
              ethoVectRs = cv2.resize(self.ethoCurrent,(1000,100))
              
                    
              if noError == True:
                  self.photo = PIL.ImageTk.PhotoImage(image = PIL.Image.fromarray(rawFrame))
                  self.canvas.create_image(2, 2, image = self.photo, anchor = tkinter.NW)

                  self.imSTtoShow = PIL.ImageTk.PhotoImage(image = PIL.Image.fromarray(imST))
                  self.canvas.create_image(500, 2, image = self.imSTtoShow, anchor = tkinter.NW)

                  self.imEtho = PIL.ImageTk.PhotoImage(image = PIL.Image.fromarray(self.ethoRawRs))
                  self.canvasEtho.create_image(0, 0, image = self.imEtho, anchor = tkinter.NW)

                  self.imEthoVect = PIL.ImageTk.PhotoImage(image = PIL.Image.fromarray(ethoVectRs))
                  self.canvasEtho.create_image(0, 150, image = self.imEthoVect, anchor = tkinter.NW)
                  
                  self.canvasEtho.create_line(self.frameInd*self.posRatio, 0, self.frameInd*self.posRatio, 100, fill="#476042")
                  self.canvasEtho.create_line(self.ethoMarker[0][0]*1, 0, self.ethoMarker[0][0]*1, 100, fill="red")
                  self.canvasEtho.create_line(self.ethoMarker[0][1]*1, 0, self.ethoMarker[0][1]*1, 100, fill="blue")

                  self.canvasEtho.create_line(self.frameInd+self.startFrame+self.startZoom, 150, self.frameInd+self.startFrame+self.startZoom, 250, fill="#476042")

                  

                  self.boolBottomPressed = False
      
         self.window.after(self.delay, self.update)

class BufferRecord:

    def __init__ (self,ethoLength, modelPathName, videoSource=0,startFrame=0,endFrame=50):

        self.cap = cv2.VideoCapture(videoSource)
        self.frameWidth = self.cap.get(cv2.CAP_PROP_FRAME_WIDTH)
        self.frameHeight = self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
        self.modelPathName = modelPathName

        quadrant = 3
        
        self.frRec, self.STRec, self.labelRec, self.probRec = self.run_ABRS_in_window (ethoLength,startFrame,endFrame,quadrant)

        

    def get_raw_frame(self,ind):

        frame = self.frRec[ind,:,:]

        return frame

    def get_ST_image(self,ind):

        im3CRaw = self.STRec[ind,:,:,:]
        
        if np.count_nonzero(im3CRaw[:,:,0])>6400:            
            im3CRaw[:,:,0] = np.zeros((80,80))
        
        if np.count_nonzero(im3CRaw[:,:,1])>800:            
            im3CRaw[:,:,1] = np.zeros((80,80))
        
        rgbArray = np.zeros((80,80,3), 'uint8')
        rgbArray[..., 2] = im3CRaw[:,:,0]
        rgbArray[..., 1] = im3CRaw[:,:,1]
        rgbArray[..., 0] = im3CRaw[:,:,2]

        return rgbArray

    def get_predictions(self,ind):

        label = self.labelRec[ind]
        prob = self.probRec[ind,:]

        return label, prob

    def get_ST_image_record(self):

        return self.STRec 

    def run_ABRS_in_window(self,ethoLength,startFrame,endFrame,fb=0):

        newSize = (400,400)


        if fb>0:
           splitRatio = 2
        if fb==0:
           splitRatio = 1

        kernelSize = (endFrame - startFrame) + 1
        
        smoothingWindow = 89

        windowSize = 100 #size of window for training -- ignore in this version

        winST = 16

        halfWindowSpeed = 15

        movementThreshold = 150

        ind = 0
        beh = 0
        indBehDur = 0

        prevFrame = np.zeros((400,400))
        frRec = np.zeros((16+1,newSize[0]*newSize[1]))

        frCropedRec = np.zeros((kernelSize,int(self.frameHeight/splitRatio),int(self.frameWidth/splitRatio)))
               
        STRec = np.zeros((kernelSize,80,80,3))

        labelRec = np.zeros((kernelSize,1))
        probRec = np.zeros((kernelSize,10))

        model = keras.models.load_model(self.modelPathName)
        model.summary()

        for frameInd in range(startFrame,endFrame,1):

            self.cap.set(1,frameInd)

            ret, frame = self.cap.read() #
                
            #print(np.shape(frame))

            print(frameInd)

            if np.size(np.shape(frame)) != 0:            
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) #
            else:
                gray = np.zeros((int(self.frameHeight/splitRatio), int(self.frameHeight/splitRatio)))

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

            if fb == 0:
                #grayCroped = np.resize(gray,(cropedSize,cropedSize))
                 grayCroped = gray

            frCropedRec[ind,:,:] = grayCroped    
            #grayCropedRS = cv2.resize(grayCroped,(newSize[0],newSize[1]))
            
            currentFrame = rs.astype(float)/1
            #currentFrame = grayCropedRS.astype(float)/1
            
            diffFrame = currentFrame - prevFrame
            prevFrame = currentFrame

            diffFrameAbs = np.absolute(diffFrame)

            frameVect = currentFrame.reshape(1,newSize[0]*newSize[1])
            frameVectFloat = frameVect.astype(float);

            frRecShort = np.delete(frRec, 0, 0)
            frRec = np.vstack((frRecShort,frameVectFloat))

            sumFrRec = np.sum(frRec,0)
            
            posDic, maxMovement, cfrVectRec, frameVectFloatRec = getting_frame_record(frRec, 0, winST,fb)
          
            im3CRaw = create_3C_image (cfrVectRec)
            STRec[ind,:,:,:] = im3CRaw
                
            if np.count_nonzero(im3CRaw[:,:,0])>6400:            
                im3CRaw[:,:,0] = np.zeros((80,80))
                
            if np.count_nonzero(im3CRaw[:,:,1])>800:            
                im3CRaw[:,:,1] = np.zeros((80,80))
                
            rgbArray = np.zeros((80,80,3), 'uint8')
            rgbArray[..., 2] = im3CRaw[:,:,0]
            rgbArray[..., 1] = im3CRaw[:,:,1]
            rgbArray[..., 0] = im3CRaw[:,:,2]
            #im3C = Image.fromarray(rgbArray)

            X_rs = np.zeros((1,80,80,3))
        
            X_rs[0,:,:,:] = im3CRaw

            X = X_rs/256  # normalize

            predictionsProb = model.predict(X)

            predictionLabel = np.zeros((1,np.shape(predictionsProb)[0]))
            predictionLabel[0,:] = np.argmax(predictionsProb,axis=1)

            labelRec[ind] = predictionLabel
            probRec[ind,:] = predictionsProb
                
            ind = ind + 1


        return frCropedRec, STRec, labelRec, probRec      
 
 
 # SET all the values below:


videoSource = 'CantonS_dust_1 19.avi'
modelPathName = 'C:\\Users\\ravbar\\Desktop\\ABRS_Python_GHws1\\modelConv2ABRS_3C_train_with_descendingcombinedwithothers_avi_10'
savedLabelsPath = 'C:\\Users\\ravbar\\Desktop\\ABRS_Python_GHws1\\Labels_and_Training_Data_from_LabelMaker\\labels'
savedTrainingDataPath = 'C:\\Users\\ravbar\\Desktop\\ABRS_Python_GHws1\\Labels_and_Training_Data_from_LabelMaker\\training_data'



dictParam = {"zoomEthoLength" : 1000, "modelPathName" : modelPathName, "videoSource" : videoSource, "savedLabelsPath" : savedLabelsPath, "savedTrainingDataPath" : savedTrainingDataPath}


LabelMakerGUI(tkinter.Tk(), "ABRS Label Maker", dictParam)




