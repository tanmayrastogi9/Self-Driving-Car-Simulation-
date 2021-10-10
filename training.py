# -*- coding: utf-8 -*-
"""
Created on Mon May 17 20:38:57 2021

@author: Tanmay
"""
print('Setting UP')
import os
os.environ[ 'TF_CPP_MIN_LOG LEVEL'] = '3'


import pandas as pd
import numpy as np
from sklearn.utils import shuffle
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import matplotlib.image as mpimg
from imgaug import augmenters as iaa
import cv2
import random
import socketio
import eventlet

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Convolution2D, Flatten, Dense
from tensorflow.keras.optimizers import Adam



def importDataInfo():
    coloums =['Center','Steering','Throttle', 'Brake' , 'Speed']
    data = pd.read_csv('Data/driving_log.csv' , names = coloums)
    print(filename(data['Center'][0]))
    data['Center'] = data['Center'].apply(filename)
    print('Total Images Imported:', data.shape[0])
    return data 


def filename(filepath) :
    return filepath.split('\\')[-1]


def balanceData(data, display=True):
    nBins = 31
    samplesperbin= 1000
    hist, bins = np.histogram(data['Steering'],nBins)
    if display:
        center = (bins[:-1] + bins[1:])*0.5
        print(center)
        plt.bar(center, hist, width =0.06)
        plt.plot ((-1,1) ,(samplesperbin, samplesperbin))
        plt.show()
          
    removeIndexList = []
    for j in range(nBins):    
        binDataList = []    
        for i in range (len (data['Steering'])):    
            if data['Steering'][i] >= bins[j] and data['Steering'][i]<= bins[j+1]:    
               binDataList.append(i)    
        binDataList= shuffle(binDataList)    
        binDataList= binDataList[samplesperbin:]    
        removeIndexList.extend(binDataList)        
    print('removed images :' ,len(removeIndexList))
    data.drop(data.index[removeIndexList] ,inplace = True)
    print('Reamining image : ' , len(data))
    
    if display:
        hist, _ =  np.histogram(data['Steering'], nBins)
        plt.bar(center ,hist,width = 0.06)
        plt.plot((-1,1),(samplesperbin ,samplesperbin)) 
        plt.show()
    
    return data 



def loadData(path , data):
    imagesPath = []
    steering = []
    for i in range (len (data)):
        indexedData  = data.iloc[i] 
        #print(indexedData)
        imagesPath.append(os.path.join(path , 'IMG', indexedData[0]))
        steering.append(float(indexedData[3]))
    imagesPath = np.asarray(imagesPath)
    steering = np.asarray(steering)
    return imagesPath, steering



def augmentImage (imgPath, steering):
    img=mpimg.imread (imgPath)
    print(np.random.rand(), np.random. rand(), np.random.rand(),np.random.rand())
    # PAN
    if np.random.rand() < 0.5:
        pan = iaa.Affine(translate_percent={'x':(-0.1,0.1), "y":(-0.1,0.1)})
        img= pan.augment_image (img)
    #ZOOM
    if np.random.rand() < 0.5:
        zoom = iaa.Affine(scale=(1,1.2))
        img = zoom.augment_image(img)
    # BRIGHTNESS
    if np.random.rand() < 0.5:
        brightness = iaa.Multiply((0.4,1.2))
        img= brightness.augment_image(img)
    ##FLIP
    if np.random.rand() < 0.5:
        img = cv2.flip(img, 1)
        steering = -steering
        
    
    return img , steering


def preProcessing (img):
    img  = img[60:135 ,:, :]
    img = cv2.cvtColor(img, cv2.COLOR_RGB2YUV)
    img = cv2.GaussianBlur(img,(3,3) ,0)
    img = cv2.resize(img, (200 , 66))
    img  = img /255
    
    return img

def batchGen (imagesPath, steeringList, batchSize, trainFlag):
    while True:
        imgBatch = []
        steeringBatch = []
        for i in range(batchSize):
            index = random.randint(0 ,  len(imagesPath)-1)
            if trainFlag:
                img, steering = augmentImage(imagesPath[index],steeringList[index])
            else:
                img=mpimg.imread(imagesPath[index])
                steering  = steeringList[index] 
            img = preProcessing(img)
            imgBatch.append(img)
            steeringBatch.append(steering)
            yield (np.asarray(imgBatch), np.asarray(steeringBatch))
    


def creteModel():
    model = Sequential()
    
    model.add(Convolution2D(24, (5,5) ,(2,2) , input_shape= (66,200,3), activation='elu'))
    model.add(Convolution2D(36, (5,5) ,(2,2) , activation='elu'))
    model.add(Convolution2D(48, (5,5), (2,2), activation='elu'))
 
    model.add(Convolution2D(64, (3,3)  ,activation='elu'))
    model.add(Convolution2D(64, (3,3)  ,activation='elu'))
 
    model.add(Flatten())
    model.add(Dense(100, activation='elu'))
    model.add(Dense(50, activation='elu')) 
    model.add(Dense(10, activation='elu'))
    
    model.add(Dense(1))

    model.compile(Adam(lr=0.0001) ,loss= 'mse')


    return model






#-----------------------------------------------------------------------------
#1
data = importDataInfo()

#2
data = balanceData(data ,display = True )

#3
imagepath,steer=loadData("Data", data)
print(imagepath[0] , steer[0])

#4
XTrain, xVal, yTrain, yVal = train_test_split (imagepath, steer,test_size=0.2, random_state=5)
print('total length Xtrain : ' ,len(XTrain))
print('total length xVal : ' ,len(xVal))
print('total length yTrain : ' ,len(yTrain))
print('total length yVal : ' ,len(yVal))


#img , st  = augmentImage("test.jpg" , 0)
#plt.imshow(img)
#plt.show()

#imgRe = preProcessing(mpimg.imread('test.jpg'))
#plt.imshow(imgRe)
#plt.show()


#5

model = creteModel()
model.summary()


#6
model_fit = model.fit(batchGen(XTrain, yTrain, 10, 1),steps_per_epoch=50,epochs=10,
          validation_data=batchGen(xVal, yVal, 10, 0) ,validation_steps=50)

#7
model.save('model.h5')
print("Model Saved Successful")

plt.plot(model_fit.history['loss'])
plt.plot(model_fit.history['val_loss'])
plt.legend(['Training Validation'])
plt.ylim([0,0.05])
plt.title('Loss')
plt.xlabel('Epoch')
plt.show()



















