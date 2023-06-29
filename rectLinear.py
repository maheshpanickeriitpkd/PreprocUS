# -*- coding: utf-8 -*-
"""
Created on Sat Jun 17 15:46:48 2023
Adoped from https://github.com/alimuldal/phasepack
@author: Mahesh Panicker (mahesh@iitpkd.ac.in)
Madhavanunni AN (madhavanunni.an@gmail.com)
"""

import numpy as np
from sklearn import linear_model

def indices(a, func):
    return [i for (i, val) in enumerate(a) if func(val)]

def rectConvert(Img,X1,Y1,X2,Y2):
    
    try:
        lr = linear_model.RANSACRegressor(min_samples=25)
        lr.fit(np.reshape(Y1,(np.shape(Y1)[0],1)), np.reshape(X1,(np.shape(X1)[0],1)));
        slope1, intercept1 = float(lr.estimator_.coef_), float(lr.estimator_.intercept_)
    except:
        slope1, intercept1 = np.polyfit(Y1, X1, 1)
        
    
        # plt.plot(Y1,X1,linestyle = 'None',marker='*')
        # plt.plot(Y2,X2,linestyle = 'None',marker='*')
        # plt.plot(np.array(Y1), slope1*np.array(Y1) + intercept1)
        # plt.plot(np.array(Y2), slope2*np.array(Y2) + intercept2)
        # plt.show()

    
    try:
        lr = linear_model.RANSACRegressor(min_samples=25)
        lr.fit(np.reshape(Y2,(np.shape(Y2)[0],1)), np.reshape(X2,(np.shape(X2)[0],1)));
        slope2, intercept2 = float(lr.estimator_.coef_), float(lr.estimator_.intercept_)
    except:
        slope2, intercept2 = np.polyfit(Y2, X2, 1)
    
    # plt.plot(Y1,X1,linestyle = 'None',marker='*')  
    # plt.plot(Y2,X2,linestyle = 'None',marker='*')
    # plt.plot(np.array(Y1), slope1*np.array(Y1) + intercept1)
    # plt.plot(np.array(Y2), slope2*np.array(Y2) + intercept2)
    # plt.show()   
   
    
    line1 = np.int16(slope1*np.arange(0,np.shape(Img)[1])+intercept1)
    line1[np.where(line1<0)]=0
    line2 = np.int16(slope2*np.arange(0,np.shape(Img)[1])+intercept2)
    line2[np.where(line2>255)]=255
    
    ImgNew = np.zeros(np.shape(Img))
    for ii in range(np.shape(Img)[0]):
        ImgNew[ii,:]=np.interp(np.linspace(line1[ii],line2[ii],np.shape(Img)[1]),np.arange(line1[ii],line2[ii]),Img[ii,line1[ii]:line2[ii]])      
    
    return ImgNew

def imageExtract(Img,ImgMask):
    
    ImgMask=ImgMask.astype('uint8')    
    rows,cols = np.shape(ImgMask)

    X1=[]
    Y1=[]
    X2=[]
    Y2=[]
    for ii in range(10,rows-50):
        ImgMaskIndTemp = indices(ImgMask[ii,:], lambda x: x > 0)
        if(np.size(ImgMaskIndTemp)>0):
            X1.append(ImgMaskIndTemp[0])
            X2.append(ImgMaskIndTemp[-1])
            Y1.append(ii)
            Y2.append(ii)
    
    ImgNew = rectConvert(Img,X1,Y1,X2,Y2)    
  
    return ImgNew

def imageExtractFinal(Img,ImgMask):
    ImgMaskIndX = indices(ImgMask[128,:], lambda x: x > 0.8)
    ImgMaskIndY = indices(ImgMask[:,128], lambda x: x > 0.8)
    ImgFinal=cv2.resize(Img[0,ImgMaskIndY[0]:ImgMaskIndY[-1],ImgMaskIndX[0]:ImgMaskIndX[-1],0],(256,256))
    return ImgFinal
