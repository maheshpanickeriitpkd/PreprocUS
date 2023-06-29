# -*- coding: utf-8 -*-
"""
@author: Mahesh Panicker (mahesh@iitpkd.ac.in)
Collaborators: Arpan Tripathi  (tripathiarpan20@gmail.com), Madhavanunni AN  (madhavanunni.an@gmail.com)
"""

#%% Import necessary libraries
import os
import numpy as np
import cv2
import argparse
import torch
from rectLinear import imageExtract
from specProbMap import *


#%% Parsing the necessary inputs
parser = argparse.ArgumentParser("Ultrasound PreProcessing")
parser.add_argument('-s','--IMG_DIR', type=str, default=r'\Data', help="path to input dataset (required)")
parser.add_argument('--IMG_SIZE', type=int, default=256, help="Image Size (default: 128x128)")
parser.add_argument('--encoder_name', type=str, default='resnet34', help="U-net Backbone (default: resnet34)")
parser.add_argument('--TEST_BATCH_SIZE', type=int, default=16, help="batch size of validation data (default: 16)")
parser.add_argument('--PRETRAINED_PATH', type=str, default='./model/uNet.pth', help="path to pretrained weights (default: '../data/bst_model256_fold4_0.977.bin')")
parser.add_argument('--Rectilinear', type=bool, default=True)
parser.add_argument('--PleuraProbMap', type=bool, default=True)
parser.add_argument('--RadonMapHorz', type=bool, default=True)
parser.add_argument('--RadonMapVert', type=bool, default=True)
args = parser.parse_args()

#%% Main function
def main():  
    
    #Loading the model for image region segmentation
    device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = torch.load(args.PRETRAINED_PATH,map_location=device)    
    fullDirName = os.getcwd()+args.IMG_DIR    
    file_names=os.listdir(fullDirName)
    
    if(not(os.path.exists('./Results/ '))):
            os.mkdir('./Results/ ')
    
    for fileIdx in range(len(file_names)):
        print(fileIdx)
        #Read the images
        img_path  = os.path.join(fullDirName, file_names[fileIdx]) 
        image_org     = cv2.imread(img_path, 1)
        orgShape  = np.shape(image_org)
        
        if(image_org.ndim==3):
            image= cv2.cvtColor(image_org, cv2.COLOR_BGR2GRAY) #grayscale conversion        
        
        if (args.Rectilinear):
            image     = cv2.resize(image_org,(args.IMG_SIZE,args.IMG_SIZE)) 
            image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)             
            image     = np.moveaxis(image, -1, 0)        
            image     = np.expand_dims(image,axis=0)
            
            #Convert to tensor for pytorch processing
            imageTensor = torch.from_numpy(image)    
            imageTensor= imageTensor.to(device)      
            
            #Evaluate pytorch model
            model = model.to(device)
            model.eval()
        
            seg_out = model(imageTensor.float())
            seg_out=seg_out.cpu().detach().numpy()
            seg_out = seg_out[0]                  
            seg_out=(seg_out > 0.5).astype(np.int_) 
                        
            Img=np.multiply(seg_out[0],image_gray)  
            ImgTemp=imageExtract(Img,seg_out[0])
            
            image = cv2.resize(ImgTemp,(orgShape[1],orgShape[0])) 
            FileName_Rect = './Results/' + 'Rectilinear_' + file_names[fileIdx]
            cv2.imwrite(FileName_Rect,image)
        
        image_norm = normalise(image)
        
        if (args.PleuraProbMap):
        # Extracting pleural probability maps
            ImgapROI,ImgbpROI,sh,ibs,LP = pleura_prob_map(image, minwl = 10)     
            
            FileName_PPM = './Results/' + 'PleuraProbMap_' + file_names[fileIdx]
            cv2.imwrite(FileName_PPM,image*ImgapROI)
        
        if (args.RadonMapHorz):
            #Performing Radon transform for extracting only horizontal
            irh = radonHorzTrans(image,n = 50,filter='hann')
            irhprod = normalise(irh)*image_norm;
            
            FileName_radonHorz = './Results/' + 'RadonHorz_' + file_names[fileIdx]
            cv2.imwrite(FileName_radonHorz,irhprod*255)        
        
        if (args.RadonMapVert):
            #Performing Radon transform for extracting only vertical structures            
            irv = radonVertTrans(image,n = 50,filter='hann')        
            irvprod = normalise(irv)*image_norm;        
                    
            FileName_radonVert = './Results/' + 'RadonVert_' + file_names[fileIdx]
            cv2.imwrite(FileName_radonVert,irvprod*255)        
        
if __name__ == '__main__':
    main()
