'''
Aum Sri Sai Ram

Implementation of Affectnet dataset class

Authors: Darshan Gera and Dr. S. Balasubramanian, SSSIHL
Date: 10-07-2020
Email: darshangera@sssihl.edu.in


Reference:
1. Mollahosseini, A., Hasani, B. and Mahoor, M.H., 2017. "AffectNet: A database for facial expression, valence,
    and arousal computing in the wild". IEEE Transactions on Affective Computing, 10(1), pp.18-31.

Labels: 0: Neutral, 1: Happiness, 2: Sadness, 3: Surprise, 4: Fear, 5: Disgust, 6: Anger, 7: Contempt, 8: None, 9: Uncertain, 10: No-Face

No of samples in Manually annoated set for each of the class are below:
0:74874 1:134415 2:25459 3:14090 4:6378 5:3803 6:24882 7:3750 

2. For occlusion, pose30 and 45 datasets refer to https://github.com/kaiwang960112/Challenge-condition-FER-dataset based on 
Kai Wang, Xiaojiang Peng, Jianfei Yang, Debin Meng, and Yu Qiao , Shenzhen Institutes of Advanced Technology, Chinese Academy of Sciences
{kai.wang, xj.peng, db.meng, yu.qiao}@siat.ac.cn
"Region Attention Networks for Pose and Occlusion Robust Facial Expression Recognition".
'''


import torch.utils.data as data
from PIL import Image, ImageFile
import os
import glob
import os
import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
import matplotlib.image as mpimg
import pandas as pd
import cv2
import matplotlib.pyplot as plt
from skimage import io
from torchvision import transforms
import random
ImageFile.LOAD_TRUNCATED_IAMGES = True

def PIL_loader(path):
    try:
        with open(path, 'rb') as f:
            return Image.open(f).convert('RGB')
    except IOError:
        print('Cannot load image ' + path)

def switch_expression(expression_argument):
    switcher = {
         0:'neutral',
         1:'Happiness',
          2: 'Sadness',
        3: 'Surprise',
4: 'Fear', 5: 'Disgust', 6: 'Anger',
7: 'Contempt', 8: 'None', 9: 'Uncertain', 10: 'No-Face'
    }
    return switcher.get(expression_argument, 0) 

def default_reader(fileList, num_classes):
    imgList = []
    if fileList.find('validation.csv')>-1: 
       start_index = 0
       max_samples = 100000
    else:
       start_index = 1
       max_samples = 20000 


    num_per_cls_dict = dict()
    for i in range(0, num_classes):
        num_per_cls_dict[i] = 0

    if num_classes == 7:
       exclude_list = [7, 8,9,10]
    else:
       exclude_list = [8,9,10]

    expression_0 = 0
    expression_1 = 0
    expression_2 = 0
    expression_3 = 0
    expression_4 = 0
    expression_5 = 0
    expression_6 = 0
    expression_7 = 0

    '''
    Below Ist two options for occlusion and pose case and 3rd one for general
    '''
    f = open('../data/Affectnetmetadata/validation.csv','r')
    lines = f.readlines()

    random.shuffle(lines) 

    if fileList.find('occlusion') > -1:
       fp = open(fileList,'r')
       for names in fp.readlines():
           _, target, image_path,_  = names.split('/') 
           image_path = image_path.strip()
    
           for line in lines:
               if line.find(image_path)>-1:
                  
                  imgPath  = line.strip().split(',')[0] 
                  (x,y,w,h)  = line.strip().split(',')[1:5]
            
                  expression = int(line.strip().split(',')[6])  
                  if expression not in exclude_list:
                     imgList.append([imgPath,(int(x),int(y),int(w),int(h)), expression]) 
                     num_per_cls_dict[expression] = num_per_cls_dict[expression] + 1                     
       fp.close()
       return imgList, num_per_cls_dict 
    elif fileList.find('pose') > -1:
       fp = open(fileList,'r')
       for names in fp.readlines():
           target, image_path  = names.split('/')
           image_path = image_path.strip()  
           for line in lines:
               if line.find(image_path) > -1:                  
                  imgPath  = line.strip().split(',')[0] 
                  (x,y,w,h)  = line.strip().split(',')[1:5]
                  expression = int(line.strip().split(',')[6])  
                  if expression not in exclude_list: 
                     imgList.append([imgPath,(int(x),int(y),int(w),int(h)), expression])
                     num_per_cls_dict[expression] = num_per_cls_dict[expression] + 1
        
       fp.close()
       return imgList, num_per_cls_dict 
    
             
    else:   #training or validation affectnet set

        fp = open(fileList, 'r')
        for line in fp.readlines()[start_index:]:  
            
            imgPath  = line.strip().split(',')[0] 
            (x,y,w,h)  = line.strip().split(',')[1:5]
            
            expression = int(line.strip().split(',')[6])

            if expression == 0:
               expression_0 = expression_0 + 1            
               if expression_0 > max_samples:
                  continue
  
            if expression == 1:
               expression_1 = expression_1 + 1
               if expression_1 > max_samples:
                  continue  

            if expression == 2:
               expression_2 = expression_2 + 1
               if expression_2 > max_samples:
                  continue  

            if expression == 3:
               expression_3 = expression_3 + 1
               if expression_3 > max_samples:
                  continue  

            if expression == 4:
               expression_4 = expression_4 + 1
               if expression_4 > max_samples:
                  continue  

            if expression == 5:
               expression_5 = expression_5 + 1
               if expression_5 > max_samples:
                  continue  

            if expression == 6:
               expression_6 = expression_6 + 1
               if expression_6 > max_samples:
                  continue  

            if expression == 7:
               expression_7 = expression_7 + 1
               if expression_7 > max_samples:
                  continue  
            #Adding only list of first 8 expressions 
            if expression not in exclude_list: 
               imgList.append([imgPath,(int(x),int(y),int(w),int(h)), expression])
               num_per_cls_dict[expression] = num_per_cls_dict[expression] + 1 
        fp.close()
        return imgList,num_per_cls_dict


class ImageList(data.Dataset):
    def __init__(self, root, fileList, num_classes=7,  transform=None, list_reader=default_reader, loader=PIL_loader):
        self.root = root
        self.cls_num = num_classes
        self.imgList, self.num_per_cls_dict =  list_reader(fileList, self.cls_num)
        self.transform = transform
        self.loader = loader
        self.fileList  = fileList

    def __getitem__(self, index):
        imgPath, (x,y,w,h), target_expression = self.imgList[index]

        face = self.loader(os.path.join(self.root, imgPath))       
        if self.transform is not None:
            face = self.transform(face)
        
        return  face, target_expression 

    def __len__(self):
        return len(self.imgList)

    def get_cls_num_list(self):
        cls_num_list = []
        for i in range(self.cls_num):
            cls_num_list.append(self.num_per_cls_dict[i])
        return cls_num_list


