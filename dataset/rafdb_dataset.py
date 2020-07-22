"""
Aum Sri Sai Ram
By DG on 06-06-2020

Dataset class for RAFDB : 7 Basic emotions

Purpose: To return images from RAFDB dataset

Output:  bs x c x w x h        
            
"""

import torch.utils.data as data
from PIL import Image, ImageFile
import os
import pickle
import numpy as np
import matplotlib.image as mpimg
import pandas as pd
import cv2
import matplotlib.pyplot as plt
import torch
from torchvision import transforms
from torchvision.utils import make_grid
ImageFile.LOAD_TRUNCATED_IAMGES = True
import random as rd 

def PIL_loader(path):
    try:
        with open(path, 'rb') as f:
            return Image.open(f).convert('RGB')
    except IOError:
        print('Cannot load image ' + path)

def default_reader(fileList):
    #print(fileList)
    counter_loaded_images_per_label = [0 for _ in range(7)]

    num_per_cls_dict = dict()
    for i in range(0, 7):
        num_per_cls_dict[i] = 0

    imgList = []
    if fileList.find('occlusion_list.txt') > -1:
       fp = open(fileList,'r')
       for names in fp.readlines():
           image_path, target, _  = names.split(' ')  #Eg. test_0025_aligned 3 3 #name, Ist target, 2nd target
           image_path = image_path.strip()+'.jpg'
           target = int(target) 
           target = change_emotion_label_same_as_affectnet(target)
           num_per_cls_dict[target] = num_per_cls_dict[target] + 1 
           imgList.append((image_path, target))           
       return imgList ,    num_per_cls_dict
             
    elif  fileList.find('pose') > -1:
       fp = open(fileList,'r')
       for names in fp.readlines():
           target, image_path  = names.split('/')  #Eg. for each entry before underscore lable and afterwards name in 1/fer0034656.jpg
           image_path = image_path.strip()
           #print(target,image_path)
           target = int(target) 
           target = change_emotion_label_same_as_affectnet(target)
           num_per_cls_dict[target] = num_per_cls_dict[target] + 1 
           imgList.append((image_path, target))
       return imgList ,    num_per_cls_dict 
    else:#test/train/validation.csv
 
       fp = open(fileList,'r')

       for names in fp.readlines():
           image_path, target  = names.split(' ')  #Eg. for each entry before underscore lable and afterwards name in 1_fer0034656.png 8 0, 2_fer0033878.png 8 0

           name,ext = image_path.strip().split('.')                #imagename is name.jpg  --->  name_algined.jpg

           image_path  = name + '_aligned.' + ext

           target  =  int(target) - 1 #labels are from 1-7
 
           target = change_emotion_label_same_as_affectnet(target)
           
           counter_loaded_images_per_label[target] += 1 

           num_per_cls_dict[target] = num_per_cls_dict[target] + 1 

           imgList.append((image_path, int(target)))

       fp.close()
       '''
       print(fileList, ' has total: ',sum(counter_loaded_images_per_label))
       for i in range(7):
           print('Exp: {} #{} %{:.2f}'.format(get_class(i), counter_loaded_images_per_label[i], 
                                         100.0 * (counter_loaded_images_per_label[i] / sum(counter_loaded_images_per_label))))    
       '''
       return imgList, num_per_cls_dict
             
'''
RAF-DB Labels : 1-7 made 0-6
def get_class(idx):  #class expression label
        classes = {
           0: 'Surprise',
           1: 'Fear',
           2: 'Disgust',
           3: 'Happiness',
           4: 'Sadness',
           5: 'Anger',
           6: 'Neutral'
        }

        return classes[idx]
'''

#Affectnet labels
def get_class(idx):
        classes = {
            0: 'Neutral',
            1: 'Happy',
            2: 'Sad',
            3: 'Surprise',
            4: 'Fear',
            5: 'Disgust',
            6: 'Anger',
            7: 'Contempt'}

        return classes[idx]

def change_emotion_label_same_as_affectnet(emo_to_return):
        """
        Parse labels to make them compatible with AffectNet.  
        #https://github.com/siqueira-hc/Efficient-Facial-Feature-Learning-with-Wide-Ensemble-based-Convolutional-Neural-Networks/blob/master/model/utils/udata.py
        """

        if emo_to_return == 0:
            emo_to_return = 3
        elif emo_to_return == 1:
            emo_to_return = 4
        elif emo_to_return == 2:
            emo_to_return = 5
        elif emo_to_return == 3:
            emo_to_return = 1
        elif emo_to_return == 4:
            emo_to_return = 2
        elif emo_to_return == 5:
            emo_to_return = 6
        elif emo_to_return == 6:
            emo_to_return = 0

        return emo_to_return



class ImageList(data.Dataset):
    def __init__(self, root, fileList,  transform=None, list_reader=default_reader, loader=PIL_loader):
        self.root = root
        self.cls_num = 7
        self.imgList, self.num_per_cls_dict = list_reader(fileList)
        self.transform = transform
        self.loader = loader
        self.is_save = True
        self.totensor = transforms.ToTensor()


    def __getitem__(self, index):
        imgPath, target_expression = self.imgList[index]
        #print(imgPath, target_expression)
        img = self.loader(os.path.join(self.root, imgPath))
        if self.transform is not None:
            img = self.transform(img)
        return img, target_expression

    def __len__(self):
        return len(self.imgList)

    def get_cls_num_list(self):
        cls_num_list = []
        for i in range(self.cls_num):
            cls_num_list.append(self.num_per_cls_dict[i])
        return cls_num_list

if __name__=='__main__':
   testlist = default_reader('../data/RAFDB/EmoLabel/val_raf_db_list_pose_45.txt')
   imagesize =  224
   transform = transforms.Compose([transforms.Resize((imagesize,imagesize)), transforms.ToTensor()])
   for i in range(20):
       print(testlist[i])
   
   dataset = ImageList(root='../data/RAFDB/Image/aligned/', fileList ='../data/RAFDB/EmoLabel/val_raf_db_list_pose_45.txt', transform = transform     )

   fdi = iter(dataset)
   for i, data in enumerate(fdi):
        if i < 1:
           print(' ', data[0].size(),data[1] )
        else:
           break
   
