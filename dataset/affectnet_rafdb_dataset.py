'''
Aum Sri Sai Ram
By Darshan on 08-05-20

Combined for training AffectnEt+Rafdb, for testing fedro testing



Reference:
  Mollahosseini, A., Hasani, B. and Mahoor, M.H., 2017. AffectNet: A database for facial expression, valence,
    and arousal computing in the wild. IEEE Transactions on Affective Computing, 10(1), pp.18-31.

0: Neutral, 1: Happiness, 2: Sadness, 3: Surprise, 4: Fear, 5: Disgust, 6: Anger,
7: Contempt, 8: None, 9: Uncertain, 10: No-Face

NOTE: Removing 8: None, 9: Uncertain, 10: No-Face while training for FLATCAM.

classes = {
            0: 'Neutral',
            1: 'Happy',
            2: 'Sad',
            3: 'Surprise',
            4: 'Fear',
            5: 'Disgust',
            6: 'Anger',
            7: 'Contempt'}

No of samples in Manuall annoated set for each of the class are below:
0:74874
1:134415
2:25459
3:14090
4:6378
 5:3803
 6:24882
7:3750

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
    return switcher.get(expression_argument, 0) #default neutral expression

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

def default_reader_rafdb(fileList):
    
    counter_loaded_images_per_label = [0 for _ in range(7)]

    imgList = []
    if fileList.find('train_label.txt') > -1:
 
       fp = open(fileList,'r')

       for names in fp.readlines():
           image_path, target  = names.split(' ')  #Eg. for each entry before underscore lable and afterwards name in 1_fer0034656.png 8 0, 2_fer0033878.png 8 0

           name,ext = image_path.strip().split('.')                #imagename is name.jpg  --->  name_algined.jpg

           image_path  = name + '_aligned.' + ext

           target  =  int(target) - 1 #labels are from 1-7
 
           target = change_emotion_label_same_as_affectnet(target)
           
           counter_loaded_images_per_label[target] += 1 

           imgList.append((image_path,'rafdb',(0,0,0,0), int(target),)) #'0000-bbox' is added to make same number of elements in each tuple from both dataset

       fp.close()
       
       print(fileList, ' has total: ',sum(counter_loaded_images_per_label))
       for i in range(7):
           print('Exp: {} #{} %{:.2f}'.format(get_class(i), counter_loaded_images_per_label[i], 
                                         100.0 * (counter_loaded_images_per_label[i] / sum(counter_loaded_images_per_label))))    
    
       return imgList 
             


def default_reader_affectnet(fileList):
    imgList = []
    if fileList.find('validation.csv')>-1: #hardcoded for Affectnet dataset
       start_index = 0
       max_samples = 100000
    else:
       start_index = 1
       max_samples = 5000


    expression_0 = 0
    expression_1 = 0
    expression_2 = 0
    expression_3 = 0
    expression_4 = 0
    expression_5 = 0
    expression_6 = 0
    expression_7 = 0
    
 
    if fileList.find('training') > -1:     

        fp = open(fileList, 'r')
        for line in fp.readlines()[start_index:]:  #Ist line is header for automated labeled images
            
            imgPath  = line.strip().split(',')[0] #folder/imagename
            (x,y,w,h)  = line.strip().split(',')[1:5]#bounding box coordinates
            
            expression = int(line.strip().split(',')[6])#emotion label
            #print(imgPath, (x,y,w,h), expression)
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

            if expression not in [ 7, 8,9,10]: #Adding only list of first 8 expressions 
               imgList.append([imgPath,'affectnet',(int(x),int(y),int(w),int(h)), expression])
        fp.close()
        print('Number of samples in each class are:')
        print(expression_0, expression_1,expression_2,expression_3,expression_4,expression_5,expression_6)
        print('total included ', len(imgList))
        return imgList


class ImageList(data.Dataset):
    def __init__(self, root, fileList,  transform=None, list_reader=default_reader_affectnet, loader=PIL_loader):
        self.root = root
        if fileList.find('training.csv')>-1:
           self.imgList_affectnet = default_reader_affectnet(fileList)
        
        self.imgList_rafdb = default_reader_rafdb('../data/RAFDB/EmoLabel/train_label.txt')     
        
        self.imgList =  self.imgList_affectnet + self.imgList_rafdb

        self.transform = transform

        self.loader = loader


    def __getitem__(self, index):
        imgPath, dataset, (x,y,w,h), target_expression = self.imgList[index]
 
        #print(imgPath, (x,y,w,h), target_expression, dataset)
        if dataset == 'affectnet':        
           area = (x,y,w,h)    
           img = self.loader(os.path.join(self.root, imgPath))               
           face = img#.crop(area)        
           if self.transform is not None:
              face = self.transform(face)        
           return  face, target_expression
        elif dataset == 'rafdb':        
           img = self.loader(os.path.join('../data/RAFDB/Image/aligned/', imgPath))               
           if self.transform is not None:
              img = self.transform(img)        
           return  img, target_expression
        else:
           print('Error dataset')
 

    def __len__(self):
        return len(self.imgList)





if __name__=='__main__':
   
   filelist = default_reader_affectnet('../data/Affectnetmetadata/training.csv')

   
   rootfolder= '../data/AffectNetdataset/Manually_Annotated_Images/'

   filename = '../data/Affectnetmetadata/training.csv'

   transform = transforms.Compose([transforms.Resize((224, 224)),
                                           transforms.ToTensor()])

   dataset = ImageList(rootfolder, filename, transform)

   fdi = iter(dataset)
   img_list = []
   target_list = []
   for i, data in enumerate(fdi):
       if i < 2:
          print(data[0][0].size(), data[1])
          continue
       else:
          break


