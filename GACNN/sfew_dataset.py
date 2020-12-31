'''
Aum Sri Sai Ram
Authors: Darshan Gera and Dr. S. Balasubramanian, SSSIHL
Date: 18-12-2020
Email: darshangera@sssihl.edu.in

It returns 24-landmarks points  along with image and label for SFEW dataset 
'''


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
from PIL import ImageDraw
from PIL import ImageFont
import numpy as np
from numpy import linspace
from matplotlib import cm
import math

from torchvision import transforms

ImageFile.LOAD_TRUNCATED_IAMGES = True

def PIL_loader(path):
    try:
        with open(path, 'rb') as f:
            return Image.open(f).convert('RGB')
    except IOError:
        print('Cannot load image ' + path)
        
def convert68to24(landmarks_68 ,input_imgsize, target_imgsize):
    '''
    a) Occlusion Aware Facial Expression Recognition Using CNN With Attention Mechanism
    b)  https://github.com/mysee1989/PG-CNN/blob/master/convert_point/pts68_24

    Out of 68 points : 24 are recomputed along with score as minimum of points considered
    '''
    landmarks_68 = np.transpose(landmarks_68)
    #print(landmarks_68.shape) #np array of size 68x3
   
    landmarks_24 = []

    #16 standard landmark points from eyebrow, eyes, nose and mouth
    single_points = [19, 22, 23, 26, 39, 37, 44, 46, 28, 30, 49, 51, 53, 55, 59, 57] 
    # 2-Point from left eye, 2 from right eye, next left cheek and right cheek = 6 points for averaging
    double_points = [
            [20, 38],
            [25, 45],
            [41, 42],
            [47, 48],
            [18, 59],
            [27, 57]
            ]  
    # 2 more points at offfset from left mouth corner:49 and right mouth corner:55

    
    #First add 16
    for index in single_points:
        landmark = landmarks_68[index-1].reshape(1,3)    
        landmarks_24.append(landmark)
    #print(landmarks_24, landmarks_24[0].shape)
 
    #Add average 6    
    for ele in double_points:
        point1 = landmarks_68[ele[0]-1][:2]
        score1 = landmarks_68[ele[0]-1][2]
        point2 = landmarks_68[ele[1]-1][:2]
        score2 = landmarks_68[ele[1]-1][2]
        midpoint = np.mean(np.array([point1, point2]), axis=0).reshape(1,2)
        score = np.array(min(score1,score2)).reshape(1,1)
        #print(point1,point2, midpoint.shape, (score).shape )
        midpoint_score = np.append(midpoint,score,axis=1).reshape(1,3)
        landmarks_24.append(midpoint_score)
    

    #add last 2 from mouth corners
    #offset is 16 for 256x256 image, so 
    offset =  int((16.0 * input_imgsize) / 256.0) 
    #print(offset)
    # 23rd point: for left mouth offset point 
    left_mouth_corner_point = landmarks_68[49-1,:2] 
    left_mouth_corner_offset_point = left_mouth_corner_point - offset
    left_mouth_corner_offset_point_score = landmarks_68[49-1,2].reshape(1,1)
  
    left_mouth_corner_new = np.append(left_mouth_corner_offset_point.reshape(1,2),left_mouth_corner_offset_point_score,axis=1).reshape(1,3)
    right_mouth_corner_point = landmarks_68[55-1,:2]
    right_mouth_corner_offset_point = np.array([right_mouth_corner_point[0]-offset, right_mouth_corner_point[1]+offset])
    right_mouth_corner_offset_point_score = landmarks_68[55-1,2].reshape(1,1)
    right_mouth_corner_new = np.append(right_mouth_corner_offset_point.reshape(1,2),right_mouth_corner_offset_point_score,axis=1).reshape(1,3)
    #print(right_mouth_corner_new, left_mouth_corner_new)
    #[IMP: this 4 for 23rd and 24th point is chosen depending on size of image. for 256x256 size image it was 16]
    # 24th point: for right mouth offset point

    landmarks_24.append(left_mouth_corner_new)
    landmarks_24.append(right_mouth_corner_new)
    #print(landmarks_24, landmarks_24[0].shape)


    landmarks_24 = np.asarray(landmarks_24, np.float32).reshape(24,3)
    
    landmarks_24_scaled = landmarks_24 * target_imgsize  / (input_imgsize)
    
    
    #print(landmarks_24.shape )
    return landmarks_24_scaled#landmarks_24

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
        
def default_reader(fileList,  landmarksfile, num_expression = 7 ,input_imgsize = 100, target_imgsize = 224):

    with open(landmarksfile, 'rb') as fp:
        landmarks_dict = pickle.load(fp)

    #print(landmarks_dict['train_05374_aligned.jpg'])  #Test
    
    counter_loaded_images_per_label = [0 for _ in range(7)]

    imgList = []
  
    fp = open(fileList,'r')

    for line in fp.readlines():
           image_path  = line.strip().split(' ')[0] #folder/imagename
           expression = int(line.strip().split(' ')[1])#emotion label
           target = change_emotion_label_same_as_affectnet(expression)

           if landmarks_dict.get(image_path,None) is None:
                     print(target,image_path) #Missing image
                     continue
           else:
                     landmarks_68 = landmarks_dict[image_path]
                     
           landmarks = convert68to24(landmarks_68 ,input_imgsize, target_imgsize) #Recompute 24 from 68

           counter_loaded_images_per_label[target] += 1 

           landmarks_list = [(int(landmarks[i][0]),int(landmarks[i][1])) for i in range(0,24)]
           imgList.append((image_path, int(target), landmarks_list))
           #print(imgList)
           
    fp.close()
    print(fileList, ' has total: ',sum(counter_loaded_images_per_label))
    return imgList 
             
                     
    
class ImageList(data.Dataset):
    def __init__(self, root, fileList, landmarksfile, input_imgsize = 143,
                               target_imgsize = 28, transform = None, list_reader = default_reader, loader=PIL_loader):
        self.root = root
        self.input_imgsize = input_imgsize
        self.target_imgsize = target_imgsize
        self.imgList = list_reader(fileList, landmarksfile, input_imgsize = input_imgsize  , target_imgsize = target_imgsize )
        self.transform = transform
        self.loader = loader
        self.is_save = False
        self.totensor = transforms.ToTensor()
        self.score_threshold = 0.6
 
    def __getitem__(self, index):
        imgPath, target_expression, landmarks_24 = self.imgList[index]
        #print(imgPath, target_expression)
        img = self.loader(os.path.join(self.root, imgPath))

        
        if self.transform is not None:
            img = self.transform(img)

        return img, target_expression,  torch.tensor(landmarks_24)
        
        
    def __len__(self):
        return len(self.imgList)




if __name__=='__main__':
   rootfolder= '../data/SFEW/Train_Aligned_Faces_SAN/'
   filename = '../data/SFEW/sfew_train.txt'    

   
   imagesize =  224
   transform = transforms.Compose([transforms.Resize((imagesize,imagesize)), transforms.ToTensor()])
   
   dataset = ImageList(root=rootfolder, fileList = filename, 
                   landmarksfile = '../data/SFEW/sfew_train_landmarks_scores.pkl', transform = transform     )

   fdi = iter(dataset)
   for i, data in enumerate(fdi):
        if i < 1:
           print('loading ',i,  data[0].size(),data[1], data[2])
        else:
           break