'''
Aum Sri Sai Ram

By Darshan on 06-05-20

Dataset class for FERPLUS : 8 Basic emotions

Code ref:
  i)` Code based on https://github.com/microsoft/FERPlus.
 
'''
import os, sys, shutil, csv
import random as rd
from os import listdir
from PIL import Image
import numpy as np
import random
import torch
import torch.nn.functional as F
import torch.utils.data as data
import pdb
from PIL import Image, ImageFile
import torchvision.transforms as transforms

ImageFile.LOAD_TRUNCATED_IAMGES = True

def PIL_loader(path):
    try:
        with open(path, 'rb') as f:
            return Image.open(f).convert('RGB')
    except IOError:
        print('Cannot load image ' + path)


def make_emotion_compatible_to_affectnet(emo_to_return):
        """
        Parse labels to make them compatible with AffectNet.
        :param idx:
        :return:
        """

        if emo_to_return == 2:
            emo_to_return = 3
        elif emo_to_return == 3:
            emo_to_return = 2
        elif emo_to_return == 4:
            emo_to_return = 6
        elif emo_to_return == 6:
            emo_to_return = 4

        return emo_to_return


def _process_data( emotion_raw, mode): #return emotion based on the mode
        '''
        Based on https://arxiv.org/abs/1608.01041, we process the data differently depend on the training mode:
        Majority: return the emotion that has the majority vote, or unknown if the count is too little.
        Probability or Crossentropty: convert the count into probability distribution.abs
        Multi-target: treat all emotion with 30% or more votes as equal.
        '''
        size = len(emotion_raw)
        #print('raw emotion',emotion_raw)
        emotion_unknown     = [0.0] * size
        emotion_unknown[-2] = 1.0

        # remove emotions with a single vote (outlier removal)
        for i in range(size):
            if emotion_raw[i] < 1.0 + sys.float_info.epsilon:
                emotion_raw[i] = 0.0

        sum_list = sum(emotion_raw)
        emotion = [0.0] * size

        if mode == 'majority':
            # find the peak value of the emo_raw list
            maxval = max(emotion_raw)
            if maxval > 0.5*sum_list:
                emotion[np.argmax(emotion_raw)] = maxval
            else:
                emotion = emotion_unknown   # force setting as unknown
        elif (mode == 'probability') or (mode == 'crossentropy'):
            sum_part = 0
            count = 0
            valid_emotion = True
            while sum_part < 0.75 * sum_list and count < 3 and valid_emotion:
                maxval = max(emotion_raw)
                for i in range(size):
                    if emotion_raw[i] == maxval:
                        emotion[i] = maxval
                        emotion_raw[i] = 0
                        sum_part += emotion[i]
                        count += 1
                        if i >= 8:  # unknown or non-face share same number of max votes
                            valid_emotion = False
                            if sum(emotion) > maxval:   # there have been other emotions ahead of unknown or non-face
                                emotion[i] = 0
                                count -= 1
                            break
            if sum(emotion) <= 0.5 * sum_list or count > 3: # less than 50% of the votes are integrated, or there are too many emotions, we'd better discard this example
                emotion = emotion_unknown   # force setting as unknown
                
        return emotion



def default_reader(fileList, mode, num_expression=8):
    imgList = []
    i = 0
    
    if fileList.find('occlusion_list.txt') > -1:
       fp = open(fileList,'r')
       for names in fp.readlines():
           target, image_path  = names.split('_')  #Eg. for each entry before underscore lable and afterwards name in 1_fer0034656.png 8 0, 2_fer0033878.png 8 0
           image_path = image_path.strip().split(' ')[0]
           imgList.append((image_path, int(target)))
       return imgList 
             
    elif  fileList.find('pose') > -1:
       fp = open(fileList,'r')
       for names in fp.readlines():
           target, image_path  = names.split('/')  #Eg. for each entry before underscore lable and afterwards name in 1/fer0034656.jpg
           image_path = image_path.strip().replace('.jpg','.png') 
           #print(target,image_path)
           imgList.append((image_path, int(target)))
       return imgList     
    else:#test/train/validation.csv
        with open(fileList) as csvfile:
              emotion_label = csv.reader(csvfile)
              for row in emotion_label:
                  image_path = row[0]
                  emotion_raw = list(map(float, row[2:len(row)]))             
                  emotion = _process_data(emotion_raw, mode)                  
                  idx = np.argmax(emotion) 
                  if idx < num_expression: # not unknown or non-face 
                     emotion = emotion[:-2]
                     #print(emotion)
                     emotion = [float(i)/sum(emotion) for i in emotion]              
                     imgList.append((image_path,emotion))
             
        return imgList

class ImageList(data.Dataset):
    def __init__(self, root, fileList, transform=None, num_expressions = 8, list_reader=default_reader, loader = PIL_loader, mode = 'majority'):
        self.root = root
        self.transform = transform
        self.loader = loader
        self.fileList  = fileList
        self.training_mode = mode
        
        self.num_expression =  num_expressions

        self.imgList = list_reader(fileList, self.training_mode, self.num_expression)
        self.num_per_cls_dict = self.get_class_wise_count()
        #print('checking class wise list: ', self.num_per_cls_dict)

    def get_class_wise_count(self):
        num_per_cls_dict = dict()
        for i in range(0, self.num_expression):
            num_per_cls_dict[i] = 0

        for _,target in self.imgList:
            if  not self.fileList.find('occlusion_list.txt') > -1 and not self.fileList.find('pose') > -1:
                target_int = self._process_target(target)
                target_int = make_emotion_compatible_to_affectnet(target_int)
            else:
                target_int = target
                target_int = make_emotion_compatible_to_affectnet(target_int)

            num_per_cls_dict[target_int] = num_per_cls_dict[target_int] + 1 
        return num_per_cls_dict
            
    def _process_target(self, target):
        '''
        Based on https://arxiv.org/abs/1608.01041 the target depend on the training mode.
        Majority or crossentropy: return the probability distribution generated by "_process_data"
        Probability: pick one emotion based on the probability distribtuion.
        Multi-target: not implemented
        '''
        if self.training_mode == 'majority' or self.training_mode == 'crossentropy':
            target = np.argmax(target)
            return target
        elif self.training_mode == 'probability':            
            idx = np.random.choice(len(target), p=target)
            return idx
        
    def __getitem__(self, index):

        imgPath, target_expression = self.imgList[index]

        crops = []

        img = self.loader(os.path.join(self.root, imgPath))

        if self.transform is not None:
               img = self.transform(img)
        
        if  not self.fileList.find('occlusion_list.txt') > -1 and not self.fileList.find('pose') > -1:
            target_expression = self._process_target(target_expression) #Gives emotion label as integer based on mode

        target_expression = make_emotion_compatible_to_affectnet(target_expression) #converts into emtion category same as that of affectnet

        return img, target_expression

    def __len__(self):
        return len(self.imgList)

    def get_cls_num_list(self):
        cls_num_list = []
        for i in range(self.num_expression):
            cls_num_list.append(self.num_per_cls_dict[i])
        return cls_num_list


if __name__=='__main__':
  
   imagesize =  224
   transform = transforms.Compose([transforms.Resize((imagesize,imagesize)), transforms.ToTensor()])

   mode = 'majority'
  
   dataset = ImageList(root='../data/FERPLUS/Dataset/Images/FER2013Train_aligned/', fileList ='../data/FERPLUS/Dataset/Labels/FER2013Train/label.csv',transform = transform,mode=mode )
   '''
   dataset = ImageList(root='../data/FERPLUS/Dataset/Images/FER2013Test_aligned/', fileList ='../data/FERPLUS/Dataset/Labels/FER2013Test/pose_30_list.txt',transform = transform, mode =mode)

   dataset = ImageList(root='../data/FERPLUS/Dataset/Images/FER2013Test_aligned/', fileList ='../data/FERPLUS/Dataset/Labels/FER2013Test/pose_45_list.txt',transform = transform, mode =mode)

   dataset = ImageList(root='../data/FERPLUS/Dataset/Images/FER2013Test_aligned/', fileList ='../data/FERPLUS/Dataset/Labels/FER2013Test/occlusion_list.txt',transform = transform, mode =mode)
   '''
   
   fdi = iter(dataset)
   for i, data in enumerate(fdi):
        if i < 2:
           print(i, data[0].size(),  data[1])
        else:
           break

   


