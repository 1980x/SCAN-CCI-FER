'''
Aum Sri Sai Ram
Implementation of CK+ dataset class for cross validation 
Authors: Darshan Gera and Dr. S. Balasubramanian, SSSIHL
Date: 10-07-2020
Email: darshangera@sssihl.edu.in



labels are :
#labels:   0=Neutral 1=anger, 2=contempt, 3=disgust, 4=fear, 5=happy, 6=sadness, 7=surprise

'''

import pickle
import torch.utils.data as data
from PIL import Image, ImageFile
import os
import glob
import os
import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
from torchvision import transforms

def loademotionlabels(): # labels
    d = dict()
    with open('../data/CK+/metafile/CK+last3frameslabels.txt','r') as fp:     # No neutral, only all others
         lines = fp.readlines()
         for line in lines:
             name, label = line.split(' ')
             key = name.strip().split('/')[-1]
             d[key] = label
         #print(len(d.keys()))
    return d

ImageFile.LOAD_TRUNCATED_IAMGES = True

def PIL_loader(path):
    try:
        with open(path, 'rb') as f:
            return Image.open(f).convert('RGB')
    except IOError:
        print('Cannot load image ' + path)


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
            emo_to_return = 0
        elif emo_to_return == 1:
            emo_to_return = 6
        elif emo_to_return == 2:
            emo_to_return = 7
        elif emo_to_return == 3:
            emo_to_return = 5
        elif emo_to_return == 4:
            emo_to_return = 4
        elif emo_to_return == 5:
            emo_to_return = 1
        elif emo_to_return == 6:
            emo_to_return = 2
        elif emo_to_return == 7:
            emo_to_return = 3

        return emo_to_return


def default_reader(fileList, num_classes):
    imgList = []

    num_per_cls_dict = dict()
    for i in range(0, num_classes):
        num_per_cls_dict[i] = 0

    labels_dict = loademotionlabels()
    with open(fileList, 'r') as fp:
        for imgPath in fp.readlines():

            key = imgPath.strip()#[0:8]              
            
            folder, subfolder = key.split('_')[0],key.split('_')[1]
            imgPath = folder+'/'+subfolder+'/'+key
            #print(folder, subfolder, key, imgPath) 
            expression = int(labels_dict[key])            
            expression = change_emotion_label_same_as_affectnet(expression)
            '''
            Since neutral face is missing in list of cross validation meta files so adding manualy below:
            '''  
            neutral_image_name = key[:-6]+'01.png'
            neutral_imgPath = folder+'/'+subfolder+'/'+neutral_image_name
            #print(imgPath,expression, neutral_imgPath, 0)
             
            if expression in [0,1,2,3,4,5,6,7]:
               imgList.append([imgPath, expression])
               num_per_cls_dict[expression] = num_per_cls_dict[expression] + 1 
               
               if [neutral_imgPath,0] not in imgList:# For adding neutral image for each subject
                  imgList.append([neutral_imgPath,0])
                  num_per_cls_dict[0] = num_per_cls_dict[0] + 1 
               
            else:
               continue
 
            
        fp.close()
        print('Total included ', len(imgList), num_per_cls_dict)

        return imgList,num_per_cls_dict
class ImageList(data.Dataset):
    def __init__(self, root, fileList,  transform=None, list_reader=default_reader, loader=PIL_loader):
        self.root = root
        self.cls_num = 8
        self.imgList, self.num_per_cls_dict =  list_reader(fileList, self.cls_num)
        self.transform = transform
        self.loader = loader
        self.fileList  = fileList

    def __getitem__(self, index):
        imgPath, target_expression = self.imgList[index]
        img = self.loader(os.path.join(self.root, imgPath))       
        if self.transform is not None:
            img = self.transform(img)
        
        return  img, target_expression 

    def __len__(self):
        return len(self.imgList)

    def get_cls_num_list(self):
        cls_num_list = []
        for i in range(self.cls_num):
            cls_num_list.append(self.num_per_cls_dict[i])
        return cls_num_list




if __name__=='__main__':

   rootfolder= '../data/CK+/cohn_kanade_images_aligned/'
   filename = '../data/CK+/metafile/cv/train_ids_5.csv'
   default_reader(fileList=filename, num_classes=8)
   assert(False)    
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


