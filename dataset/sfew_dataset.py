'''
Aum Sri Sai Ram

Implementation of SFEW dataset class

Authors: Darshan Gera and Dr. S. Balasubramanian, SSSIHL
Date: 10-07-2020
Email: darshangera@sssihl.edu.in
'''

import torch.utils.data as data
from PIL import Image, ImageFile
import os
import glob
import os
import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np

from torchvision import transforms

ImageFile.LOAD_TRUNCATED_IAMGES = True

def PIL_loader(path):
    try:
        with open(path, 'rb') as f:
            return Image.open(f).convert('RGB')
    except IOError:
        print('Cannot load image ' + path)

def default_reader(fileList, num_classes):
    imgList = []

    num_per_cls_dict = dict()
    for i in range(0, num_classes):
        num_per_cls_dict[i] = 0


    with open(fileList, 'r') as fp:
        for line in fp.readlines():  
            
            imgPath  = line.strip().split(' ')[0] #folder/imagename
            expression = int(line.strip().split(' ')[1])#emotion label
            imgList.append([imgPath, expression])
            num_per_cls_dict[expression] = num_per_cls_dict[expression] + 1 
        fp.close()
        print('Total included ', len(imgList))
        return imgList,num_per_cls_dict


class ImageList(data.Dataset):
    def __init__(self, root, fileList,  transform=None, list_reader=default_reader, loader=PIL_loader):
        self.root = root
        self.cls_num = 7
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
   rootfolder= '../data/SFEW/Train_Aligned_Faces/'
   filename = '../data/SFEW/sfew_train.txt'


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


