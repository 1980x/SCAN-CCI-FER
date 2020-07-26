'''
Aum Sri Sai Ram
28-06-20
Email: darshangera@sssihl.edu.in
Implementation of Oulucasiadataset class for cross validation training
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
            emo_to_return = 5
        elif emo_to_return == 3:
            emo_to_return = 4
        elif emo_to_return == 4:
            emo_to_return = 1
        elif emo_to_return == 5:
            emo_to_return = 2
        elif emo_to_return == 6:
            emo_to_return = 3

        return emo_to_return


def default_reader(fileList, num_classes, fold, train = 'True'):
    
    imgList = []

    num_per_cls_dict = dict()
    for i in range(0, num_classes):
        num_per_cls_dict[i] = 0
    
    all_identities = [i for i in range(80)] 
    
    lookup = [i for i in range(fold*8,fold*8+8)]
    if train:
       lookup = list(set(all_identities) - set(lookup))
    #print(lookup)
          
    

    with open(fileList, 'r') as fp:
        for line in fp.readlines():              
            imgPath, identity_exp  = line.strip().split(' ')[0].strip(),line.strip().split(' ')[1] #folder/imagename
            expression = int(identity_exp.strip().split('_')[1])#emotion label
            identity =   int(identity_exp.strip().split('_')[0])#identity label
            
            if identity in lookup:
               expression = change_emotion_label_same_as_affectnet(expression)
               if expression >= 0:               #exclude neutral > 0, include neutral >= 0
                  imgList.append([imgPath.strip(), expression])
                  num_per_cls_dict[expression] = num_per_cls_dict[expression] + 1 
               #print(imgPath,identity_exp, identity, expression)
        
        print('Total included ', len(imgList), num_per_cls_dict)
        return imgList,num_per_cls_dict
    
class ImageList(data.Dataset):
    def __init__(self, root, fileList, fold, is_train = True, num_classes = 7,  transform=None, list_reader=default_reader, loader=PIL_loader):
        self.root = root
        self.cls_num = num_classes
        self.fold = fold
        self.imgList, self.num_per_cls_dict =  list_reader(fileList, self.cls_num, self.fold, is_train)
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

   
   rootfolder= '../data/AuthorOluCasia/ourOluCasiatest/'
   filename = '../data/AuthorOluCasia/ourOluCasia_labelstest.txt'
   folds = 10
   classes = 7
   """
   for fold in range(folds):
       print(fold, 'train')
       default_reader( filename, classes, fold,True)
       print(fold, 'test')
       default_reader( filename, classes, fold, False)
   """
   transform = transforms.Compose([transforms.Resize((224, 224)),
                                           transforms.ToTensor()])
   dataset = ImageList(rootfolder, filename, fold = 0, is_train = False, transform=transform)

   fdi = iter(dataset)
   img_list = []
   target_list = []
   for i, data in enumerate(fdi):
       if i < 2:
          print(data[0][0].size(), data[1])
          continue
       else:
          break
   
          


