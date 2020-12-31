'''
Authors: Darshan Gera and Dr. S. Balasubramanian, SSSIHL
Date: 18-12-2020
Email: darshangera@sssihl.edu.in

It returns 24-landmarks points  along with image and label for FERPlus dataset

Reference:          
         i) https://github.com/Microsoft/FERPlus
         ii) https://github.com/siqueira-hc/Efficient-Facial-Feature-Learning-with-Wide-Ensemble-based-Convolutional-Neural-Networks
         iii) https://github.com/mysee1989/PG-CNN/blob/master/convert_point/pts68_24
'''
    
           


import torch.utils.data as data
from PIL import Image, ImageFile
import os
import sys
import csv
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

def PIL_loader(path):
    try:
        with open(path, 'rb') as f:
            return Image.open(f).convert('RGB')
    except IOError:
        print('Cannot load image ' + path)


def average_point(num1, num2, rescale_arr):
    x1 = rescale_arr[num1*2-2]
    y1 = rescale_arr[num1*2-1]
    x2 = rescale_arr[num2*2-2]
    y2 = rescale_arr[num2*2-1]
    x = (x1+x2)/2
    y = (y1+y2)/2
    return x,y
    
    
def convert68to24(landmarks_list_68 ,input_imgsize, target_imgsize):
    #print(landmarks_list_68)

    #Converting list of tuples in a single list
    land_mark = []
    for pair in landmarks_list_68:
        #print(pair)
        land_mark = land_mark + [pair[0]] + [pair[1]]
    #print(land_mark)

    total_instance  =  len(landmarks_list_68) * 2

    final_res = np.zeros([total_instance, 48], np.int32)

    final_file_name_label = []
    idx = 0

    point_num = 0
    
    rescale_arr = np.asarray(land_mark, np.float32)
    #collect reslt
    single = []
    #au1: 22, 23
    point = [19, 22, 23, 26,  39, 37, 44, 46, 28, 30, 49, 51, 53, 55, 59, 57] #16 standard landmark points from eyebrow, eyes, nose and mouth

    double_point = [
            [20, 38],
            [25, 45],
            [41, 42],
            [47, 48],
            [18, 59],
            [27, 57]
            ]  # 2-Point from left eye, 2 from right eye, next lft cheeck and eight cheek = 6 points for averaging

    for ele in point:
        single.append(rescale_arr[ele*2-2])
        single.append(rescale_arr[ele*2-1])          # Ist 16 points added


    
    for ele in double_point:
        x,y = average_point(ele[0], ele[1], rescale_arr)
        single.append(x)
        single.append(y) # 6 points added obtained from averaging

    single.append(rescale_arr[49*2-2]-4)    #[IMP: this 4 for 23rd and 24th point is chosen depending on size of image. for 256x256 size image it was 16]
    single.append(rescale_arr[49*2-1]-4)              # 23rd point: for left cheek 2nd point
    single.append(rescale_arr[55*2-2]+4)
    single.append(rescale_arr[55*2-1]-4)              # 24th point: for right cheek 2nd point
    point_num = len(single)/2

    #print(single)
    result_arr = np.asarray(single, np.float32)
    scale_single = (result_arr) * target_imgsize  / (input_imgsize)#256.0;
    rescale_single = scale_single + 0.5;
    rescale_single.astype(int)

    #print(rescale_single, len(rescale_single) )
    #convert back into list of tuples
    landmarks_list_24 = [(int(rescale_single[i]),int(rescale_single[i+1])) for i in range(0,len(rescale_single),2)] 
    #print(landmarks_list_24)
    return landmarks_list_24


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
            while sum_part < 0.75*sum_list and count < 3 and valid_emotion:
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
            if sum(emotion) <= 0.5*sum_list or count > 3: # less than 50% of the votes are integrated, or there are too many emotions, we'd better discard this example
                emotion = emotion_unknown   # force setting as unknown
            
        elif mode == 'multi_target': #this not tested
            threshold = 0.3
            for i in range(size):
                if emotion_raw[i] >= threshold*sum_list:
                    emotion[i] = emotion_raw[i]
            if sum(emotion) <= 0.5 * sum_list: # less than 50% of the votes are integrated, we discard this example
                emotion = emotion_unknown   # set as unknown
        
        return [float(i)/sum(emotion) for i in emotion]




def default_reader(fileList, landmarksfile, mode, num_expression=8 ,input_imgsize = 48, target_imgsize = 224):

    with open(landmarksfile, 'rb') as fp:
        preds_dict = pickle.load(fp)

    landmarks_dict = dict()

    for index, (key,value) in enumerate(preds_dict.items()):
        new_key = key.split('/')[-1]
        landmarks_dict[new_key] = value
        
    counter_loaded_images_per_label = [0 for _ in range(8)]

    landmarks_included_list = [i for i in range(1,69)] #][29,34,42,47,49,55,67] 

    imgList = []
    j = 0
    if fileList.find('occlusion_list.txt') > -1: 
       fp = open(fileList,'r')
       for names in fp.readlines():
           target, image_path  = names.split('_')  #Eg. for each entry before underscore lable and afterwards name in 1_fer0034656.png 8 0, 2_fer0033878.png 8 0
           image_path = image_path.strip().split(' ')[0]
           #print(target,image_path)

           if landmarks_dict[image_path] is None:
                     continue
           else:
                     landmarks = landmarks_dict[image_path][0] #It is a list.
                     #print(landmarks) 

           landmarks_list = [(int(landmarks[i][0]),int(landmarks[i][1])) for i in range(landmarks.shape[0]) if (i+1) in landmarks_included_list]
           imgList.append((image_path, int(target), landmarks_list))

       return imgList 
             
    elif  fileList.find('pose') > -1:  

       fp = open(fileList,'r')
       for names in fp.readlines():
           target, image_path  = names.split('/')  #Eg. for each entry before underscore lable and afterwards name in 1/fer0034656.jpg
           image_path = image_path.strip().replace('.jpg','.png') 
           #print(target,image_path)

           if landmarks_dict[image_path] is None:
                     continue
           else:
                     landmarks = landmarks_dict[image_path][0] #It is a list.

           landmarks_list = [(int(landmarks[i][0]),int(landmarks[i][1])) for i in range(landmarks.shape[0]) if (i+1) in landmarks_included_list]
           imgList.append((image_path, int(target), landmarks_list))
       return imgList  
          
    else:#test/train/validation.csv
         with open(fileList) as csvfile: 
              emotion_label = csv.reader(csvfile) 
              for row in emotion_label:  
                  j = j + 1
                  image_path = row[0]
             
                  #Get list of landmarks from landmarks_dict
                  if landmarks_dict[image_path] is None:
                     continue
                  else:
                     landmarks = landmarks_dict[image_path][0] #It is a list.



                  #print(landmarks)
                  landmarks_list_68 = [(int(landmarks[i][0]),int(landmarks[i][1])) for i in range(landmarks.shape[0]) if (i+1) in landmarks_included_list]
                  landmarks_list = convert68to24(landmarks_list_68 ,input_imgsize, target_imgsize)
                  
             
                  emotion_raw = list(map(float, row[2:len(row)]))
             
                  emotion = _process_data(emotion_raw, mode)

                  idx = np.argmax(emotion) 
                  if idx < num_expression: # not unknown or non-face 
                     emotion = emotion[:-2]
                     emotion = [float(i)/sum(emotion) for i in emotion]              
                     imgList.append((image_path, emotion, landmarks_list))
                  '''
                  if j > 10:
                     break  
                  '''
                  #print(counter_loaded_images_per_label)
         return imgList


def get_class(idx):  #class expression label
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


class ImageList(data.Dataset):
    def __init__(self, root, fileList, landmarksfile,  num_expressions = 8, blocksize = 28, imagesize = 224, transform=None, 
                                            list_reader=default_reader, loader=PIL_loader, mode = 'majority'):
        self.root = root

        self.training_mode = mode

        input_imgsize = 48
        block_imgsize = 28
        self.imgList = list_reader(fileList, landmarksfile, self.training_mode, num_expressions, input_imgsize , target_imgsize = block_imgsize )

        self.transform = transform

        self.loader = loader

        self.totensor = transforms.ToTensor()
        
        self.resize = transforms.Resize((imagesize,imagesize))

    def _process_target(self, target):
        '''
        Based on https://arxiv.org/abs/1608.01041 the target depend on the training mode.
        Majority or crossentropy: return the probability distribution generated by "_process_data"
        Probability: pick one emotion based on the probability distribtuion.
        Multi-target:
        '''
        if self.training_mode == 'majority' or self.training_mode == 'crossentropy':
            target = np.argmax(target)
            return target
        elif self.training_mode == 'probability':            
            idx = np.random.choice(len(target), p=target)
            return idx
        elif self.training_mode == 'multi_target':        #this case is not tested
            new_target = np.array(target)
            new_target[new_target>0] = 1.0
            epsilon = 0.001     # add small epsilon in order to avoid ill-conditioned computation
            return (1-epsilon)*new_target + epsilon*np.ones_like(target)

    
    def __getitem__(self, index):
        imgPath, target_expression, landmarks_list = self.imgList[index]
        #print(imgPath, target_expression,landmarks_list)
        img = self.loader(os.path.join(self.root, imgPath))

        if self.transform is not None:
               img = self.transform(img)
        
        target_expression = self._process_target(target_expression) #Gives emotion label as integer based on mode

        target_expression = make_emotion_compatible_to_affectnet(target_expression) #converts into emtion category same as that of affectnet
        return img, target_expression, torch.tensor(landmarks_list)

    def __len__(self):
        return len(self.imgList)



