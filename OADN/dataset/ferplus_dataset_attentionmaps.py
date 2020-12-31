'''
Aum Sri Sai Ram
Implementation of FERPlus class for OADN : it returns attention maps along with image, label
Authors: Darshan Gera and Dr. S. Balasubramanian, SSSIHL
Date: 16-12-2020
Email: darshangera@sssihl.edu.in

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
import math
from torchvision import transforms
from torchvision.utils import make_grid
import torch.nn.functional as F

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
    return landmarks_24


def _gaussian(
        size=3, sigma=0.25, amplitude=1, normalize=False, width=None,
        height=None, sigma_horz=None, sigma_vert=None, mean_horz=0.5,
        mean_vert=0.5, score = 1): #score added by DG 

    #amplitude = score * amplitude #BY DG

    if width is None:
        width = size
    if height is None:
        height = size
    if sigma_horz is None:
        sigma_horz = sigma
    if sigma_vert is None:
        sigma_vert = sigma
    center_x = mean_horz * width + 0.5
    center_y = mean_vert * height + 0.5
    gauss = np.empty((height, width), dtype=np.float32)
    # generate kernel
    for i in range(height):
        for j in range(width):
            gauss[i][j] = amplitude * math.exp(-(math.pow((j + 1 - center_x) / (
                sigma_horz * width), 2) / 2.0 + math.pow((i + 1 - center_y) / (sigma_vert * height), 2) / 2.0))
    if normalize:
        gauss = gauss / np.sum(gauss)
    return gauss

def draw_gaussian(image, point, sigma, score):
    # Check if the gaussian is inside
    ul = [np.floor(np.floor(point[0]) - 3 * sigma),
          np.floor(np.floor(point[1]) - 3 * sigma)]
    br = [np.floor(np.floor(point[0]) + 3 * sigma),
          np.floor(np.floor(point[1]) + 3 * sigma)]
    if (ul[0] > image.shape[1] or ul[1] >
            image.shape[0] or br[0] < 1 or br[1] < 1):
        return image
    size = 6 * sigma + 1
    g = _gaussian(size)
    g_x = [int(max(1, -ul[0])), int(min(br[0], image.shape[1])) -
           int(max(1, ul[0])) + int(max(1, -ul[0]))]
    g_y = [int(max(1, -ul[1])), int(min(br[1], image.shape[0])) -
           int(max(1, ul[1])) + int(max(1, -ul[1]))]
    img_x = [int(max(1, ul[0])), int(min(br[0], image.shape[1]))]
    img_y = [int(max(1, ul[1])), int(min(br[1], image.shape[0]))]
    assert (g_x[0] > 0 and g_y[1] > 0)
    correct = False
    while not correct:
        try:
            image[img_y[0] - 1:img_y[1], img_x[0] - 1:img_x[1]
            ] = image[img_y[0] - 1:img_y[1], img_x[0] - 1:img_x[1]] + g[g_y[0] - 1:g_y[1], g_x[0] - 1:g_x[1]]
            correct = True
        except:
            print('img_x: {}, img_y: {}, g_x:{}, g_y:{}, point:{}, g_shape:{}, ul:{}, br:{}'.format(img_x, img_y, g_x, g_y, point, g.shape, ul, br))
            ul = [np.floor(np.floor(point[0]) - 3 * sigma),
                np.floor(np.floor(point[1]) - 3 * sigma)]
            br = [np.floor(np.floor(point[0]) + 3 * sigma),
                np.floor(np.floor(point[1]) + 3 * sigma)]
            g_x = [int(max(1, -ul[0])), int(min(br[0], image.shape[1])) -
                int(max(1, ul[0])) + int(max(1, -ul[0]))]
            g_y = [int(max(1, -ul[1])), int(min(br[1], image.shape[0])) -
                int(max(1, ul[1])) + int(max(1, -ul[1]))]
            img_x = [int(max(1, ul[0])), int(min(br[0], image.shape[1]))]
            img_y = [int(max(1, ul[1])), int(min(br[1], image.shape[0]))]
            pass
    image[image > 1] = 1
    return image

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

           if landmarks_dict[image_path] is None:
                     continue
           else:
                     landmarks = landmarks_dict[image_path][0] #It is a list.
                     #print(landmarks) 

           landmarks_list = [(int(landmarks[i][0]),int(landmarks[i][1]), landmarks[i][2]) for i in range(landmarks.shape[0]) if (i+1) in landmarks_included_list]
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

           landmarks_list = [(int(landmarks[i][0]),int(landmarks[i][1],landmarks[i][2])) for i in range(landmarks.shape[0]) if (i+1) in landmarks_included_list]
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
                     landmarks_68 = landmarks_dict[image_path]#[0] #It is a list.

                  landmarks = convert68to24(landmarks_68 ,input_imgsize, target_imgsize) 

                  landmarks_list = [(landmarks[i][0],landmarks[i][1], landmarks[i][2]) for i in range(0,24)]
             
                  emotion_raw = list(map(float, row[2:len(row)]))
             
                  emotion = _process_data(emotion_raw, mode)

                  idx = np.argmax(emotion) 
                  if idx < num_expression: # not unknown or non-face 
                     emotion = emotion[:-2]
                     emotion = [float(i)/sum(emotion) for i in emotion]              
                     imgList.append((image_path, emotion, landmarks_list))
                  
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
    def __init__(self, root, fileList, landmarksfile,  num_expressions = 8, input_imgsize = 48,
                               target_imgsize = 224, transform=None, 
                                            list_reader=default_reader, loader=PIL_loader, mode = 'majority'):
        self.root = root

        self.training_mode = mode

        
        self.input_imgsize = input_imgsize
        self.target_imgsize = target_imgsize
        
        self.imgList = list_reader(fileList, landmarksfile, self.training_mode, num_expressions, input_imgsize=self.input_imgsize , target_imgsize = self.target_imgsize  )

        self.transform = transform

        self.loader = loader

        self.is_save = False
        self.totensor = transforms.ToTensor()
        self.score_threshold = 0.6
        

    
    def generate_gaussian_attention_maps(self, image, landmarks_24, input_imgsize, target_imgsize, threshold): # target_imgsize is not being used. attention_map size will be that of input imagesize

        assert isinstance(image, Image.Image), 'image type is not PIL.Image.Image'
        _image = np.array(image.convert('L'))
       
    
        attention_map = np.zeros_like(_image, np.float32)
        final_attention_gaussian_maps = []
        pts =  landmarks_24.copy()
        num_points = len(pts)#.shape[1]
        visiable_points = []
        for idx in range(num_points):
            if pts[idx][2] > threshold: #checking score is greater than threshold
                visiable_points.append( True )
            else:
                visiable_points.append( False )
        visiable_points = np.array( visiable_points )
        #print (' points with score > threshold : {}'.format( np.sum(visiable_points) ))
        for idx in range(num_points):
            attention_map = np.zeros_like(_image, np.float32)
            if visiable_points[ idx ]:
               point = (pts[idx][0], pts[idx][1])
               score = pts[idx][2]
               attention_map = draw_gaussian(attention_map, point, sigma = 7, score = score )   #Sigma            
            final_attention_gaussian_maps.append(attention_map)

            if self.is_save:
               attention_gaussian_map = Image.fromarray(np.uint8(255 * attention_map))
               attention_gaussian_map.save( str(idx)+'_attention_map.jpg')
        
        #print(final_attention_gaussian_maps[0].shape) 
        #temp = final_attention_gaussian_maps[0]
        #print(temp[10:40,10:40])
        return np.array(final_attention_gaussian_maps)
     
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
        imgPath, target_expression, landmarks_24 = self.imgList[index]
        #print(landmarks_24)
        img = self.loader(os.path.join(self.root, imgPath))

        
        target_expression = self._process_target(target_expression) #Gives emotion label as integer based on mode

        target_expression = make_emotion_compatible_to_affectnet(target_expression) #converts into emtion category same as that of affectnet
        
        #attention_map size will be that of input imagesize, target_size not required here
        attention_maps = self.generate_gaussian_attention_maps(img, landmarks_24, self.input_imgsize, self.target_imgsize, self.score_threshold)
        attention_maps = torch.from_numpy(attention_maps).unsqueeze(0)
        attention_maps = F.interpolate(attention_maps,(14,14), mode='bilinear', align_corners =  False)
        
        if self.transform is not None:
               img = self.transform(img)
           
        return img, target_expression,  attention_maps#attention map size is that of input_imagesize

    def __len__(self):
        return len(self.imgList)
    

if __name__=='__main__':
   testlist = default_reader('../data/FERPLUS/Dataset/Labels/FER2013Test/label.csv',landmarksfile = '../data/FERPLUS/Dataset/FERPLUS_test_landmarks_scores.pkl', mode='majority')

  

   imagesize =  224
   transform = transforms.Compose([transforms.Resize((imagesize,imagesize)), transforms.ToTensor()])
   
   dataset = ImageList(root='../data/FERPLUS/Dataset/Images/FER2013Test/', fileList ='../data/FERPLUS/Dataset/Labels/FER2013Test/label.csv', 
                   landmarksfile = '../data/FERPLUS/Dataset/FERPLUS_test_landmarks_scores.pkl',  transform = transform)

   fdi = iter(dataset)
   for i, data in enumerate(fdi):
        if i < 1:
           print('', data[0].size(), data[1],len(data[2]), data[2].size())
        else:
           break
 


