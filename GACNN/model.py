'''
Aum Sri Sai Ram
15-Dec-2020
By Darshan and S Balasubramanian
GACNN : Implementation of Occlusion Aware Facial Expression Recognition Using CNN With Attention Mechanism (2019 IEEE transactions on IP)

'''
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
import numpy as np
import torchvision
from torchvision import datasets, models, transforms
import matplotlib.pyplot as plt
import time
import os
import copy
from ferplus_dataset import ImageList
from  torch.utils.data import DataLoader

class View(nn.Module):
    """Changes view using a nn.Module."""

    def __init__(self, *shape):
        super(View, self).__init__()
        self.shape = shape

    def forward(self, input):
        return input.view(*self.shape)

class Net(nn.Module):
    def __init__(self, num_classes=7):
              
        super(Net, self).__init__()
         
        self.inputdim = 512
        self.num_regions = 24       
        self.base = models.vgg16(pretrained=True).features[:21]    #21 correspond to conv4_2 (9th layer) in vgg16 
        
        self.local_attention_block =   nn.ModuleList([  nn.Sequential( 
                                        nn.MaxPool2d(2),           
                                        nn.Conv2d(self.inputdim, 128, kernel_size=1),
                                        nn.ReLU(),
                                        nn.BatchNorm2d(128),
                                        View(-1,128*3*3),                                                                                
                                        nn.Linear(128*3*3,64),
                                        nn.ReLU(),                                         
                                        nn.Linear(64,1),                                                                                         
                                        nn.Sigmoid()) 
                                       for i in range(self.num_regions)])  
        
        self.global_attention_block =    nn.Sequential(nn.MaxPool2d(2),           
                                        nn.Conv2d(self.inputdim, 128, kernel_size=1),
                                        nn.ReLU(), 
                                        nn.BatchNorm2d(128),
                                        View(-1,128*7*7),                                                                               
                                        nn.Linear(128*7*7,64),
                                        nn.ReLU(),                                        
                                        nn.Linear(64,1),                                                                                         
                                        nn.Sigmoid() ) 
                                        
        self.PG_unit_1 =  nn.ModuleList([ nn.Sequential(          
                                        nn.Conv2d(512, 512, kernel_size=3, padding = 1),
                                        nn.ReLU())                  
                                    for i in range(self.num_regions)])                       
                                                
        self.PG_unit_2 =  nn.ModuleList([ nn.Sequential(
                                        View(-1,512*6*6),          
                                        nn.Linear(512*6*6, 64), 
                                        nn.ReLU())                  
                                    for i in range(self.num_regions)])    

        self.GG_unit_1 =   nn.Sequential(
                                        nn.MaxPool2d(2),          
                                        nn.Conv2d(512, 512, kernel_size=3, padding = 1),
                                        nn.ReLU() )                
                                                        
                                                
        self.GG_unit_2 =  nn.Sequential( 
                                        View(-1,512*14*14),         
                                        nn.Linear(512*14*14, 512), 
                                        nn.ReLU())                  
                                  

        self.fc1 = nn.Linear(2048,1024)
       
        self.fc2 = nn.Linear(1024, num_classes)
                                                                                               
         
    def pg_unit(self, patch, region_num):        
        out = self.PG_unit_1[region_num](patch)
        att_wt = self.local_attention_block[region_num](out)
        out = self.PG_unit_2[region_num](out)
        att_out = out * att_wt
        return att_out 

    def gg_unit(self, patch):
        temp = self.GG_unit_1(patch)
        global_wt =  self.global_attention_block(temp)
        global_features = global_wt * self.GG_unit_2(temp)
        return global_features          
               
    def forward(self, x, landmarks_list): #landmarks list  
        bs = x.size(0)     

        out = self.base(x)

        landmarks_list[landmarks_list < 3] = 3 
        landmarks_list[landmarks_list > 24] = 24
          
        patches_24 = [[out[i, :, x-3:x+3, y-3:y+3] for x, y in landmarks_list[i]] for i in range(bs)] 
        patches_24 = torch.stack([torch.stack(patches_24[i], dim=0) for i in range(bs)], dim=0) #bs x 24 x 512 x 6 x 6 
        local_features = torch.cat([self.pg_unit(patches_24[:, i], i) for i in range(24)], dim=1)
       
        global_features = self.gg_unit(out)    
        
        features = torch.cat((local_features, global_features), dim = 1)
        
        f = self.fc1(features)       
        f = self.fc2(f) 
        
        return f 
       
         
         
       
       
         


