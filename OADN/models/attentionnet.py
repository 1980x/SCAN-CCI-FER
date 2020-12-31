'''
Aum Sri Sai Ram
Authors: Darshan Gera and Dr. S. Balasubramanian, SSSIHL
Date: 16-12-2020
Email: darshangera@sssihl.edu.in
Ref: 
Occlusion-Adaptive Deep Network for Robust Facial Expression Recognition(2020)

'''
import torch
import torch.nn as nn
import copy
import torch.nn.functional as F
from os import path, makedirs


class LandmarksAttentionBranch(nn.Module):
    def __init__(self, inputdim = 2048, num_maps = 24, num_classes = 8 ): 
        super(LandmarksAttentionBranch, self).__init__()

        self.num_maps = num_maps

        self.fc = nn.Sequential(
            #nn.Dropout(),
            nn.Linear(inputdim, 256),
            nn.ReLU(inplace = True),
            nn.Linear(256, num_classes)
        )

    def forward(self, x, attention_maps):       
        
        _,c,h,w = x.size()
      
        #attention_maps = F.interpolate(attention_maps,(h,w), mode='bilinear', align_corners =  False) #commented for affectnet
        attention_maps = attention_maps.squeeze()      
        attentive_features = []
        for i in range(self.num_maps):
             att_map = attention_maps[:,i,:,:].unsqueeze(1)
             feature = x * att_map
             ap_feature = F.adaptive_avg_pool2d(feature,(1,1)).squeeze()
             attentive_features.append(gap_feature)
             
        all_features = torch.stack(attentive_features, dim = 2)
        
        max_feature = F.max_pool1d(all_features, self.num_maps).squeeze(2)
        
        output = self.fc(max_feature) 
        return output    




class RegionBranch(nn.Module):
    '''
    It gives prediction for 4 regions by dividing global feature of size 2048x14x14 into 4 non-verlapping regions of size 2048x7x7
    Feature map size : 2048 to 256 using reduction_factor = 8  
    Predictions are stacked across last dimension 
    '''
    def __init__(self, inputdim = 2048, num_regions = 4, num_classes = 8, reduction_factor = 8): #num_classes = 7 ??
        super(RegionBranch, self).__init__()

        self.num_regions = num_regions
        self.num_classes = num_classes  

              
        self.globalavgpool = nn.ModuleList([nn.AdaptiveAvgPool2d(1)  for i in range(num_regions)])      
       
            
        self.region_net = nn.Sequential( nn.Linear(inputdim, int(inputdim/reduction_factor)), nn.ReLU(), nn.Linear(int(inputdim/reduction_factor), num_classes)) 
        self.classifiers = nn.ModuleList([ self.region_net for i in range(num_regions)])           

    def forward(self, x): 
        bs, c, w, h = x.size()
        region_size = int(x.size(2) / (self.num_regions/2) )         
        patches = x.unfold(2, region_size, region_size).unfold(3,region_size,region_size)         
        patches = patches.contiguous().view(bs, c, -1, region_size, region_size).permute(0,2,1,3,4) 
        output = []
        for i in range(int(self.num_regions)):
            f = patches[:,i,:,:,:]
            f = self.globalavgpool[i](f).squeeze()            
            output.append(self.classifiers[i](f))      
            
        output_stacked = torch.stack(output, dim = 2)
        
        return output_stacked 
   



def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

#********************************************************************************************************

#Testing 

def main():
    device = torch.device("cuda" )
    basenet = resnet50()
    attnet = LandmarksAttentionBranch()
    regionnet = RegionBranch()
    #print(count_parameters(attnet)) 
    
if __name__=='__main__':
    main()


#********************************************************************************************************

