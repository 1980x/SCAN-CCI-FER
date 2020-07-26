'''
Aum Sri Sai Ram

Implementation of LANDMARK GUIDANCE INDEPENDENT SPATIO-CHANNEL
ATTENTION AND COMPLEMENTARY CONTEXT INFORMATION
BASED FACIAL EXPRESSION RECOGNITION 

https://arxiv.org/pdf/2007.10298v1.pdf
Authors: Darshan Gera and Dr. S. Balasubramanian, SSSIHL
Date: 20-07-2020
Email: darshangera@sssihl.edu.in

Our Idea:
Implementation of  i)local attention and global attention using Attention branch using
     non-overlapping regions of size 
              a) 16 regions 45x45 from 224x224 image
              b) 4 regions 45x44 
              c) 4 regions 44x45
              d) 1 regions 44x44

      Corresponding features maps from 512x28x28   correspond to :
              a) 16 regions 6x6 from 28x28 feature map obtained from resnet pertrained model
              b) 4 regions 6x4 
              c) 4 regions 4x6
              d) 1 regions 4x4

            ii) whole image as input for global attention with global features map of size 28x28

Model:
i)Attention branch: For local and global attention networks 
    a) Local attention:
           Input: 512x28x28 feature maps (from layer2 of resnet50)
           Output: List of local attentive feature maps for 25 non-overlapping patches
    b) Global attention:
           Input: 512x28x28 feature maps
           Output: 512x28x28 attentive feature maps
    c) CELoss for feature map obtained from concatenation of Local blocks features + Global attentive feature

ii) Region branch:
     Input:bsx1024x14x14 (from layer3 of resnet50) 
     Ouput:bsx8x4  (4 regions prediction stacked) 

'''

import torch
import torch.nn as nn
import copy
import torch.nn.functional as F
from os import path, makedirs


#********************************************************************************************************
# Attention Net for learning attentive map of size same that of input feature map 
class Attentiomasknet(nn.Module):
    def __init__(self, inputdim = 512):
        super(Attentiomasknet, self).__init__()
        self.attention = nn.Sequential(            
            nn.Conv2d(inputdim, inputdim, kernel_size=3, stride=1, padding=1, bias=False),
            nn.PReLU(inputdim),
            nn.BatchNorm2d(inputdim),
            nn.Sigmoid(),
        ) 
    def forward(self, x):
        y = self.attention(x)
        return y

#********************************************************************************************************
#Below function extracts all 25 non-overlapping patches from 512x28x28 feature map
# Returns a list of 16 6x6, followed by 4 6x4, followed by 4 4x6 and next 1 4x4 patches local feature maps
def extract_patches_attentivefeatures(attention_branch_feat): 
        patches_list = []
        bs, c, w, h = attention_branch_feat.size() 
        region_size = 6
        x = attention_branch_feat[:,:,:24,:24]   
        patches = x.unfold(2, region_size, region_size).unfold(3, region_size, region_size) 
        patches_6_6 = patches.contiguous().view(bs, c, -1, region_size, region_size).permute(0,2,1,3,4) 
        patches_list.append(patches_6_6)
        x = attention_branch_feat[:, :, :24, 24:]
        patches_6_4 = x.unfold(2, region_size, region_size).permute(0,2,1,4,3) 
        patches_list.append(patches_6_4)
        x = attention_branch_feat[:, :, 24:, :24] 
        patches_4_6 = x.unfold(3, region_size, region_size).permute(0,2,1,3,4)
        patches_list.append(patches_4_6)
        patches_4_4 = attention_branch_feat[:,:,24:,24:].unsqueeze(1)
        patches_list.append(patches_4_4) 
        return patches_list
       

#********************************************************************************************************    
#i) Learns attentive feature maps for each of non-overlapping local patches as well as for global whole face using Attentiomasknet

class AttentionBranch(nn.Module):
    def __init__(self, inputdim = 512, num_regions = 25, num_classes = 8 ):
        super(AttentionBranch, self).__init__()
        self.num_regions = num_regions

        self.local_attention_masks = nn.ModuleList([Attentiomasknet(inputdim)   for i in range(num_regions)])
        
        self.localattention_globalavgpool = nn.ModuleList([nn.AdaptiveAvgPool2d(1)  for i in range(num_regions)])        
        self.localattention_maxpool = nn.MaxPool1d(num_regions)
        
        self.global_attention_mask = Attentiomasknet(inputdim)
        self.globalattention_globalavgpool = nn.AdaptiveAvgPool2d(1)
      
        self.fc = nn.Linear(2 * inputdim, num_classes) 

    def forward(self, x):  
        local_features_patches_list = extract_patches_attentivefeatures(x)        
        gap_local_features = []

        k = 0  
        #Iterate through 16:6x6,4:6x4, 4:4x6 and 1:4x4 patches
        for i in range(len(local_features_patches_list)): 
            patches_feature = local_features_patches_list[i]

            for j in range(patches_feature.size(1)):
                f = patches_feature[:,j,:,:,:]

                local_attention_mask = self.local_attention_masks[k](f)
                f = f * local_attention_mask            
                localfeat = self.localattention_globalavgpool[k](f)
                localfeat = localfeat.squeeze(3).squeeze(2) 
                gap_local_features.append(localfeat)
                k += 1
       
        gap_local_features_stack = torch.stack(gap_local_features, dim = 2) 
        gap_local_features = self.localattention_maxpool(gap_local_features_stack).squeeze(2)
        global_attention_mask = self.global_attention_mask(x)
        global_features = global_attention_mask * x       
        gap_global_features = self.globalattention_globalavgpool(global_features).squeeze(3).squeeze(2)
        combined_attention_features  = torch.cat([gap_local_features, gap_global_features], dim = 1 )
        combined_attention_features  = F.normalize(combined_attention_features, p=2, dim=1)
        output = self.fc(combined_attention_features)

        return local_features_patches_list, global_features, output

#********************************************************************************************************    
#i) Learns attentive feature maps for each of non-overlapping local patches as well as for global whole face using Attentiomasknet
#Here attention net is shared for all local patches
class SharedAttentionBranch(nn.Module):
    def __init__(self, inputdim = 512, num_regions = 25, num_classes = 8 ):
        super(SharedAttentionBranch, self).__init__()
        self.num_regions = num_regions

        self.local_attention_masks = Attentiomasknet(inputdim)
        
        self.localattention_globalavgpool = nn.ModuleList([nn.AdaptiveAvgPool2d(1)  for i in range(num_regions)])        
        self.localattention_maxpool = nn.MaxPool1d(num_regions)
        
        self.global_attention_mask = Attentiomasknet(inputdim)
        self.globalattention_globalavgpool = nn.AdaptiveAvgPool2d(1)
      
        self.fc = nn.Linear(2 * inputdim, num_classes) 

    def forward(self, x):  
        local_features_patches_list = extract_patches_attentivefeatures(x)        
        gap_local_features = []

        k = 0  
        #Iterate through 16:6x6,4:6x4, 4:4x6 and 1:4x4 patches
        for i in range(len(local_features_patches_list)): 
            patches_feature = local_features_patches_list[i]

            for j in range(patches_feature.size(1)):
                f = patches_feature[:,j,:,:,:]

                local_attention_mask = self.local_attention_masks(f)
                f = f * local_attention_mask            
                localfeat = self.localattention_globalavgpool[k](f)
                localfeat = localfeat.squeeze(3).squeeze(2) 
                gap_local_features.append(localfeat)
                k += 1
       
        gap_local_features_stack = torch.stack(gap_local_features, dim = 2) 
        gap_local_features = self.localattention_maxpool(gap_local_features_stack).squeeze(2)
        global_attention_mask = self.global_attention_mask(x)
        global_features = global_attention_mask * x       
        gap_global_features = self.globalattention_globalavgpool(global_features).squeeze(3).squeeze(2)
        combined_attention_features  = torch.cat([gap_local_features, gap_global_features], dim = 1 )
        combined_attention_features  = F.normalize(combined_attention_features, p=2, dim=1)
        output = self.fc(combined_attention_features)

        return local_features_patches_list, global_features, output

#********************************************************************************************************

#********************************************************************************************************
#Ref: Below function borrowed from  https://github.com/BangguWu/
#It calculates ECA attention
class eca_layer(nn.Module):
    """Constructs a ECA module.
    Args:
        channel: Number of channels of the input feature map
        k_size: Adaptive selection of kernel size #5 for 1024 channels and 5 for 2048
    """
    def __init__(self, channel, k_size = 5):
        super(eca_layer, self).__init__()

        self.avg_pool = nn.AdaptiveAvgPool2d(1)
     
        self.conv = nn.Conv1d(1, 1, kernel_size=k_size, padding=(k_size - 1) // 2, bias=False) 

        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # x: input features with shape [b, c, h, w]
        b, c, h, w = x.size()

        # feature descriptor on the global spatial information
        y = self.avg_pool(x)

        # Two different branches of ECA module
        y = self.conv(y.squeeze(-1).transpose(-1, -2)).transpose(-1, -2).unsqueeze(-1)

        # Multi-scale information fusion
        y = self.sigmoid(y)

        return x * y.expand_as(x)

#********************************************************************************************************
class RegionBranch(nn.Module):
    '''
    It gives prediction for 4 regions by dividing global feature of size 1024x14x14 into 4 non-verlapping regions of size 1024x7x7
    Feature map size : 1024 to 256 using reduction_factor = 4  
    Predictions are stacked across last dimension 
    '''
    def __init__(self, inputdim = 1024, num_regions = 4, num_classes = 8, reduction_factor = 4):
        super(RegionBranch, self).__init__()

        self.num_regions = num_regions
        self.num_classes = num_classes  
        self.eca = eca_layer( channel = inputdim, k_size = 5) 
        self.globalavgpool = nn.ModuleList([nn.AdaptiveAvgPool2d(1)  for i in range(num_regions)])                         
        self.region_net = nn.Sequential( nn.Linear(inputdim, int(inputdim/reduction_factor)), nn.ReLU(), nn.Linear(int(inputdim/reduction_factor), num_classes)) 
        self.classifiers = nn.ModuleList([ self.region_net for i in range(num_regions)])           

    def forward(self, x): 
        #Input: x is bsx1024x14x14 corresponding to global facial feature
        bs, c, w, h = x.size()
        x = self.eca(x) 
        region_size = int(x.size(2) / (self.num_regions/2) ) 

        patches = x.unfold(2, region_size, region_size).unfold(3,region_size,region_size) 
        patches = patches.contiguous().view(bs, c, -1, region_size, region_size).permute(0,2,1,3,4)
        output = []
        for i in range(int(self.num_regions)):
            f = patches[:,i,:,:,:] 
            f = self.globalavgpool[i](f).squeeze(3).squeeze(2)
            
            output.append(self.classifiers[i](f))      

        output_stacked = torch.stack(output, dim = 2)
        return output_stacked
   
#********************************************************************************************************

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

#********************************************************************************************************

#Testing 

def main():
    device = torch.device("cpu" )
    x = torch.rand(2,512,28,28) 
    t = extract_patches_attentivefeatures(x)
    net = AttentionBranch()
    print(count_parameters(net))
    local_feature_list, global_feature,output = net(x)
    for i in range(len(local_feature_list)):
        print(local_feature_list[i].size())
    regionnet = RegionBranch()
    print(count_parameters(regionnet))
    
    x = torch.rand(2,1024,14,14) 
    output = regionnet(x)
    print('outut: ',output.size())
    
    
if __name__=='__main__':
    main()


#********************************************************************************************************

