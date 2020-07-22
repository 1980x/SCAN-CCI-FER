#Aum Sri Sai Ram
#Code borrowed  from https://github.com/ufoym/imbalanced-dataset-sampler
#Imbalanced Dataset Sampler


import torch
import torch.utils.data
import torchvision
import pdb
import numpy as np

from  dataset.affectnet_dataset import ImageList as  affectnetImageList
from  dataset.rafdb_dataset import ImageList as rafdbImageList
from  dataset.ferplus_dataset import ImageList as ferplusImageList
from  dataset.affectnet_rafdb_dataset import ImageList as affectnet_rafdb_ImageList
from  dataset.ckplus_dataset_cv import ImageList as  ckplusImageList
from  dataset.oulucasia_dataset_cv import ImageList as  oulucasiaImageList

class ImbalancedDatasetSampler(torch.utils.data.sampler.Sampler):
    """Samples elements randomly from a given list of indices for imbalanced dataset
    Arguments:
        indices (list, optional): a list of indices
        num_samples (int, optional): number of samples to draw
    """

    def __init__(self, dataset, indices=None, num_samples=None):
        self.indices = list(range(len(dataset))) \
            if indices is None else indices
        self.num_samples = len(self.indices) \
            if num_samples is None else num_samples
        label_to_count = {}
        for idx in self.indices:
            label = self._get_label(dataset, idx)
            if label in label_to_count:
                label_to_count[label] += 1
            else:
                label_to_count[label] = 1
                
        # weight for each sample
        weights = [1.0 / label_to_count[self._get_label(dataset, idx)]
                   for idx in self.indices]
        self.weights = torch.DoubleTensor(weights)

    def _get_label(self, dataset, idx):
        dataset_type = type(dataset)
        if dataset_type is affectnetImageList:
            return dataset.imgList[idx][2]       
        elif dataset_type is rafdbImageList:
            return dataset.imgList[idx][1]
        elif dataset_type is ferplusImageList:
            target = dataset.imgList[idx][1] 
            target = np.argmax(target) #majority class only handled
            return target  
        elif dataset_type is affectnet_rafdb_ImageList:
            return dataset.imgList[idx][3] 
        elif dataset_type is ckplusImageList:
            return dataset.imgList[idx][1]
        elif dataset_type is oulucasiaImageList:
            return dataset.imgList[idx][1]     
        else:
            raise NotImplementedError
                
    def __iter__(self):
        return (self.indices[i] for i in torch.multinomial(
            self.weights, self.num_samples, replacement=True))

    def __len__(self):
        return self.num_samples
