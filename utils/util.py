import torch
import shutil
import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
from sklearn.utils.multiclass import unique_labels
import pickle

def load_state_dict(model, fname):
    """
    Set parameters converted from Caffe models authors of VGGFace2 provide.
    See https://www.robots.ox.ac.uk/~vgg/data/vgg_face2/.
    Arguments:
        model: model
        fname: file name of parameters converted from a Caffe model, assuming the file format is Pickle.
    """
    with open(fname, 'rb') as f:
        weights = pickle.load(f, encoding='latin1')

    own_state = model.state_dict()
    #print(own_state.keys())
    for name, param in weights.items():
        new_name = 'module.' + name 
        if new_name in own_state and ( name.find('layer4')<=-1 and name.find('fc')<=-1): #don't load layer4 +fc #name.find('layer4')<=-1 and name.find('layer4')<=-1 and
            
            try:
                #print('copying', new_name)
                own_state[new_name].copy_(torch.from_numpy(param))
            except Exception:
                raise RuntimeError('While copying the parameter named {}, whose dimensions in the model are {} and whose '\
                                   'dimensions in the checkpoint are {}.'.format(name, own_state[name].size(), param.size()))
    print('\nModel for layers:1-3 loaded with vggface2 ', fname)      



