Aum Sri Sai Ram

Authors: Darshan Gera and Dr. S. Balasubramanian, SSSIHL

This is an implemenation of following paper:

Ref: Hui Ding, Peng Zhou, and Rama Chellappa. Occlusion-adaptive deep network for robust facial expression recognition. arXiv preprint arXiv:2005.06040, 2020

Requirements: pytorch >=1.3, torchvision

Files required: Landmarks scores need to be generated along with 68 landmarks points for each of datasets required in dataset file. 

Reference: https://ghttps://github.com/D-X-Y/landmark-detection/tree/master/SAN

(Contact authors if you need these files)

Folders:
dataset: This folder has dataset class for each of dataset AffectNet, RAFDB, Ferplus and SFEW.
model: This folder has backbone resent model and attention based model.
logs: This has logs of training and testing different models
Each of train_datasetname.py has training and testing code for respective dataset.

Contact darshangera@sssihl.edu.in in case of any queries.
