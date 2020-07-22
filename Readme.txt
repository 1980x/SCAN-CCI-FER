Aum Sri Sai Ram
Our model is implemented in Pytorch
Contents:
    1. models folder has base resnet model and attentionnet contains our SCAN and CCI branch.
    2. dataset folder has dataset class for each of datasets separately.
    3. utils has util.py for loading pretrained vggfacenet model.
    4. pretrainedmodels for storing pretrained weights from vggface2 model
              (https://github.com/ox-vgg/vgg_face2)
              
Usage:
   python train_dataset.py
