'''
Aum Sri Sai Ram
Authors: Darshan Gera and Dr. S. Balasubramanian, SSSIHL
Date: 18-12-2020
Email: darshangera@sssihl.edu.in
Purpose: FER on SFEW using GACNN
'''
# External Libraries
import argparse
import os,sys,shutil
import time
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
import torch.nn.functional as F
import torchvision.datasets as datasets
import torchvision.models as models
import torchvision.transforms as transforms
import math
from PIL import Image

#dataset class and model 

import scipy.io as sio
import numpy as np
import pdb
from statistics import mean 
os.environ['CUDA_VISIBLE_DEVICES'] = '2'

from model import Net
from sfew_dataset import ImageList




#######################################################################################################################################
# Training settings
parser = argparse.ArgumentParser(description='SFEW expression recognition')

# DATA

parser.add_argument('--root_path', type=str, default='../data/SFEW/',
                    help='path to root path of images')

parser.add_argument('--database', type=str, default='sfew',
                    help='Which Database for train. (flatcam, ferplus, affectnet)')

parser.add_argument('--train_list', type=str, default = '../data/SFEW/sfew_train.txt',
                    help='path to training list')

parser.add_argument('--valid_list', type=str, default = '../data/SFEW/sfew_val.txt',
                    help='path to validation list')

parser.add_argument('--epochs', default=100, type=int, metavar='N', help='number of total epochs to run')

parser.add_argument('--start-epoch', default=0, type=int, metavar='N', help='manual epoch number (useful on restarts)')

parser.add_argument('-b', '--batch-size', default=64, type=int, metavar='N', help='mini-batch size (default: 256)')

parser.add_argument('-b_t', '--batch-size_t', default=64, type=int, metavar='N', help='mini-batch size (default: 256)')

parser.add_argument('--lr', '--learning-rate', default=0.0001, type=float, metavar='LR', help='initial learning rate')

parser.add_argument('--momentum', default=0.9, type=float, metavar='M',  help='momentum')

parser.add_argument('--weight-decay', '--wd', default=1e-4, type=float,  metavar='W', help='weight decay (default: 1e-4)')

parser.add_argument('--print-freq', '-p', default=1000, type=int,metavar='N', help='print frequency (default: 10)')

parser.add_argument('--resume', default='checkpoints_affectnet8/model_best.pth.tar', type=str, metavar='PATH',   help='path to latest checkpoint (default: none)')

parser.add_argument('--pretrained', default='pretrainedmodels/vgg_msceleb_resnet50_ft_weight.pkl', type=str, metavar='PATH', 
                    help='path to latest checkpoint (default: none)')

parser.add_argument('-e', '--evaluate', dest='evaluate', action='store_true', 
                    help='evaluate model on validation set')


parser.add_argument('--model_dir','-m', default='checkpoints_sfew', type=str)

parser.add_argument('--imagesize', type=int, default = 224, help='image size (default: 224)')

parser.add_argument('--end2end', default=True,\
        help='if true, using end2end with dream block, else, using naive architecture')

parser.add_argument('--num_classes', type=int, default=7, help='number of expressions(class)')

parser.add_argument('--num_attentive_regions', type=int, default=25, help='number of non-overlapping patches(default:25)')

parser.add_argument('--num_regions', type=int, default=4, help='number of non-overlapping patches(default:4)')

parser.add_argument('--train_rule', default='None', type=str, help='data sampling strategy for train loader:Resample, DRW,Reweight, None')

parser.add_argument('--loss_type', default="CE", type=str, help='loss type:Focal, CE')

parser.add_argument('--train_landmarksfile', type=str, default = '../data/SFEW/sfew_train_landmarks_scores.pkl',
                    help='path to landmarksdictionary')
                    
parser.add_argument('--test_landmarksfile', type=str, default = '../data/SFEW/sfew_valid_landmarks_scores.pkl',
                    help='path to landmarksdictionary')


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

parser.add_argument('--workers', type=int, default = 8,
                    help='how many workers to load data')


args = parser.parse_args()

best_prec1 = 0
#######################################################################################################################################
def main():
    #Print args
    global args, best_prec1
    args = parser.parse_args()
    print('\n\t\t\t\t Aum Sri Sai Ram\nFER on SFEW using GACNN\n\n')
    print(args)
    print('\nimg_dir: ', args.root_path)
    print('\ntrain rule: ',args.train_rule, ' and loss type: ', args.loss_type, '\n')

    mean = [0.5, 0.5, 0.5]
    std = [0.5, 0.5, 0.5]
    imagesize = args.imagesize
    train_transform = transforms.Compose([
            transforms.RandomHorizontalFlip(p=0.5),            
            transforms.Resize((args.imagesize, args.imagesize)),
            transforms.ToTensor(),
            transforms.Normalize(mean,std)
        ])

    
    valid_transform = transforms.Compose([
            transforms.Resize((args.imagesize,args.imagesize)),
            transforms.ToTensor(),
            transforms.Normalize(mean,std)
        ])

    val_data = ImageList(root=args.root_path +'Val_Aligned_Faces_SAN/', fileList=args.valid_list, landmarksfile = args.test_landmarksfile,
                  transform=valid_transform)   
    
    val_loader = torch.utils.data.DataLoader(val_data, args.batch_size, shuffle=False, num_workers=8)

    train_dataset = ImageList(root=args.root_path+'Train_Aligned_Faces_SAN/', fileList = args.train_list, landmarksfile = args.train_landmarksfile,
                  transform=train_transform)


    #train_loader = DataLoader(train_dataset, args.batch_size, shuffle=True, num_workers=8)
    #Instead use sampler 
    #train_loader = torch.utils.data.DataLoader(train_dataset, args.batch_size, sampler=ImbalancedDatasetSampler(train_dataset), num_workers=8)
    

    if args.train_rule == 'None':
       train_sampler = None  
       per_cls_weights = None 
    
    if args.loss_type == 'CE':
       criterion = nn.CrossEntropyLoss(weight=per_cls_weights).to(device)
    else:
       warnings.warn('Loss type is not listed')
       return
    

        
    train_loader = torch.utils.data.DataLoader(train_dataset, args.batch_size, shuffle=(train_sampler is None),
                                                   num_workers=args.workers, pin_memory=True, sampler=train_sampler)    
    
    print('length of SFEW train Database: ' + str(len(train_dataset)))
    print('length of SFEW valid Database: ' + str(len(val_loader.dataset)))
    # prepare model
    
    basemodel = torch.nn.DataParallel(Net(num_classes=args.num_classes)).to(device)
    
    criterion = nn.CrossEntropyLoss().to(device)

    criterion1 = criterion#MarginLoss(loss_lambda=0.5).to(device)
    optimizer1 =  torch.optim.SGD([{"params": basemodel.parameters(), "lr": args.lr, "momentum":args.momentum,
                                 "weight_decay":args.weight_decay}])
    if args.resume:
        if os.path.isfile(args.resume):
            print("=> loading checkpoint '{}'".format(args.resume))
            checkpoint = torch.load(args.resume)
            #args.start_epoch = checkpoint['epoch']
            #best_prec1 = checkpoint['best_prec1']
            pretrained_state_dict = checkpoint['state_dict']
            model_state_dict = basemodel.state_dict()        
            #print(model_state_dict.keys()) 
            for key in pretrained_state_dict:
                if  ((key=='module.fc2.weight')|(key=='module.fc2.bias')):
                    print(key) 
                    pass
                else:   
                    #print(key)             
                    model_state_dict[key] = pretrained_state_dict[key]

            basemodel.load_state_dict(model_state_dict, strict = True)
        
            
            print("=> loaded checkpoint '{}' (epoch {})"
                  .format(args.resume, checkpoint['epoch']))
        else:
            print("=> no checkpoint found at '{}'".format(args.resume))

   
    print('Training starting:\n')
    for epoch in range(args.start_epoch, args.epochs):
        if epoch > 50:
           adjust_learning_rate(optimizer1, epoch)

        # train for one epoch
        
        train(train_loader, basemodel, criterion,  optimizer1,  epoch)
        
        prec1 = validate(val_loader, basemodel, criterion, optimizer1,  epoch)
        
        
        
        print("Epoch: {}   Test Acc: {}".format(epoch, prec1))
        # remember best prec@1 and save checkpoint
        
        is_best = prec1 > best_prec1

        best_prec1 = max(prec1.to(device).item(), best_prec1)
        
        save_checkpoint({
            'epoch': epoch + 1,            
            'state_dict': basemodel.state_dict(),
            'best_prec1': best_prec1,
            'optimizer' : optimizer1.state_dict(),
        }, is_best.item())
        

def train(train_loader,  model,  criterion, optimizer1, epoch):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top2 = AverageMeter()
    top5 = AverageMeter()
    loss = AverageMeter()
    
     
    # switch to train mode
    #model.train()

    end = time.time()


    for i, (input, target, landmarks) in enumerate(train_loader):
        # measure data loading time
        data_time.update(time.time() - end)

        input = input.to(device)

        target = target.to(device)
        
        # compute output
        preds = model(input, landmarks)
        prec = accuracy(preds, target, topk=(1,))
        loss1 = criterion(preds, target) 
        loss.update(loss1.item(), input.size(0))
        top1.update(prec[0], input.size(0))

         # compute gradient and do SGD step
        optimizer1.zero_grad()     
        loss1.backward()
        optimizer1.step()
        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0:
            print('Training Epoch: [{0}][{1}/{2}]\t'
                  'Time  ({batch_time.avg})\t'
                  'Data ({data_time.avg})\t'
                  'loss  ({loss.avg})\t'
                    'Prec1  ({top1.avg}) \t'.format(
                   epoch, i, len(train_loader), batch_time = batch_time, data_time=data_time, loss=loss, top1=top1))
                   #data_time=data_time, loss=losses,  top1=top1))


def validate(val_loader,  model, criterion,  optimizer1, epoch):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top2 = AverageMeter()
    top5 = AverageMeter()
    loss = AverageMeter()
    
    mode =  'Testing'
    # switch to evaluate mode
    model.eval()
    
    end = time.time()

    corrects = [0 for _ in range(2 + 1)] #2 predictions due to ce + wing +1(majority)


    with torch.no_grad():         
        for i, (input, target, landmarks) in enumerate(val_loader):        
            data_time.update(time.time() - end)
            input = input.to(device) 
            target = target.to(device)
            preds = model(input, landmarks)
            prec = accuracy(preds, target, topk=(1,))
            loss1 = criterion(preds, target) 
            loss.update(loss1.item(), input.size(0))
            #print(prec)
            top1.update(prec[0], input.size(0))
            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if i % args.print_freq == 0:
               print('Testing Epoch: [{0}][{1}/{2}]\t'
                  'Time  ({batch_time.avg})\t'
                  'Data ({data_time.avg})\t'
                  'loss  ({loss.avg})\t'
                    'Prec1  ({top1.avg}) \t'.format(
                   epoch, i, len(val_loader), batch_time = batch_time, data_time=data_time, loss=loss, top1=top1))
        print('Testing Epoch: [{0}][{1}/{2}]\t'
                  'Time  ({batch_time.avg})\t'
                  'Data ({data_time.avg})\t'
                  'loss  ({loss.avg})\t'
                    'Prec1  ({top1.avg}) \t'.format(
                   epoch, i, len(val_loader), batch_time = batch_time, data_time=data_time, loss=loss, top1=top1))

    return top1.avg

def save_checkpoint(state, is_best, filename='checkpoint.pth.tar'):
    epoch_num = state['epoch']
    full_filename = os.path.join(args.model_dir, str(epoch_num)+'_'+ filename)
    full_bestname = os.path.join(args.model_dir, 'model_best.pth.tar')
    
    if is_best:
        torch.save(state, full_bestname)
        #shutil.copyfile(full_filename, full_bestname)


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count



def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res
    
def adjust_learning_rate(optimizer, epoch):
        print('\n******************************\n\tAdjusted learning rate: '+str(epoch) +'\n')    
        for param_group in optimizer.param_groups:
           param_group['lr'] *= 0.95
           print(param_group['lr'])              
        

if __name__ == '__main__':
    main()
