'''
Aum Sri Sai Ram
Aum Sri Sai Ram
Authors: Darshan Gera and Dr. S. Balasubramanian, SSSIHL
Date: 18-12-2020
Email: darshangera@sssihl.edu.in
Purpose: FER on FERPlus using GACNN
'''

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
from model import Net
from ferplus_dataset import ImageList
import scipy.io as sio
import numpy as np
import pdb
from statistics import mean 
os.environ['CUDA_VISIBLE_DEVICES'] = '0,1,2'

model_names = sorted(name for name in models.__dict__
    if name.islower() and not name.startswith("__")
    and callable(models.__dict__[name]))


parser = argparse.ArgumentParser(description='PyTorch Ferplus Training using novel attention+region branches')

parser.add_argument('--root_path', type=str, default='../data/FERPLUS/Dataset/',
                    help='path to root path of images')
parser.add_argument('--database', type=str, default='FERPLUS',
                    help='Which Database for train. (Flatcam, FERPLUS)')


parser.add_argument('-j', '--workers', default=16, type=int, metavar='N',
                    help='number of data loading workers (default: 4)')
parser.add_argument('--epochs', default=100, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('-b', '--batch-size', default=128, type=int,
                    metavar='N', help='mini-batch size (default: 256)')
parser.add_argument('-b_t', '--batch-size_t', default=128, type=int,
                    metavar='N', help='mini-batch size (default: 256)')
parser.add_argument('--lr', '--learning-rate', default=0.01, type=float,
                    metavar='LR', help='initial learning rate')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum')
parser.add_argument('--weight-decay', '--wd', default=1e-4, type=float,
                    metavar='W', help='weight decay (default: 1e-4)')
parser.add_argument('--print-freq', '-p', default=100, type=int,
                    metavar='N', help='print frequency (default: 10)')
parser.add_argument('--resume', default='', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')
parser.add_argument('-e', '--evaluate', dest='evaluate', action='store_true',
                    help='evaluate model on validation set')
parser.add_argument('--model_dir','-m', default='checkpoints', type=str)

parser.add_argument('--train_list', type=str, default = '../data/FERPLUS/Dataset/Labels/FER2013TrainValid/label.csv',
                    help='path to training list')
parser.add_argument('--valid_list', type=str, default = '../data/FERPLUS/Dataset/Labels/FER2013Valid/label.csv',
                    help='path to validation list')
parser.add_argument('--test_list', type=str, default = '../data/FERPLUS/Dataset/Labels/FER2013Test/label.csv',
                    help='path to test list')

parser.add_argument('--train_landmarksfile', type=str, default = '../data/FERPLUS/Dataset/FERPLUS_train_valid_landmarks.pkl',
                    help='path to landmarksdictionary')
parser.add_argument('--test_landmarksfile', type=str, default = '../data/FERPLUS/Dataset/FERPLUS_test_landmarks.pkl',
                    help='path to landmarksdictionary')
parser.add_argument('--valid_landmarksfile', type=str, default = '../data/FERPLUS/Dataset/FERPLUS_valid_landmarks.pkl',
                    help='path to landmarksdictionary')

parser.add_argument('--imagesize', type=int, default = 224, help='image size (default: 224)')

parser.add_argument('--end2end', default=True,\
        help='if true, using end2end with dream block, else, using naive architecture')

parser.add_argument('--num_classes', type=int, default=8,
                    help='number of expressions(class)')

parser.add_argument('--num_regions', type=int, default=24,
                    help='number of regions(crops)')


best_prec1 = 0

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def main():
    global args, best_prec1
    args = parser.parse_args()
    print('\n\t\t\t\t Aum Sri Sai Ram\nFER on FERPLUS using GACNN\n\n')
    print('img_dir:', args.root_path)
    

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

    train_data = ImageList(root=args.root_path +'Images/FER2013TrainValid/' ,landmarksfile=args.train_landmarksfile, fileList=args.train_list,
                  transform=train_transform, imagesize = args.imagesize, mode='majority')


    train_loader = torch.utils.data.DataLoader(
        train_data,
        batch_size=args.batch_size, shuffle=True,
        num_workers=args.workers, pin_memory=True)
    

    test_data = ImageList(root=args.root_path+'Images/FER2013Test/', landmarksfile=args.test_landmarksfile,fileList=args.test_list,
                  transform=valid_transform, mode='majority')


    
    test_loader = torch.utils.data.DataLoader(test_data, batch_size=args.batch_size_t, shuffle=False,
                                           num_workers=args.workers, pin_memory=True)

    print('length of  train+valid Database for training: ' + str(len(train_loader.dataset)))

    print('length of  test Database: ' + str(len(test_loader.dataset)))


    
    # prepare model
    basemodel = torch.nn.DataParallel(Net(num_classes=args.num_classes)).to(device)
    
    criterion = nn.CrossEntropyLoss().to(device)

    criterion1 = criterion#MarginLoss(loss_lambda=0.5).to(device)
    '''
    optimizer1 = torch.optim.SGD(filter(lambda p: p.requires_grad, model.parameters()), args.lr,
                                 momentum=args.momentum,
                                 weight_decay=args.weight_decay)
    '''
    optimizer1 =  torch.optim.SGD([{"params": basemodel.parameters(), "lr": args.lr, "momentum":args.momentum,
                                 "weight_decay":args.weight_decay}])
    '''                              
    optimizer1.add_param_group({"params": attention_model.parameters(), "lr": args.lr, "momentum":args.momentum,
                                 "weight_decay":args.weight_decay})
    '''
    
    
        
    
    if args.resume:
        if os.path.isfile(args.resume):
            print("=> loading checkpoint '{}'".format(args.resume))
            checkpoint = torch.load(args.resume)
            args.start_epoch = checkpoint['epoch']
            best_prec1 = checkpoint['best_prec1']
            basemodel.load_state_dict(checkpoint['base_state_dict'])
            
            optimizer1.load_state_dict(checkpoint['optimizer1'])
            
            print("=> loaded checkpoint '{}' (epoch {})"
                  .format(args.resume, checkpoint['epoch']))
        else:
            print("=> no checkpoint found at '{}'".format(args.resume))

    #cudnn.benchmark = True

    #pdb.set_trace()
    '''
    print ('args.evaluate',args.evaluate)
    if args.evaluate:
        prec1 = validate(val_loader, model, criterion, criterion1, valid = True)
        test_prec1 = validate(test_loader, model, criterion, criterion1, valid = False)
        print("Epoch: {} Valid Acc: {} and Test Acc: {}".format(epoch, prec1,test_prec1))
        return
    '''
    print('Training starting:\n')
    for epoch in range(args.start_epoch, args.epochs):
        if epoch > 0:
           adjust_learning_rate(optimizer1, epoch)

        # train for one epoch
        
        train(train_loader, basemodel, criterion,  optimizer1,  epoch)
        
        prec1 = validate(test_loader, basemodel, criterion, optimizer1,  epoch)
        
        
        
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
    torch.save(state, full_filename)
    if is_best:
        shutil.copyfile(full_filename, full_bestname)


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

"""
def adjust_learning_rate(optimizer1, epoch):
    
    if epoch in [int(args.epochs*0.2), int(args.epochs*0.3), int(args.epochs*0.4)]: #change to 10
        print('Adjusted learning rate')
        for param_group in optimizer1.param_groups:
            param_group['lr'] *= 0.1
"""

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
