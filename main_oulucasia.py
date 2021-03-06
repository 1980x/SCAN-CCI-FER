'''
Aum Sri Sai Ram
Implementation of Landmark Guidance Independent Spatio-channel Attention and
Complementary Context Information based Facial Expression Recognition on OuluCASIA dataset for 10 fold crossvalidation
https://arxiv.org/pdf/2007.10298v1.pdf
Authors: Darshan Gera and Dr. S. Balasubramanian, SSSIHL
Date: 20-07-2020
Email: darshangera@sssihl.edu.in
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
import utils.util as util
#dataset class and model 

import scipy.io as sio
import numpy as np
import pdb
from statistics import mean 

os.environ['CUDA_VISIBLE_DEVICES'] = '0'

from models.attentionnet import AttentionBranch, RegionBranch, count_parameters
from models.resnet import resnet50
from dataset.oulucasia_dataset_cv import ImageList
from dataset.sampler import ImbalancedDatasetSampler


#######################################################################################################################################
# Training settings
parser = argparse.ArgumentParser(description='OULUCASIA Cross Validation expression recognition')

# DATA

parser.add_argument('--root_path', type=str, default='../data/AuthorOluCasia/ourOluCasiatest/',
                    help='path to root path of images')

parser.add_argument('--database', type=str, default='OULUCASIA Cross Validation',
                    help='Which Database for train. ()')

parser.add_argument('--epochs', default=40, type=int, metavar='N', help='number of total epochs to run')

parser.add_argument('--folds', default=10, type=int, metavar='N', help='cross validation folds')

parser.add_argument('--metafile', type=str, default = '../data/AuthorOluCasia/ourOluCasia_labelstest.txt',
                    help='path to list containing all files')


parser.add_argument('--start-epoch', default=0, type=int, metavar='N', help='manual epoch number (useful on restarts)')

parser.add_argument('-b', '--batch-size', default=48, type=int, metavar='N', help='mini-batch size (default: 256)')

parser.add_argument('-b_t', '--batch-size_t', default=128, type=int, metavar='N', help='mini-batch size (default: 256)')

parser.add_argument('--lr', '--learning-rate', default=0.00001, type=float, metavar='LR', help='initial learning rate')

parser.add_argument('--momentum', default=0.9, type=float, metavar='M',  help='momentum')

parser.add_argument('--weight-decay', '--wd', default=1e-3, type=float,  metavar='W', help='weight decay (default: 1e-4)')

parser.add_argument('--print-freq', '-p', default=1000, type=int,metavar='N', help='print frequency (default: 10)')

parser.add_argument('--resume', default='checkpoints/fedro_model_best.pth.tar', type=str, metavar='PATH',   help='path to')

parser.add_argument('--pretrained', default='checkpoints/fedro_model_best.pth.tar', type=str, metavar='PATH', 
                    help='path to latest checkpoint (default: none)')

parser.add_argument('-e', '--evaluate', dest='evaluate', action='store_true', help='evaluate model on validation set')


parser.add_argument('--model_dir','-m', default='checkpoints', type=str)

parser.add_argument('--imagesize', type=int, default = 224, help='image size (default: 224)')

parser.add_argument('--end2end', default=True,\
        help='if true, using end2end with dream block, else, using naive architecture')

parser.add_argument('--num_classes', type=int, default=7, help='number of expressions(class)')

parser.add_argument('--num_attentive_regions', type=int, default=25, help='number of non-overlapping patches(default:25)')

parser.add_argument('--num_regions', type=int, default=4, help='number of non-overlapping patches(default:4)')

parser.add_argument('--train_rule', default='Resample', type=str, help='data sampling strategy for train loader:Resample, DRW,Reweight, None')

parser.add_argument('--loss_type', default="CE", type=str, help='loss type:Focal, CE')


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

parser.add_argument('--workers', type=int, default = 8,
                    help='how many workers to load data')


args = parser.parse_args()


#######################################################################################################################################
def main():
    #Print args
    global args, best_prec1
    args = parser.parse_args()
    print('\n\t\t\t\t Aum Sri Sai Ram\nFER on OULUCASIA using Local and global Attention along with region branch (non-overlapping patches)\n\n')
    print(args)
    print('\nimg_dir: ', args.root_path)
    print('\ntrain rule: ',args.train_rule, ' and loss type: ', args.loss_type, '\n')

    mean = [0.5, 0.5, 0.5]
    std = [0.5, 0.5, 0.5]
    imagesize = args.imagesize
    train_transform = transforms.Compose([
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ColorJitter(brightness=0.4, contrast = 0.3, saturation = 0.25, hue = 0.05),    
            transforms.Resize((args.imagesize, args.imagesize)),
            transforms.ToTensor(),
            transforms.Normalize(mean,std)
        ])

    
    valid_transform = transforms.Compose([
            transforms.Resize((args.imagesize,args.imagesize)),
            transforms.ToTensor(),
            transforms.Normalize(mean,std)
        ])

        

   
    # prepare model
   
    
    for fold in range(0,args.folds):
        print('\n***************************************************************************************\n\nFold is ',fold) 
        basemodel = resnet50(pretrained = False)
        attention_model = AttentionBranch(inputdim = 512, num_regions = args.num_attentive_regions, num_classes = args.num_classes)
        region_model = RegionBranch(inputdim = 1024, num_regions = args.num_regions, num_classes = args.num_classes)

        basemodel = torch.nn.DataParallel(basemodel).to(device)
        attention_model = torch.nn.DataParallel(attention_model).to(device)
        region_model = torch.nn.DataParallel(region_model).to(device)

        print('\nNumber of parameters:')
        print('Base Model: {}, Attention Branch:{}, Region Branch:{} and Total: {}'.format(count_parameters(basemodel),count_parameters(attention_model),  count_parameters(region_model), count_parameters(basemodel)+count_parameters(attention_model)+count_parameters(region_model)))  

        best_prec1 = 0

        optimizer =  torch.optim.SGD([{"params": basemodel.parameters(), "lr": 0.00001, "momentum":args.momentum,
                                 "weight_decay":args.weight_decay}])
    
        optimizer.add_param_group({"params": attention_model.parameters(), "lr": args.lr, "momentum":args.momentum,
                                 "weight_decay":args.weight_decay})
    
        optimizer.add_param_group({"params": region_model.parameters(), "lr": args.lr, "momentum":args.momentum,
                                 "weight_decay":args.weight_decay})
             
        if args.pretrained:
            if os.path.isfile(args.pretrained):
               print("=> loading checkpoint '{}'".format(args.resume))
               checkpoint = torch.load(args.resume)
               basemodel.load_state_dict(checkpoint['base_state_dict'])
               attention_model.load_state_dict(checkpoint['attention_state_dict'])
               region_model.load_state_dict(checkpoint['region_state_dict'])
               optimizer.load_state_dict(checkpoint['optimizer'])
               print("=> loaded checkpoint '{}' (epoch {})"
                  .format(args.resume, checkpoint['epoch']))
            else:
               print("=> no checkpoint found at '{}'".format(args.resume))            
        
        #root, fileList, fold, is_train = True, num_classes = 7,  transform=None,
        val_data = ImageList(root=args.root_path , fileList=args.metafile,is_train= False, fold = fold, num_classes = args.num_classes,
                  transform=valid_transform)   
    
        val_loader = torch.utils.data.DataLoader(val_data, args.batch_size, shuffle=False, num_workers=8)

        train_dataset = ImageList(root=args.root_path, fileList=args.metafile,is_train= True, fold = fold, num_classes = args.num_classes,
                  transform=train_transform)

        cls_num_list = train_dataset.get_cls_num_list()
        

        if args.train_rule == 'None':
           train_sampler = None  
           per_cls_weights = None 
        elif args.train_rule == 'Resample':
           train_sampler = ImbalancedDatasetSampler(train_dataset)
           per_cls_weights = None
        elif args.train_rule == 'Reweight':
           train_sampler = None
           beta = 0.9999                 #1:normal weighting
           effective_num = 1.0 - np.power(beta, cls_num_list)
           per_cls_weights = (1.0 - beta) / np.array(effective_num)
           per_cls_weights = per_cls_weights / np.sum(per_cls_weights) * len(cls_num_list)
           per_cls_weights = torch.FloatTensor(per_cls_weights).to(device)
    
        if args.loss_type == 'CE':
           criterion = nn.CrossEntropyLoss(weight=per_cls_weights).to(device)
        elif args.loss_type == 'Focal':
           criterion = FocalLoss(weight=per_cls_weights, gamma=2).to(device)
        else:
           warnings.warn('Loss type is not listed')
           return
  
        train_loader = torch.utils.data.DataLoader(train_dataset, args.batch_size, shuffle=(train_sampler is None),
                                                   num_workers=args.workers, pin_memory=True, sampler=train_sampler)    
    
        print('length of train Database: ' + str(len(train_dataset)))
        print('length of test Database: ' + str(len(val_loader.dataset)))

        print('\nTraining starting: ', fold)
        
        for epoch in range(args.start_epoch, args.epochs):
            train(train_loader, basemodel, attention_model, region_model, criterion,  optimizer, epoch)
            #adjust_learning_rate(optimizer, epoch)
            prec1 = validate(val_loader, basemodel, attention_model, region_model, criterion,  epoch)
            print("Epoch: {}   Test Acc: {}".format(epoch, prec1))
            # remember best prec@1 and save checkpoint
            is_best = prec1 > best_prec1

            best_prec1 = max(prec1.to(device).item(), best_prec1)
            if is_best:
               print('\n*********************************\nBest so far is :\t',best_prec1)
        
            save_checkpoint({
            'epoch': epoch + 1,            
            'base_state_dict': basemodel.state_dict(),
            'attention_state_dict': attention_model.state_dict(),
            'region_state_dict': region_model.state_dict(),
            'best_prec1': best_prec1,
            'optimizer' : optimizer.state_dict(),
            },  is_best.item(),'checkpoint.pth.tar', fold)
        
def train(train_loader,  basemodel, attention_model, region_model, criterion,  optimizer, epoch):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top2 = AverageMeter()
    top5 = AverageMeter()
    att_loss = AverageMeter()
    region_loss = AverageMeter()
    overall_loss = AverageMeter()
    region_prec = []
     
    end = time.time()

    for i, (input, target) in enumerate(train_loader):
        # measure data loading time
        data_time.update(time.time() - end)
        input = input.to(device)
        target = target.to(device)
        
        #compute output 
        attention_branch_feat, region_branch_feat = basemodel(input)      
        local_features_list, global_features, attention_preds = attention_model(attention_branch_feat)
        region_preds = region_model(region_branch_feat)

        #Attention Branch Loss: loss1
        loss1 = criterion(attention_preds, target) #attention CELoss
        #Region Branch Loss: loss2        
        for j in range(4):
            if j == 0:
               loss2 = criterion(region_preds[:,:,j], target) #region celoss loss from Ist region branch 
            else:
               loss2 += criterion(region_preds[:,:,j], target) #region celoss loss for rest 3 regions from region branch
            
        att_loss.update(loss1.item(), input.size(0))
        region_loss.update(loss2.item(), input.size(0))

        att_wt = 0.2
        loss = att_wt * loss1 + (1 - att_wt) *loss2 # weights for both branches
        overall_loss.update(loss.item(), input.size(0))
        all_predictions = torch.cat([attention_preds.unsqueeze(2), region_preds], dim=2)
        avg_predictions = torch.mean(all_predictions, dim=2)
        avg_prec = accuracy(avg_predictions,target,topk=(1,))        
        top1.update(avg_prec[0], input.size(0))

        # compute gradient and do SGD step
        optimizer.zero_grad()
        
        loss.backward()
        
        optimizer.step()
        
        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0:
            print('Training Epoch: [{0}][{1}/{2}]\t'
                  'Time  ({batch_time.avg})\t'
                  'Data ({data_time.avg})\t'
                  'att_loss  ({att_loss.avg})\t'
                  'region_loss ({region_loss.avg})\t'
                  'overall_loss ({overall_loss.avg})\t' 
                  'Prec1  ({top1.avg}) \t'.format(
                  epoch, i, len(train_loader), batch_time = batch_time, data_time=data_time, 
                  att_loss=att_loss,region_loss=region_loss,overall_loss=overall_loss,  top1=top1))



def validate(val_loader,  basemodel, attention_model, region_model, criterion,  epoch):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    att_loss = AverageMeter()
    region_loss = AverageMeter()
    overall_loss = AverageMeter()
    mode =  'Testing'
    # switch to evaluate mode
    basemodel.eval()
    attention_model.eval()
    region_model.eval()
    end = time.time()

    with torch.no_grad():         
        for i, (input, target) in enumerate(val_loader):        
            data_time.update(time.time() - end)
            input = input.to(device) 
            target = target.to(device)
            attention_branch_feat, region_branch_feat = basemodel(input)
            local_features_list, global_features, attention_preds = attention_model(attention_branch_feat)
            region_preds = region_model(region_branch_feat)    
            #Attention Branch Loss: loss1
            loss1 = criterion(attention_preds, target) #attention CELoss
            #Region Branch Loss: loss2        
            for j in range(4):
                if j == 0:
                   loss2 = criterion(region_preds[:,:,j], target) #region celoss loss from Ist region branch 
                else:
                   loss2 += criterion(region_preds[:,:,j], target) #region celoss loss for rest 3 regions from region branch
                
            att_loss.update(loss1.item(), input.size(0))
            region_loss.update(loss2.item(), input.size(0))

            att_wt = 0.2
            loss = att_wt * loss1 + (1 - att_wt) * loss2 # weights for both branches

            overall_loss.update(loss.item(), input.size(0))
            all_predictions = torch.cat([attention_preds.unsqueeze(2), region_preds], dim=2)
            avg_predictions = torch.mean(all_predictions, dim=2)
            avg_prec = accuracy(avg_predictions,target,topk=(1,))       
            top1.update(avg_prec[0], input.size(0))

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()            
            
        print('{0} [{1}/{2}]\t'
                  #'Time {batch_time.val} ({batch_time.avg})\t'
                  'att_loss  ({att_loss.avg})\t'
                  'region_loss ({region_loss.avg})\t'
                  'overall_loss ({overall_loss.avg})\t' 
                  'Prec@1  ({top1.avg})\t'
                  .format(mode, i, len(val_loader),  att_loss=att_loss, region_loss=region_loss, overall_loss=overall_loss,  top1=top1))


    return top1.avg





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

def adjust_learning_rate(optimizer, epoch):
    """Sets the learning rate to the initial LR decayed"""
    for param_group in optimizer.param_groups:
           param_group['lr'] *= 0.95



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

def save_checkpoint(state, is_best, filename='checkpoint.pth.tar', fold = 0):
    epoch_num = state['epoch']
    full_bestname = os.path.join(args.model_dir, str(fold)+' model_best.pth.tar')
    if is_best:
       torch.save(state, full_bestname)
        


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


def adjust_learning_rate(optimizer, epoch):
   
        for param_group in optimizer.param_groups:
            param_group['lr'] *= 0.95



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




if __name__ == "__main__":
    
    main()
    print("Process has finished!")
   
     
    




    

