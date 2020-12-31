'''
Aum Sri Sai Ram
Authors: Darshan Gera and Dr. S. Balasubramanian, SSSIHL
Date: 16-12-2020
Email: darshangera@sssihl.edu.in
Purpose: FER on SFEW using OADN pretrained on affectnet
Ref: Occlusion-Adaptive Deep Network for Robust Facial Expression Recognition(2020)
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
import util
#dataset class and model 

import scipy.io as sio
import numpy as np
import pdb
from statistics import mean 
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
from models.attentionnet import LandmarksAttentionBranch as AttentionBranch, RegionBranch, count_parameters
from models.resnet import resnet50
from dataset.sfew_dataset_attentionmaps import ImageList




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

parser.add_argument('--epochs', default=60, type=int, metavar='N', help='number of total epochs to run')

parser.add_argument('--start-epoch', default=0, type=int, metavar='N', help='manual epoch number (useful on restarts)')

parser.add_argument('-b', '--batch-size', default=64, type=int, metavar='N', help='mini-batch size (default: 256)')

parser.add_argument('-b_t', '--batch-size_t', default=64, type=int, metavar='N', help='mini-batch size (default: 256)')

parser.add_argument('--lr', '--learning-rate', default=0.0001, type=float, metavar='LR', help='initial learning rate')

parser.add_argument('--momentum', default=0.9, type=float, metavar='M',  help='momentum')

parser.add_argument('--weight-decay', '--wd', default=1e-4, type=float,  metavar='W', help='weight decay (default: 1e-4)')

parser.add_argument('--print-freq', '-p', default=1000, type=int,metavar='N', help='print frequency (default: 10)')

parser.add_argument('--resume', default='checkpoints_affectnet7/model_best.pth.tar', type=str, metavar='PATH',   help='path to latest checkpoint (default: none)')

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
    print('\n\t\t\t\t Aum Sri Sai Ram\nFER on SFEW using OADN\n\n')
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

    val_data = ImageList(root=args.root_path +'Val_Aligned_Faces_SAN/', fileList=args.valid_list,landmarksfile = args.test_landmarksfile,
                  transform=valid_transform)   
    
    val_loader = torch.utils.data.DataLoader(val_data, args.batch_size, shuffle=False, num_workers=8)

    train_dataset = ImageList(root=args.root_path+'Train_Aligned_Faces_SAN/', fileList = args.train_list,landmarksfile = args.train_landmarksfile,
                  transform=train_transform)


    
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
    basemodel = resnet50(pretrained = False)
    attention_model = AttentionBranch(inputdim = 2048, num_maps = 24, num_classes = args.num_classes)
    region_model = RegionBranch(inputdim = 2048, num_regions = 4, num_classes = args.num_classes)

    basemodel = torch.nn.DataParallel(basemodel).to(device)
    attention_model = torch.nn.DataParallel(attention_model).to(device)
    region_model = torch.nn.DataParallel(region_model).to(device)

    print('\nNumber of parameters:')
    print('Base Model: {}, Attention Branch:{}, Region Branch:{} and Total: {}'.format(count_parameters(basemodel),count_parameters(attention_model),  count_parameters(region_model), count_parameters(basemodel)+count_parameters(attention_model)+count_parameters(region_model)))  
    
    criterion = nn.CrossEntropyLoss().to(device)

    
    
    optimizer =  torch.optim.SGD([{"params": region_model.parameters(), "lr": args.lr, "momentum":args.momentum,
                                 "weight_decay":args.weight_decay}])
        
    optimizer.add_param_group({"params": attention_model.parameters(), "lr": args.lr, "momentum":args.momentum,
                                 "weight_decay":args.weight_decay})
    optimizer.add_param_group({"params": basemodel.parameters(), "lr": 0.0001, "momentum":args.momentum,
                                 "weight_decay":args.weight_decay})
    
    if args.pretrained:
        
        util.load_state_dict(basemodel,'pretrainedmodels/vgg_msceleb_resnet50_ft_weight.pkl')
        
    if args.resume:
        if os.path.isfile(args.resume):
            print("=> loading checkpoint '{}'".format(args.resume))
            checkpoint = torch.load(args.resume)
            basemodel.load_state_dict(checkpoint['base_state_dict'])            
            print("=> loaded checkpoint '{}' (epoch {})"
                  .format(args.resume, checkpoint['epoch']))
            
        else:
            print("=> no checkpoint found at '{}'".format(args.resume))

    print('\nTraining starting:\n')
    for epoch in range(args.start_epoch, args.epochs):

        train(train_loader, basemodel, attention_model, region_model, criterion,  optimizer, epoch)
        
        prec1 = validate(val_loader, basemodel, attention_model, region_model, criterion,  epoch)
        
        
        print("Epoch: {}   Test Acc: {}".format(epoch, prec1))

        is_best = prec1 > best_prec1

        best_prec1 = max(prec1.to(device).item(), best_prec1)
        
        adjust_learning_rate(optimizer, epoch)
        
        save_checkpoint({
            'epoch': epoch + 1,            
            'base_state_dict': basemodel.state_dict(),
            'attention_state_dict': attention_model.state_dict(),
            'region_state_dict': region_model.state_dict(),
            'best_prec1': best_prec1,
            'optimizer' : optimizer.state_dict(),
        }, is_best.item())
        
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

    for i, (input, target,attmaps) in enumerate(train_loader):
        # measure data loading time
        data_time.update(time.time() - end)

        input = input.to(device)
        attmaps = attmaps.to(device)
        target = target.to(device)
        feature = basemodel(input)
        
        attention_preds = attention_model(feature, attmaps)

        region_preds = region_model(feature)
        
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
        
        #all_predictions = torch.cat([attention_preds.unsqueeze(2), region_preds], dim=2)
        all_predictions = region_preds
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
                  #'Time  ({batch_time.avg})\t'
                  #'Data ({data_time.avg})\t'
                  'att_loss  ({att_loss.avg})\t'
                  'region_loss ({region_loss.avg})\t'
                  'overall_loss ({overall_loss.avg})\t' 
                  'Prec1  ({top1.avg}) \t'.format(
                   epoch, i, len(train_loader), 
#                   epoch, i, len(train_loader), batch_time = batch_time, data_time=data_time, 
                  att_loss=att_loss,region_loss=region_loss,overall_loss=overall_loss,  top1=top1))



def validate(val_loader,  basemodel, attention_model, region_model, criterion, epoch):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    att_loss = AverageMeter()
    region_loss = AverageMeter()
    overall_loss = AverageMeter()
    #region_prec = []
    mode =  'Testing'
    # switch to evaluate mode
    basemodel.eval()
    attention_model.eval()
    region_model.eval()
    end = time.time()

    with torch.no_grad():         
        for i, (input, target, attmaps) in enumerate(val_loader):        
            data_time.update(time.time() - end)
            input = input.to(device) 
            target = target.to(device)
            attmaps = attmaps.to(device)
            feature = basemodel(input)
            attention_preds = attention_model(feature, attmaps)
            region_preds = region_model(feature) 
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
            
            all_predictions = region_preds

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

def save_checkpoint(state, is_best, filename='checkpoint.pth.tar'):
    epoch_num = state['epoch']
    full_filename = os.path.join(args.model_dir, str(epoch_num)+'_'+ filename)
    full_bestname = os.path.join(args.model_dir, 'model_best.pth.tar')
    torch.save(state, full_filename)

    if epoch_num%1==0 and epoch_num>=0:
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


def adjust_learning_rate(optimizer, epoch):
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
    #lr = args.lr * (0.1 ** (epoch // 30))
    
    #if epoch in [int(args.epochs*0.3), int(args.epochs*0.5), int(args.epochs*0.8)]: #change to 10
    #if epoch in [20]:#, int(args.epochs*0.3), int(args.epochs*0.4)]: #change to 10
    print('\n******************************\n\tAdjusted learning rate: \n')
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



if __name__ == '__main__':
    main()
