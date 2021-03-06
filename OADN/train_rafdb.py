'''
Aum Sri Sai Ram
Authors: Darshan Gera and Dr. S. Balasubramanian, SSSIHL
Date: 16-12-2020
Email: darshangera@sssihl.edu.in
Purpose: FER on RAFDB using OADN
Ref: Occlusion-Adaptive Deep Network for Robust Facial Expression Recognition(2020)
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

from models.attentionnet import LandmarksAttentionBranch, RegionBranch,  count_parameters
from models.resnet import resnet50
from dataset.rafdb_dataset_attentionmaps import ImageList

import scipy.io as sio
import numpy as np
import pdb

os.environ['CUDA_VISIBLE_DEVICES'] = '0'

model_names = sorted(name for name in models.__dict__
    if name.islower() and not name.startswith("__")
    and callable(models.__dict__[name]))


parser = argparse.ArgumentParser(description='PyTorch RAFDB Training using attention maps branch + region branch with landmark score based attention maps')

parser.add_argument('--root_path', type=str, default='../data/RAFDB/Image/aligned/',
                    help='path to root path of images')
parser.add_argument('--database', type=str, default='RAFDB',
                    help='Which Database for train. (RAFDB, Flatcam, FERPLUS)')
parser.add_argument('--train_list', type=str, default = '../data/RAFDB/EmoLabel/train_label.txt',
                    help='path to training list')
parser.add_argument('--test_list', type=str, default = '../data/RAFDB/EmoLabel/test_label.txt',
                    help='path to test list')
parser.add_argument('--landmarksfile', type=str, default = '../data/RAFDB/RAFDB_landmarks_scores.pkl',
                    help='path to landmarksdictionary')

parser.add_argument('--arch', '-a', metavar='ARCH', default='resnet50', choices=model_names,
                    help='model architecture: ' +
                        ' | '.join(model_names) +
                        ' (default: resent18)')

parser.add_argument('-j', '--workers', default=16, type=int, metavar='N',   help='number of data loading workers (default: 4)')

parser.add_argument('--epochs', default=60, type=int, metavar='N', help='number of total epochs to run')

parser.add_argument('--start-epoch', default=0, type=int, metavar='N', help='manual epoch number (useful on restarts)')

parser.add_argument('-b', '--batch-size', default=64, type=int, metavar='N', help='mini-batch size (default: 256)')

parser.add_argument('-b_t', '--batch-size_t', default=64, type=int, metavar='N', help='mini-batch size (default: 256)')

parser.add_argument('--lr', '--learning-rate', default=0.001, type=float, metavar='LR', help='initial learning rate')

parser.add_argument('--momentum', default=0.9, type=float, metavar='M',  help='momentum')

parser.add_argument('--weight-decay', '--wd', default=1e-4, type=float,  metavar='W', help='weight decay (default: 1e-4)')

parser.add_argument('--print-freq', '-p', default=50, type=int,metavar='N', help='print frequency (default: 10)')

parser.add_argument('--resume', default='checkpoints_rafdb/model_best.pth.tar', type=str, metavar='PATH',   help='path to latest checkpoint (default: none)')

parser.add_argument('--pretrained', default='pretrainedmodels/resnet50-19c8e357.pth', type=str, metavar='PATH', 
                    help='path to latest checkpoint (default: none)')

parser.add_argument('-e', '--evaluate', dest='evaluate', action='store_true', 
                    help='evaluate model on validation set')

parser.add_argument('--model_dir','-m', default='checkpoints_rafdb', type=str)

parser.add_argument('--end2end', default=True,
        help='if true, using end2end with dream block, else, using naive architecture')

parser.add_argument('--imagesize', type=int, default = 224, help='image size (default: 224)')

parser.add_argument('--num_classes', type=int, default=7, help='number of expressions(class)')

parser.add_argument('--num_attentive_regions', type=int, default=25, help='number of non-overlapping patches(default:25)')

parser.add_argument('--num_regions', type=int, default=4, help='number of non-overlapping patches(default:4)')
best_prec1 = 0

best_prec1 = 0

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def main():
    global args, best_prec1
    args = parser.parse_args()
    print('\n\t\t Aum Sri Sai Ram\n\t\tRAFDB FER using  Attention branch based on gaussian maps with region branch\n\n')
    print(args)
    
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

    train_dataset = ImageList(root=args.root_path, landmarksfile =  args.landmarksfile, fileList=args.train_list,
                  transform=train_transform)

    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=args.batch_size, shuffle=True,
        num_workers=args.workers, pin_memory=True)
    
    test_data = ImageList(root=args.root_path, fileList=args.test_list,landmarksfile =  args.landmarksfile, 
                  transform=valid_transform)

    test_loader = torch.utils.data.DataLoader(test_data, batch_size=args.batch_size_t, 
                         shuffle=False, num_workers=args.workers, pin_memory=True)

    print('length of RAFDB train Database: ' + str(len(train_dataset)))

    print('length of RAFDB test Database: ' + str(len(test_loader.dataset)))

    
    # prepare model
    # prepare model
    basemodel = resnet50(pretrained = False)
    attention_model = LandmarksAttentionBranch(inputdim = 1024, num_maps = 24, num_classes = args.num_classes)
    region_model = RegionBranch(inputdim = 1024, num_regions = args.num_regions, num_classes = args.num_classes)

    basemodel = torch.nn.DataParallel(basemodel).to(device)
    attention_model = torch.nn.DataParallel(attention_model).to(device)
    region_model = torch.nn.DataParallel(region_model).to(device)

    print('\nNumber of parameters:')
    print('Base Model: {}, Attention Branch:{}, Region Branch:{} and Total: {}'.format(count_parameters(basemodel),count_parameters(attention_model),  count_parameters(region_model), count_parameters(basemodel)+count_parameters(attention_model)+count_parameters(region_model)))  
    
    criterion = nn.CrossEntropyLoss().to(device)
    
    
    optimizer1 =  torch.optim.SGD([{"params": basemodel.parameters(), "lr": args.lr, "momentum":args.momentum,
                                 "weight_decay":args.weight_decay}])
    
    optimizer1.add_param_group({"params": attention_model.parameters(), "lr": args.lr, "momentum":args.momentum,
                                 "weight_decay":args.weight_decay})
    
    
    optimizer1.add_param_group({"params": region_model.parameters(), "lr": args.lr, "momentum":args.momentum,
                                 "weight_decay":args.weight_decay})
    
    if args.pretrained:

        pretrained_state_dict = torch.load('pretrainedmodels/resnet50-19c8e357.pth')
        model_state_dict = basemodel.state_dict()        
        #print(model_state_dict.keys()) 
        for key in pretrained_state_dict:
            if  ((key=='fc.weight')|(key=='fc.bias')|(key == 'feature.weight')|(key == 'feature.bias')| (key.find('layer4.')>-1) ):
                pass
            else:
                #print(key) 
                model_state_dict['module.'+key] = pretrained_state_dict[key]

        basemodel.load_state_dict(model_state_dict, strict = True)
        print('\nLoaded resent50 pretrained on imagenet.\n')

        
    
    if args.resume:
        if os.path.isfile(args.resume):
            print("=> loading checkpoint '{}'".format(args.resume))
            checkpoint = torch.load(args.resume)
            args.start_epoch = checkpoint['epoch']
            best_prec1 = checkpoint['best_prec1']
            basemodel.load_state_dict(checkpoint['base_state_dict'])
            attention_model.load_state_dict(checkpoint['attention_state_dict'])
            region_model.load_state_dict(checkpoint['region_state_dict'])
            optimizer1.load_state_dict(checkpoint['optimizer1'])
            print("=> loaded checkpoint '{}' (epoch {})"
                  .format(args.resume, checkpoint['epoch']))
        else:
            print("=> no checkpoint found at '{}'".format(args.resume))

    prec1 = validate(test_loader, basemodel, attention_model, region_model, criterion,0)
    print("Epoch: {}   Test Acc: {}".format(0, prec1))
    assert(False)
        
    print('\nTraining starting:\n')
    for epoch in range(args.start_epoch, args.epochs):



        # train for one epoch
        adjust_learning_rate(optimizer1, epoch)

        # train for one epoch
        
        train(train_loader, basemodel, attention_model, region_model, criterion,optimizer1, epoch)
        prec1 = validate(test_loader, basemodel, attention_model, region_model, criterion,   epoch)
        print("Epoch: {}   Test Acc: {}".format(epoch, prec1))
        # remember best prec@1 and save checkpoint
        is_best = prec1 > best_prec1

        best_prec1 = max(prec1.to(device).item(), best_prec1)
        
        save_checkpoint({
            'epoch': epoch + 1,
            'arch': args.arch,
            'base_state_dict': basemodel.state_dict(),
            'attention_state_dict': attention_model.state_dict(),
            'region_state_dict': region_model.state_dict(),
            'best_prec1': best_prec1,
            'optimizer1' : optimizer1.state_dict(),
            #'optimizer2' : optimizer2.state_dict(),
        }, is_best.item())
        
def train(train_loader,  basemodel, attention_model, region_model, criterion, optimizer1, epoch):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    overall_loss = AverageMeter()
    top1 = AverageMeter()
    top2 = AverageMeter()
    top5 = AverageMeter()
    att_loss = AverageMeter()
    region_loss = AverageMeter()
    losses = AverageMeter()
    region_prec = []
     
    end = time.time()

    for i, (input, target, attention_maps) in enumerate(train_loader):
        # measure data loading time
        data_time.update(time.time() - end)

        input = input.to(device)
        attention_maps = attention_maps.to(device)
        target = target.to(device)
        #print(input.size(), target.size())
        # compute output
        attention_branch_feat, region_branch_feat = basemodel(input)
        
        attention_preds = attention_model(region_branch_feat, attention_maps)

        region_preds = region_model(region_branch_feat)
        #attention_preds, region_preds  
        #print(attention_preds.size(), region_preds.size())

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

        att_wt = 0
        loss = att_wt * loss1 + (1.0 - att_wt) *loss2 # weights for both branches
        overall_loss.update(loss.item(), input.size(0))

        # measure accuracy
        '''
        att_prec = accuracy(attention_preds, target, topk=(1,)) #attention_branch accuracy
        
        region_prec.append(att_prec[0]) #both attention prediction as well as region prediction are put in region_prec

        print(region_prec, len(region_prec), region_prec[0].dtype)

        prec_stacked = torch.stack(region_prec, dim=0) 

        avg_prec = torch.max(prec_stacked)  #Take mean  prediction
           
        #print(avg_prec,avg_prec.size())
        '''
        #Use average prediction score instead below
        all_predictions = region_preds#torch.cat([attention_preds.unsqueeze(2), region_preds ],dim=2)
        _,avg_predictions = torch.max(all_predictions, dim=2)
        avg_prec = accuracy(avg_predictions,target,topk=(1,))
        #end

        top1.update(avg_prec[0], input.size(0))

        # compute gradient and do SGD step
        optimizer1.zero_grad()
        
        loss.backward()
        
        optimizer1.step()
        

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



def validate(val_loader,  basemodel, attention_model, region_model, criterion,  epoch):
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
    mode =  'Testing'
    # switch to evaluate mode
    basemodel.eval()
    attention_model.eval()
    region_model.eval()
    end = time.time()

    with torch.no_grad():         
        for i, (input, target, attention_maps) in enumerate(val_loader):        
            data_time.update(time.time() - end)
            input = input.to(device) 
            target = target.to(device)
            attention_maps = attention_maps.to(device)
            attention_branch_feat, region_branch_feat = basemodel(input)
            attention_preds = attention_model(region_branch_feat, attention_maps)
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

            att_wt = 0
            loss = att_wt * loss1 + (1.0 - att_wt) *loss2 # weights for both branches

            overall_loss.update(loss.item(), input.size(0))
            #Use average prediction score instead below
            all_predictions = region_preds#torch.cat([attention_preds.unsqueeze(2), region_preds ],dim=2)
            avg_predictions = torch.mean(all_predictions, dim=2)
            avg_prec = accuracy(avg_predictions,target,topk=(1,))
            #end
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
    
    if epoch in [int(args.epochs*0.3), int(args.epochs*0.5), int(args.epochs*0.8)]: 
        print('\n******************************\n\tAdjusted learning rate: \n')
        for param_group in optimizer.param_groups:
            param_group['lr'] *= 0.1



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
