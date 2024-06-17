# -*- coding: utf-8 -*-
import gc
import time
import os
import math
import argparse
from glob import glob
from collections import OrderedDict
import random
import warnings
from datetime import datetime
from time import time

import numpy as np
from tqdm import tqdm

from sklearn.model_selection import train_test_split
# from sklearn.externals import joblib
import joblib
from skimage.io import imread

from math import cos, pi
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.utils.data import DataLoader
import torch.backends.cudnn as cudnn
import torchvision
from torchvision import datasets, models, transforms

from dataset import Dataset

# from hausdorff import hausdorff_distance as hausdorff_dist
from metrics import dice_coef, iou_score, sensitivity_score, accuracy_score
import losses
from utils import str2bool, count_params, AverageMeter
import pandas as pd

import MABTS

from torch.utils.tensorboard import SummaryWriter

torch.manual_seed(3407)

device = torch.device('cuda:5' if torch.cuda.is_available() else 'cpu')

# writer = SummaryWriter('log')
# model_name = 'f2t2'
model_name = 'f2t1'


load_weight = False  
model_pre_path = 'models/' + str(model_name) +'/model.pth'

# arch_names = ['__name__', '__doc__', 'nn', 'F', 'Downsample_block', 'Upsample_block', 'Unet']
arch_names = list(MABTS.__dict__.keys())
loss_names = list(losses.__dict__.keys())
loss_names.append('BCEWithLogitsLoss')

IMG_PATH = glob(r"/data/zh/BrainTumorSegmentation/data/processed/2D_2018/trainImage/*")
MASK_PATH = glob(r"/data/zh/BrainTumorSegmentation/data/processed/2D_2018/trainMask/*")

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--name', default=model_name, help='model name: (default: arch+timestamp)')
    parser.add_argument('--arch', '-a', metavar='ARCH', default='Net', choices=arch_names,
                        help='model architecture: ' + ' | '.join(arch_names) + ' (default: Net)')
    parser.add_argument('--deepsupervision', default=False, type=str2bool)
    
    parser.add_argument('--dataset', default="BraTS18", help='dataset name')
    parser.add_argument('--input_channels', default=1, type=int, help='input channels')
    parser.add_argument('--image-ext', default='png', help='image file extension')
    parser.add_argument('--mask-ext', default='png', help='mask file extension')
    parser.add_argument('--aug', default=False, type=str2bool)
    parser.add_argument('--loss', default='BCEDiceLoss', choices=loss_names,
                        help='loss: ' + ' | '.join(loss_names) + ' (default: BCEDiceLoss)')

    parser.add_argument('--epochs', default=100, type=int, metavar='N', help='number of total epochs to run')
    parser.add_argument('-b', '--batch_size', default=8, type=int, metavar='N', help='mini-batch size (default: 16)')  # '--batch-size'

    parser.add_argument('--early-stop', default=10, type=int, metavar='N', help='early stopping (default: 10)')
    parser.add_argument('--optimizer', default='Adam', choices=['Adam', 'SGD'],
                        help='loss: ' + ' | '.join(['Adam', 'SGD']) + ' (default: Adam)')
    parser.add_argument('--lr', '--learning-rate', default=1e-4, type=float, metavar='LR', help='initial learning rate')
    parser.add_argument('--weight-decay', default=1e-4, type=float, help='weight decay')

    parser.add_argument('--momentum', default=0.9, type=float, help='momentum')
    parser.add_argument('--nesterov', default=False, type=str2bool, help='nesterov')

    args = parser.parse_args()

    return args
 

def adjust_learning_rate(optimizer, epoch, MAX_EPOCHES, INIT_LR, power=0.9, warmup_epoch=20, lr_min=1e-8, _type='exp', use_warmup=False): 
    lr_max = INIT_LR
    
    if use_warmup:
        if _type == 'exp': 
            if epoch >= warmup_epoch:
                lr = round(INIT_LR * np.power(1 - (epoch) / MAX_EPOCHES ,power),8)      # round( ..., 8)
            else:
                lr = round(INIT_LR * np.power(1 - (epoch) / MAX_EPOCHES ,power),8) * epoch / warmup_epoch
        else:
            if epoch >= warmup_epoch:
                lr = lr_min + (lr_max - lr_min) * (1 + math.cos(math.pi * epoch / MAX_EPOCHES)) / 2       # Cosine Annealing
            else:
                lr = lr_min + (lr_max - lr_min) * (1 + math.cos(math.pi * epoch / MAX_EPOCHES)) / 2 * epoch / warmup_epoch
    else:
        if _type == 'exp':
            lr = round(INIT_LR * np.power(1 - (epoch) / MAX_EPOCHES ,power),8)
        else:
            lr = lr_min + (lr_max - lr_min) * (1 + math.cos(math.pi * epoch / MAX_EPOCHES)) / 2
    
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

def train(args, train_loader, model, optimizer, epoch, max_epoch=None, INIT_LR=None, _type=None, lr_min=None, use_warmup=False):
    lossesss = AverageMeter()
    ious = AverageMeter()
    dices = AverageMeter()
    WT_dice_coef = AverageMeter()
    TC_dice_coef = AverageMeter()
    ET_dice_coef = AverageMeter()
    sensitives = AverageMeter()
    accuracyes = AverageMeter()

    model.train()

    adjust_learning_rate(optimizer, epoch, MAX_EPOCHES=max_epoch, INIT_LR=INIT_LR, lr_min=lr_min, _type=_type, use_warmup=use_warmup)  # Or original: 'exp' or 'CosineAnnealing'
            
    for _, (s_inp, t_inp, label, ED_G, ET_Y, NET_R) in tqdm(enumerate(train_loader), total=len(train_loader)):
        s_inp = s_inp.to(device)
        t_inp = t_inp.to(device)
        label = label.to(device)
        ED_G, ET_Y, NET_R = ED_G.to(device), ET_Y.to(device), NET_R.to(device)

        # compute output
        s_f1_gap, t_f1_gap, s_f2_gap, t_f2_gap, s_rec1, t_rec1, s_rec2, t_rec2, pred_d1_dSup_fea_list, pred_d2_dSup_fea_list, s_all, t_all, pred_s, pred_t = model(S=s_inp, T=t_inp)  # torch.Size([32, 3, 160, 160], dtype=torch.float32)
        
        # Cal OT Loss
        Loss_ot = losses.OptimalTransportLoss()
        Loss_ot_f1 = Loss_ot(s_f1_gap, t_f1_gap)
        Loss_ot_f2 = Loss_ot(s_f2_gap, t_f2_gap)
        Loss_ot = Loss_ot_f1 + Loss_ot_f2
        
        # Cal Reconstructed Loss        
        loss_rec_s_rec1 = F.l1_loss(s_rec1, t_inp)
        loss_rec_t_rec1 = F.l1_loss(t_rec1, t_inp)
        loss_rec_s_rec2 = F.l1_loss(s_rec2, t_inp)
        loss_rec_t_rec2 = F.l1_loss(t_rec2, t_inp)
        loss_rec = loss_rec_s_rec1 + loss_rec_t_rec1 + loss_rec_s_rec2 + loss_rec_t_rec2
        
        # # Cal dSup Feature Loss
        Loss_dSup = losses.AffinityKDLoss()
        affinity_kd_fea_loss = Loss_dSup(pred_d1_dSup_fea_list, pred_d2_dSup_fea_list)  # Compute feature KD loss
        loss_dSup = affinity_kd_fea_loss
        
        # Cal Solo Segmentation Loss
        s_ed, s_et, s_net = s_all[:, 0, ...], s_all[:, 1, ...], s_all[:, 2, ...]
        t_ed, t_et, t_net = t_all[:, 0, ...], t_all[:, 1, ...], t_all[:, 2, ...]
        loss_ed_align = F.mse_loss(s_ed, t_ed)   
        loss_et_align = F.mse_loss(s_et, t_et)
        loss_net_align = F.mse_loss(s_net, t_net)
        loss_solo_align_all = (loss_ed_align + loss_et_align + loss_net_align) * 0.33
        Loss_solo_seg = losses.BCEDiceLoss()
        loss_ed_seg = Loss_solo_seg(s_ed, ED_G)
        loss_et_seg = Loss_solo_seg(s_et, ET_Y)
        loss_net_seg = Loss_solo_seg(s_net, NET_R)
        loss_solo_seg_all = (loss_ed_seg + loss_et_seg + loss_net_seg) * 0.33
        loss_solo_all = 2.0 * loss_solo_align_all + loss_solo_seg_all
        
        # Cal Logit Loss    
        # cosine_loss = 1 - F.cosine_similarity(pred_s, pred_t).mean()   
        # l1_loss = F.l1_loss(pred_s, pred_t)   
        # mse_loss = F.mse_loss(pred_s, pred_t)  
        # loss_logit = F.mse_loss(pred_s, pred_t)
        Loss_kl = losses.KLDivergenceLoss()
        loss_logit = Loss_kl(pred_s, pred_t)

        # Cal Seg Loss
        Loss_seg = losses.BCEDiceLoss()
        loss_seg = Loss_seg(pred_s, label)
        
        weight_coral = 2.0
        weight_dSup = 2.0
        weight_logit = 2.0
        weight_rec = 1.0
        weight_seg = 1.0
        
        loss = weight_coral * Loss_ot + weight_rec * loss_rec + weight_logit * loss_logit + weight_seg * loss_seg + loss_solo_all + weight_dSup * loss_dSup
        
        
        iou = iou_score(pred_s, label)
        dice = dice_coef(pred_s, label)

        b, _, h, w = pred_s.shape
        wt_pre = pred_s[:, 0, :, :]
        wt_label = label[:, 0, :, :]
        wt_dice = dice_coef(wt_pre.view((b, 1, h, w)), wt_label.view((b, 1, h, w)))
        tc_pre = pred_s[:, 1, :, :]
        tc_label = label[:, 1, :, :]
        tc_dice = dice_coef(tc_pre.view((b, 1, h, w)), tc_label.view((b, 1, h, w)))
        et_pre = pred_s[:, 2, :, :]
        et_label = label[:, 2, :, :]
        et_dice = dice_coef(et_pre.view((b, 1, h, w)), et_label.view((b, 1, h, w)))

        sensitive = sensitivity_score(pred_s, label)
        accuracy = accuracy_score(pred_s, label)

        lossesss.update(loss.item(), t_inp.size(0))
        ious.update(iou, t_inp.size(0))
        dices.update(dice, t_inp.size(0))
        WT_dice_coef.update(wt_dice, t_inp.size(0))
        TC_dice_coef.update(tc_dice, t_inp.size(0))
        ET_dice_coef.update(et_dice, t_inp.size(0))
        # wt_hausdorff_distances.update(wt_hd, t_inp.size(0))
        # tc_hausdorff_distances.update(tc_hd, t_inp.size(0))
        # et_hausdorff_distances.update(et_hd, t_inp.size(0))
        sensitives.update(sensitive, t_inp.size(0))
        accuracyes.update(accuracy, t_inp.size(0))

        # compute gradient and do optimizing step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    log = OrderedDict([
        ('loss', lossesss.avg),
        ('iou', ious.avg),
        ('dice', dices.avg),
        ('wt_dice', WT_dice_coef.avg),
        ('tc_dice', TC_dice_coef.avg),
        ('et_dice', ET_dice_coef.avg),
        # ('wt_hd', wt_hausdorff_distances.avg),
        # ('tc_hd', tc_hausdorff_distances.avg),
        # ('et_hd', et_hausdorff_distances.avg),
        ('sensitive', sensitives.avg),
        ('accuracy', accuracyes.avg),
    ])

    return log

def validate(args, val_loader, model):
    lossesss = AverageMeter()
    ious = AverageMeter()
    dices = AverageMeter()
    WT_dice_coef = AverageMeter()
    TC_dice_coef = AverageMeter()
    ET_dice_coef = AverageMeter()
    # wt_hausdorff_distances = AverageMeter()
    # tc_hausdorff_distances = AverageMeter()
    # et_hausdorff_distances = AverageMeter()
    sensitives = AverageMeter()
    accuracyes = AverageMeter()

    # switch to evaluate mode
    model.eval()

    with torch.no_grad():
        for _, (_, t_inp, label, _, _, _) in tqdm(enumerate(val_loader), total=len(val_loader)):
            t_inp = t_inp.to(device)
            label = label.to(device)

            # compute output
            output = model(T=t_inp, is_train=False)  # torch.Size([32, 3, 160, 160], dtype=torch.float32)
            b, c, h, w = output.shape
            Loss_seg = losses.BCEDiceLoss()
            
            loss = Loss_seg(output, label)
            iou = iou_score(output, label)
            dice = dice_coef(output, label)

            wt_pre = output[:, 0, :, :]
            wt_label = label[:, 0, :, :]
            wt_dice = dice_coef(wt_pre.view((b, 1, h, w)), wt_label.view((b, 1, h, w)))
            tc_pre = output[:, 1, :, :]
            tc_label = label[:, 1, :, :]
            tc_dice = dice_coef(tc_pre.view((b, 1, h, w)), tc_label.view((b, 1, h, w)))
            et_pre = output[:, 2, :, :]
            et_label = label[:, 2, :, :]
            et_dice = dice_coef(et_pre.view((b, 1, h, w)), et_label.view((b, 1, h, w)))

            # hd = hausdorff_dist()
            # wt_hd = hd.compute(wt_pre, wt_label)
            # tc_hd = hd.compute(tc_pre, tc_label)
            # et_hd = hd.compute(et_pre, et_label)

            sensitive = sensitivity_score(output, label)
            accuracy = accuracy_score(output, label)

            lossesss.update(loss.item(), t_inp.size(0))
            ious.update(iou, t_inp.size(0))
            dices.update(dice, t_inp.size(0))
            WT_dice_coef.update(wt_dice, t_inp.size(0))
            TC_dice_coef.update(tc_dice, t_inp.size(0))
            ET_dice_coef.update(et_dice, t_inp.size(0))
            # wt_hausdorff_distances.update(wt_hd, t_inp.size(0))
            # tc_hausdorff_distances.update(tc_hd, t_inp.size(0))
            # et_hausdorff_distances.update(et_hd, t_inp.size(0))
            sensitives.update(sensitive, t_inp.size(0))
            accuracyes.update(accuracy, t_inp.size(0))

    log = OrderedDict([
        ('loss', lossesss.avg),
        ('iou', ious.avg),
        ('dice', dices.avg),
        ('wt_dice', WT_dice_coef.avg),
        ('tc_dice', TC_dice_coef.avg),
        ('et_dice', ET_dice_coef.avg),
        # ('wt_hd', wt_hausdorff_distances.avg),
        # ('tc_hd', tc_hausdorff_distances.avg),
        # ('et_hd', et_hausdorff_distances.avg),
        ('sensitive', sensitives.avg),
        ('accuracy', accuracyes.avg),
    ])

    return log


def main():
    args = parse_args()
    #args.dataset = "datasets"

    if args.name is None:
        if args.deepsupervision:
            args.name = '%s_%s_wDS' %(args.dataset, args.arch)
        else:
            # args.name = 'BraTS1819_zhang_UNet_woDS'
            args.name = '%s_%s_woDS' %(args.dataset, args.arch)

    if not os.path.exists('/data/zh/BrainTumorSegmentation/models/%s' %args.name):
        os.makedirs('/data/zh/BrainTumorSegmentation/models/%s' %args.name)
    
    print('Config ----------------------------------------')
    for arg in vars(args):
        print('%s: %s' %(arg, getattr(args, arg)))
    print('-----------------------------------------------')

    with open('/data/zh/BrainTumorSegmentation/models/%s/args.txt' %args.name, 'w') as f:
        for arg in vars(args):
            print('%s: %s' %(arg, getattr(args, arg)), file=f)

    joblib.dump(args, '/data/zh/BrainTumorSegmentation/models/%s/args.pkl' %args.name)

    # # define loss function (criterion)
    # if args.loss == 'BCEWithLogitsLoss':
    #     criterion = nn.BCEWithLogitsLoss().to(device)
    # else:
    #     # criterion = BCEDiceLoss()
    #     criterion = losses.__dict__[args.loss]().to(device)

    cudnn.benchmark = True

    # Data loading code
    img_paths = IMG_PATH
    mask_paths = MASK_PATH

    train_img_paths, val_img_paths, train_mask_paths, val_mask_paths = train_test_split(img_paths, mask_paths, test_size=0.15, random_state=41)
    print(" = = = > train_num : %s"%str(len(train_img_paths)))
    print(" = = = > val_num : %s"%str(len(val_img_paths)))

    # create model
    print(" = = = > creating model : %s" %args.arch)
    # model = UNet()
    model = MABTS.__dict__[args.arch](args)

    if load_weight:
        pretrained_dict = torch.load(model_pre_path, map_location='cpu')
        model_dict = model.state_dict() 
        pretrained_dict =  {k: v for k, v in pretrained_dict.items() if k in model_dict}  
        model_dict.update(pretrained_dict) 
        model.load_state_dict(model_dict)

    model = model.to(device)
    print(" = = = > model total params : %s" %str(count_params(model))) 

    if args.optimizer == 'Adam':
        optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr)
    elif args.optimizer == 'SGD':
        optimizer = optim.SGD(filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr,
            momentum=args.momentum, weight_decay=args.weight_decay, nesterov=args.nesterov)

    train_dataset = Dataset(args, train_img_paths, train_mask_paths, args.aug)
    val_dataset = Dataset(args, val_img_paths, val_mask_paths)

    # DataLoader
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, pin_memory=True, drop_last=True)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, pin_memory=True, drop_last=False)

    log = pd.DataFrame(index=[], columns=[
        'epoch', 'lr', 
        'train_loss', 'train_iou', 'train_dice', 'train_WT_dice', 'train_TC_dice', 'train_ET_dice', 'train_sensitive', 'train_accuracy', 
        'val_loss', 'val_iou', 'val_dice', 'val_WT_dice', 'val_TC_dice', 'val_ET_dice', 'val_sensitive', 'val_accuracy',
    ])

    max_epoch = args.epochs
    best_iou = 0   
    trigger = 0    
    start = time()
    
    for epoch in range(1, args.epochs+1):
        torch.cuda.empty_cache()
        
        print(' = = = > Epoch [%d/%d]' %(epoch, args.epochs))
                                                                
        # train for one epoch
        train_log = train(args, train_loader, model, optimizer, epoch, max_epoch=max_epoch, INIT_LR=args.lr*args.batch_size, lr_min=1e-8, _type='cos', use_warmup=True)
        # evaluate on validation set
        val_log = validate(args, val_loader, model)

        print('Train_Loss %.4f\tTrain_IoU %.4f\tTrain_Dice %.4f\tTrain_WT_Dice %.4f\tTrain_TC_Dice %.4f\tTrain_ET_Dice %.4f\tTrain_Sensitive %.4f\tTrain_Acc %.4f'
            %(train_log['loss'], train_log['iou'], train_log['dice'], train_log['wt_dice'], train_log['tc_dice'], train_log['et_dice'], train_log['sensitive'], train_log['accuracy']))
        print('Val_Loss %.4f\t\tVal_IoU %.4f\t\tVal_Dice %.4f\t\tVal_WT_Dice %.4f\tVal_TC_Dice %.4f\tVal_ET_Dice %.4f\tVal_Sensitive %.4f\tVal_Acc %.4f'
            %(val_log['loss'], val_log['iou'], val_log['dice'], val_log['wt_dice'], val_log['tc_dice'], val_log['et_dice'], val_log['sensitive'], val_log['accuracy']))

        # writer.add_scalar('Train_Loss', scalar_value=train_log['loss'], global_step=epoch)
        # writer.add_scalar('Train_IoU', scalar_value=train_log['iou'], global_step=epoch)
        # writer.add_scalar('Train_Dice', scalar_value=train_log['dice'], global_step=epoch)
        # writer.add_scalar('Train_WT_Dice', scalar_value=train_log['wt_dice'], global_step=epoch)
        # writer.add_scalar('Train_TC_Dice', scalar_value=train_log['tc_dice'], global_step=epoch)
        # writer.add_scalar('Train_ET_Dice', scalar_value=train_log['et_dice'], global_step=epoch)
        # writer.add_scalar('Train_Sensitive', scalar_value=train_log['sensitive'], global_step=epoch)
        # writer.add_scalar('Train_Acc', scalar_value=train_log['accuracy'], global_step=epoch)
        
        # writer.add_scalar('Val_Loss', scalar_value=val_log['loss'], global_step=epoch)
        # writer.add_scalar('Val_IoU', scalar_value=val_log['iou'], global_step=epoch)
        # writer.add_scalar('Val_Dice', scalar_value=val_log['dice'], global_step=epoch)
        # writer.add_scalar('Val_WT_Dice', scalar_value=val_log['wt_dice'], global_step=epoch)
        # writer.add_scalar('Val_TC_Dice', scalar_value=val_log['tc_dice'], global_step=epoch)
        # writer.add_scalar('Val_ET_Dice', scalar_value=val_log['et_dice'], global_step=epoch)
        # writer.add_scalar('Val_Sensitive', scalar_value=val_log['sensitive'], global_step=epoch)
        # writer.add_scalar('Val_Acc', scalar_value=val_log['accuracy'], global_step=epoch)

        tmp = pd.Series([
            epoch, args.lr,
            train_log['loss'], train_log['iou'], train_log['dice'], train_log['wt_dice'], train_log['tc_dice'], train_log['et_dice'], train_log['sensitive'], train_log['accuracy'], 
            val_log['loss'], val_log['iou'], val_log['dice'], val_log['wt_dice'], val_log['tc_dice'], val_log['et_dice'], val_log['sensitive'], val_log['accuracy'], 
        ], index=[
            'epoch', 'lr', 
            'train_loss', 'train_iou', 'train_dice', 'train_WT_dice', 'train_TC_dice', 'train_ET_dice', 'train_sensitive', 'train_accuracy', 
            'val_loss', 'val_iou', 'val_dice', 'val_WT_dice', 'val_TC_dice', 'val_ET_dice', 'val_sensitive', 'val_accuracy',
        ])

        log = log.append(tmp, ignore_index=True)
        log.to_csv('/data/zh/BrainTumorSegmentation/models/%s/log.csv' %args.name, index=False)

        trigger += 1

        if val_log['iou'] > best_iou:
            torch.save(model.state_dict(), '/data/zh/BrainTumorSegmentation/models/%s/model.pth' %args.name)
            best_iou = val_log['iou']
            print(" = = = > saved best model")
            trigger = 0

        if not args.early_stop is None:
            if trigger >= args.early_stop:
                print(" = = = > early stopping")
                break
        
    end = time()
    total_time = end - start
    log_time = pd.DataFrame(index=[], columns=['Total Time/s',])
    tmp_time = pd.Series([total_time], index=['Total Time/s'])
    log_time = log_time.append(tmp_time, ignore_index=True)
    log_time.to_csv('/data/zh/BrainTumorSegmentation/models/%s/log_total_time.csv' %args.name, index=False)
    
if __name__ == '__main__':
    main()

