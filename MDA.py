import numpy as np
import torch
import os
from collections import OrderedDict
from torch.autograd import Variable
import itertools
# import util.util as util
import ImagePool
# from .base_model import BaseModel
import networks
import torch.nn.functional as F
import torch.nn as nn
from torch.optim import Adam
# import sys
# import skimage


def bce_dice_loss(input, target):
    smooth = 1e-5
    input = torch.sigmoid(input)
    num = target.size(0)
    input = input.view(num, -1)
    target = target.view(num, -1)
    intersection = (input * target)
    dice = (2. * intersection.sum(1) + smooth) / (input.sum(1) + target.sum(1) + smooth)
    dice_loss = 1 - dice.sum() / num
    bce_loss = F.binary_cross_entropy(input, target)
    return bce_loss + dice_loss

ngf = 64
ndf = 64

class MDA(nn.Module):

    def __init__(self, args):
        super(MDA, self).__init__(args)

        nb = args.batch_size
        size = 160
        
        self.Tensor = torch.cuda.FloatTensor if self.gpu_ids else torch.Tensor
        self.isTrain = args.isTrain
        
        self.input_A = self.Tensor(nb, args.input_channels, size, size)
        self.input_B = self.Tensor(nb, args.input_channels, size, size)
        self.input_Seg = self.Tensor(nb, args.input_channels, size, size)
        '============================================================================================================'
        
        if args.seg_norm == 'CrossEntropy':
            self.input_Seg_one = self.Tensor(nb, args.output_nc, size, size)

        # load/define networks
        # The naming conversion is different from those used in the paper
        # Code (paper): G_A (G), G_B (F), D_A (D_Y), D_B (D_X)

        #         netG_A : S -> T     netG_B : T -> S
        self.netG_A = networks.define_G(args.input_channels, args.input_channels,
                                        ngf, 'resnet_6blocks', 'instance', False, args.gpu_ids)

        #         D_A判断fake_B和real_B       D_B判断fake_A和real_A
        if self.isTrain:
            self.netD_A = networks.define_D(args.input_channels, ndf, 'basic', 3, 'instance', False, args.gpu_ids)
               
        self.netG_seg = networks.define_G(args.input_channels, args.seg_classif, ngf, 'resnet_6blocks', 'batch', False, args.gpu_ids)

        if not self.isTrain or args.continue_train:
            which_epoch = args.which_epoch
            self.load_network(self.netG_A, 'G_A', which_epoch)
            if self.isTrain:
                self.load_network(self.netD_A, 'D_A', which_epoch)
                self.load_network(self.netG_seg, 'Seg_A', which_epoch)   

        if self.isTrain:
            self.old_lr = args.lr
            
            self.fake_A_pool = ImagePool(50)
            self.fake_B_pool = ImagePool(50)
            
            # define loss functions
            self.criterionGAN = networks.GANLoss(use_lsgan=True)
            # self.criterionGAN = networks.GANLoss(use_lsgan=not args.no_lsgan, tensor=self.Tensor)
            self.criterionCycle = torch.nn.L1Loss() 
            self.criterionIdt = torch.nn.L1Loss()   
            
            # initialize optimizers         
            self.optimizer_G = Adam(itertools.chain(self.netG_A.parameters(), self.netG_seg.parameters()), lr=args.lr, betas=(0.5, 0.999))
            self.optimizer_D_A = Adam(self.netD_A.parameters(), lr=args.lr, betas=(0.5, 0.999))
            
        print('---------- Networks initialized -------------')
        networks.print_network(self.netG_A)
        
        if self.isTrain:
            networks.print_network(self.netD_A)
            networks.print_network(self.netG_seg)
        print('-----------------------------------------------')

    def forward(self, input):
        xs, xt, label = input
        self.input_A.resize_(xs.size()).copy_(xs)
        self.input_B.resize_(xt.size()).copy_(xt)
        self.input_Seg.resize_(label.size()).copy_(label)
        
        self.real_A = self.input_A
        self.real_B = self.input_B
        self.real_Seg = self.input_Seg

    def backward_D_basic(self, netD, real, fake):
        # Real  
        pred_real = netD.forward(real)
        loss_D_real = self.criterionGAN(pred_real, True, self.gpu_ids)    
        # Fake
        pred_fake = netD.forward(fake.detach())    
        loss_D_fake = self.criterionGAN(pred_fake, False, self.gpu_ids)
        # Combined loss
        loss_D = (loss_D_real + loss_D_fake) * 0.5
        # backward
        loss_D.backward()
        return loss_D

    def backward_D_A(self):
        fake_B = self.fake_B_pool.query(self.fake_B)   
        
        self.loss_D_A = self.backward_D_basic(self.netD_A, self.real_B, fake_B)

    def backward_G(self):
        lambda_A = 10

        # GAN loss
        # D_A(G_A(A))
        self.fake_B = self.netG_A.forward(self.real_A)  #       (b, 1, 160, 160)
        pred_fake = self.netD_A.forward(self.fake_B)    #     (b, 1, 23, 23)
        self.loss_G_A = self.criterionGAN(pred_fake, True, self.gpu_ids)           
        
        # # Forward cycle loss
        # self.rec_A = self.netG_B.forward(self.fake_B)   
        # self.loss_cycle_A = self.criterionCycle(self.rec_A, self.real_A) * lambda_A
        
        "---------------------------------------------      Segmentation    --------------------------------------------------------------------"
        # Segmentation loss         seg_norm = 'CrossEntropy'
        self.seg_fake_B = self.netG_seg.forward(self.fake_B)    #   (b, 3, 160, 160)
        self.loss_seg = bce_dice_loss(self.seg_fake_B, self.real_Seg)
        "---------------------------------------------------------------------------------------------------------------------------------------"

        # combined loss
        self.loss_G = self.loss_G_A + self.loss_seg
        # self.loss_G = self.loss_G_A + self.loss_cycle_A + self.loss_seg
        self.loss_G.backward()

    def optimize_parameters(self):
        # # forward
        # self.forward()
        # G_A and G_B
        self.optimizer_G.zero_grad()
        self.backward_G()
        self.optimizer_G.step()
        # D_A
        self.optimizer_D_A.zero_grad()
        self.backward_D_A()
        self.optimizer_D_A.step()
        
        return {'G_loss':self.loss_G_A, 'D_loss':self.loss_D_A, 'Seg_loss':self.loss_seg}

    # def get_current_errors(self):
    #     D_A = self.loss_D_A.item()        
    #     G_A = self.loss_G_A.item()
    #     Cyc_A = self.loss_cycle_A.item()

    #     Seg_B = self.loss_seg.item()
        
    #     if self.args.identity > 0.0:
    #         idt_A = self.loss_idt_A.item()
    #         idt_B = self.loss_idt_B.item()
    #         return OrderedDict([('D_A', D_A), ('G_A', G_A), ('Cyc_A', Cyc_A), ('idt_A', idt_A), ('idt_B', idt_B)])
    #     else:
    #         return OrderedDict([('D_A', D_A), ('G_A', G_A), ('Cyc_A', Cyc_A), ('Seg', Seg_B)])

    # def get_current_visuals(self):
    #     real_A = util.tensor2im(self.real_A.data)
    #     fake_B = util.tensor2im(self.fake_B.data)
        
    #     seg_B = util.tensor2seg(torch.max(self.seg_fake_B.data,dim=1,keepdim=True)[1])
    #     manual_B = util.tensor2seg(torch.max(self.real_Seg.data,dim=1,keepdim=True)[1])
        
    #     rec_A = util.tensor2im(self.rec_A.data)
    #     real_B = util.tensor2im(self.real_B.data)
    #     fake_A = util.tensor2im(self.fake_A.data)
    #     rec_B = util.tensor2im(self.rec_B.data)
    #     if self.args.identity > 0.0:
    #         idt_A = util.tensor2im(self.idt_A.data)
    #         idt_B = util.tensor2im(self.idt_B.data)
    #         return OrderedDict([('real_A', real_A), ('fake_B', fake_B), ('rec_A', rec_A), ('idt_B', idt_B),
    #                             ('real_B', real_B), ('fake_A', fake_A), ('rec_B', rec_B), ('idt_A', idt_A)])
    #     else:
    #         return OrderedDict([('real_A', real_A), ('fake_B', fake_B), ('rec_A', rec_A), ('seg_B',seg_B), ('manual_B',manual_B),
    #                             ('real_B', real_B), ('fake_A', fake_A), ('rec_B', rec_B)])

    # def save(self, label):
        self.save_network(self.netG_A, 'G_A', label, self.gpu_ids)
        self.save_network(self.netD_A, 'D_A', label, self.gpu_ids)
        self.save_network(self.netG_seg, 'Seg_A', label, self.gpu_ids)

    def update_learning_rate(self):
        lrd = self.args.lr / self.args.niter_decay
        lr = self.old_lr - lrd
        for param_group in self.optimizer_D_A.param_groups:
            param_group['lr'] = lr
        for param_group in self.optimizer_G.param_groups:
            param_group['lr'] = lr

        print('update learning rate: %f -> %f' % (self.old_lr, lr))
        self.old_lr = lr
