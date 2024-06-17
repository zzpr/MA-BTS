import numpy as np
import cv2 #https://www.jianshu.com/p/f2e88197e81d
import random

from skimage.io import imread
from skimage import color

import torch
import torch.utils.data
from torchvision import datasets, models, transforms


class Dataset(torch.utils.data.Dataset):
    def __init__(self, args, img_paths, mask_paths, aug=False):
        self.args = args
        self.img_paths = img_paths
        self.mask_paths = mask_paths
        self.aug = aug

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, idx):
        img_path = self.img_paths[idx]
        mask_path = self.mask_paths[idx]

        npimage = np.load(img_path)     # (160, 160, 1)
        npmask = np.load(mask_path)     # (160, 160)

        npimage = npimage.transpose((2, 0, 1))  # (160,160,4) -> (1, 160, 160)
        
        # whole tumor (WT), tumor core (TC) and enhancing tumor (ET)
        # WT = NET(1) + ED(2) + ET(4) 
        WT_Label = npmask.copy()
        WT_Label[npmask == 1] = 1.
        WT_Label[npmask == 2] = 1.
        WT_Label[npmask == 4] = 1.
        # TC = NET(1) + ET(4)
        TC_Label = npmask.copy()
        TC_Label[npmask == 1] = 1.
        TC_Label[npmask == 2] = 0.
        TC_Label[npmask == 4] = 1.
        # ET(4)
        ET_Label = npmask.copy()
        ET_Label[npmask == 1] = 0.
        ET_Label[npmask == 2] = 0.
        ET_Label[npmask == 4] = 1.

        nplabel = np.empty((160, 160, 3))
        # nplabel = np.empty((192, 192, 3))
        nplabel[:, :, 0] = WT_Label
        nplabel[:, :, 1] = TC_Label
        nplabel[:, :, 2] = ET_Label
        
        nplabel = nplabel.transpose((2, 0, 1))  # (160,160,3) -> (3, 160, 160)
        
        s_img = npimage.copy()
        t_img = npimage.copy()
        
        # flair
        s_img[1:] = 0
        # t_img[1:] = 0
        
        # t1
        t_img[:1] = 0
        t_img[2:] = 0
        
        # t1ce
        # t_img[:2] = 0
        # t_img[3:] = 0
        
        # # t2
        # t_img[:3] = 0
        
        '''          SOLO Label           '''
        NET_R = np.zeros((1, npmask.shape[0], npmask.shape[1]), dtype=np.float32)
        ED_G = np.zeros_like(NET_R, dtype=np.float32)
        ET_Y = np.zeros_like(NET_R, dtype=np.float32)

        # Store the values 1, 2, and 4 from the ground truth into the respective tensors
        NET_R[0, npmask == 1] = 1
        ED_G[0, npmask == 2] = 2
        ET_Y[0, npmask == 4] = 4

        
        nplabel = nplabel.astype("float32")
        npimage = npimage.astype("float32")
        s_img = s_img.astype("float32")
        t_img = t_img.astype("float32")
        
        # # Flair     
        # has_positive_values = np.any(s_img[:1] > 0)
        # print(has_positive_values)
        
        # has_positive_values = np.any(s_img[1:] > 0)
        # print(has_positive_values)
        
        # # Other   
        # has_positive_values = np.any(t_img[:1] > 0)
        # print(has_positive_values)
        # has_positive_values = np.any(t_img[1] > 0)
        # print(has_positive_values)
        
        # has_positive_values = np.any(t_img[2:] > 0)
        # print(has_positive_values)
        
        return s_img, t_img, nplabel, ED_G, ET_Y, NET_R
    
    