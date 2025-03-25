import pandas as pd
import numpy as np
import os

import matplotlib.pyplot as plt
import random
import math
from sklearn.utils import shuffle
from sklearn.metrics import confusion_matrix

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
#Dataset 함수 내에서 data_path 파일 읽을 때, 필요한 함수로 직접 정의해줘야함. 
#import utils
#import preprocess
#import models
#import evaluation as ev
from torch.nn import init
from torch.utils.data import DataLoader, Dataset
from torch.autograd import Variable
from albumentations.augmentations import transforms
from albumentations.pytorch import ToTensor, ToTensorV2
from albumentations.core.composition import Compose, OneOf

from PIL import Image

class Dataset(Dataset):

    def __init__(self, data_path, mode = None, transform = None):
        self.data_path = data_path
        self.mode = mode
        self.transform = transform

        self.inputs = utils.read_tiff(self.data_path + 'train-input.tif')
        self.labels = utils.read_tiff(self.data_path + 'train-labels.tif')

        idx = utils.set_index(len(self.inputs), 0.7, self.mode)
        self.inputs = self.inputs[idx]
        self.labels = self.labels[idx]

    def __len__(self):
        return len(self.inputs)
    
    def __getitem__(self, index):
        inputs = self.inputs[index]
        labels = self.labels[index]

        inputs = inputs/255
        labels = labels/255

        if labels.ndim == 2:
            labels = np.expand_dims(labels, -1)
        if inputs.ndim == 2:
            inputs = np.expand_dims(inputs, -1)
        
        if self.transform is not None:
            augmented = self.transform(image = inputs, mask = labels)
            inputs, labels = augmented['image'], augmented['mask']
            inputs = torch.from_numpy(inputs).permute(2,0,1)
            labels = torch.from_numpy(labels).permute(2,0,1)

        return inputs, labels

def make_transform(mode):
    if mode == 'train':
        train_transform = Compose([
            transforms.Resize(height = 512, width = 512),
            OneOf([transforms.MotionBlur(),
                   transforms.OpticalDistortion(),
                   transforms.GaussNoise(p=0.5),
                   transforms.RandomContrast()]),
            transforms.ElasticTransform(),
            OneOf([transforms.HorizontalFlip(),
                   transforms.RandomRotate90(),
                   transforms.VerticalFlip()]),
            transforms.Normalize(0.5),(0.5)
        ])
        return train_transform
    else:
        test_transform = Compose([
            transforms.Resize(height = 512, width = 512),
            transforms.Normalize((0.5),(0.5))
        ])
        return test_transform

class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels, mid_channels = None):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.double_conve = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size = 3, padding = 1),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
    def forward(self, x):
        return self.double_conv(x)

class Down(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, out_channels)
        )
    def forward(self, x):
        return self.maxpool_conv(x)

class Up(nn.Module):
    def __init__(self, in_channels, out_channels, bilinear = False):
        super().__init__()

        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode ='bilinear', align_corners=True)
            self.conv = DoubleConv(in_channels, out_channels, in_channels //2)
        else:
            self.up = nn.ConvTranspose2d(in_channels, in_channels //2, kernel_size=2, stride = 2)
            self.conv = DoubleConv(in_channels, out_channels)
    def forward(self, x1, x2):
        x1 = self.up(x1)
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]

        x1 = F.pad(x1, [diffX // 2, diffX - diffX //2,
                        diffY // 2, diffY - diffY //2])
        x = torch.cat([x2, x1], dim = 1)
        return self.conv(x)

class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        self.conv = nn.Conv2d(in_channels,out_channels, kernel_size=1)

    def forward(self, x):
        return self.conv(x)

class UNet(nn.Module):
    def __init__(self, n_channels, n_classes, bilinear = True):
        super(UNet, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear

        self.inc = DoubleConv(n_channels, 64)
        self.down1 = Down(64, 128)
        self.down2 = Down(128, 256)
        self.down3 = Down(256,512)
        
        factor = 2 if bilinear else 1
        self.down4 = Down(512, 1024 //factor)
        self.up1= Up(1024, 512 //factor, bilinear)
        self.up2 = Up(512, 256 //factor, bilinear)
        self.up3 = Up(256, 128 //factor, bilinear)
        self.up4 = Up(128, 64, bilinear)
        self.outc = OutConv(64, n_classes)



    def forward(self, x):
        x1= self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x = self.up1(x5, x4)
        x = self.up2(x,x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        logits = self.outc(x)
        return logits
        