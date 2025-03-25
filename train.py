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

def weights_init(init_type='xavier'):
    def init_fun(m):
        classname = m.__class__.__name__
        if (classname.find('Conv') == 0 or classname.find('Linear') == 0) and hasattr(m, 'weight'):
            if init_type == 'normal':
                init.normal_(m.weight.data, 0.0, 0.02)
            elif init_type == 'xavier':
                init.xavier_normal_(m.weight.data, gain=math.sqrt(2))
            elif init_type == 'kaiming':
                init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
            elif init_type == 'orthogonal':
                init.orthogonal(m.weight.data, gain=math.sqrt(2))
            elif init_type == 'default':
                pass
            else:
                assert 0, "Unsupported initialization: {}".format(init_type)
            if hasattr(m, 'bias') and m.bias is not None:
                init.constant_(m.bias.data, 0.0)
        elif (classname.find('Norm') == 0):
            if hasattr(m, 'weight') and m.weight is not None:
                init.constant_(m.weight.data, 1.0)
            if hasattr(m, 'bias') and m.bias is not None:
                init.constant_(m.bias.data, 0.0)
    return init_fun

batch_size = 5
epochs = 200
learning_rate = 0.001

train_transform = preprocess.make_transform('train')
test_transform = preprocess.make_transform('test')

train = preprocess.Dataset(data_path, 'train', transform = train_transform)
val = preprocess.Dataset(data_path, 'val', transform = test_transform)

train_loader = DataLoader(train, batch_size = batch_size, shuffle = True)
val_loader = DataLoader(val, bathc_size = batch_size, shuffle = False)

device = 'cuda:1'
model = models.UNet(1,1).to(device)

optimizer = optim.Adam(model.parameters(), lr = learning_rate, weight_decay=1e-8)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer,'min' if model.n_classes > 1 else 'max', patience=30, factor= 0.1)
criterion = nn.BCEWithLogitsLoss()

init_weights = preprocess.weights_init('kaming')
model.apply(init_weights)

def train_model(model, train_loader, epochs, device, optimizer, scheduler, criterion, model_path, val_loader = None)
    score_dict = {}
    for epoch in range(epochs):
        model.train()
        train_loss = 0

        for i, (images, masks) in enumerate(train_loader):
            imgs = images
            true_masks = masks

            imgs = imgs.to(device = device, dtype = torch.float32)
            mask_type = torch.float32 if model.n_classes == 1 else torch.long
            true_masks = true_masks.to(device = device, dtype = mask_type)

            masks_pred = model(imgs)
            loss = criterion(masks_pred, true_masks)
            train_loss += loss.item()

            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_value_(model.parameters(), 0.1)
            optimizer.step()

            if val_loader != None:
                model.eval()
                val_loss = 0
                for j, (images, masks) in enumerate(val_loader):

                    imgs = images
                    true_masks = masks

                    imgs = imgs.to(device = device, dtype = torch.float32)
                    mask_type = torch.float32 if model.n_classes == 1 else torch.long
                    true_masks = true_masks.to(device = device, dtype = mask_type)

                    masks_pred = model(imgs)
                    loss = criterion(masks_pred, true_masks)
                    val_loss += loss.item()
                
            else: 
                val_loss = 0
                j = 0
        schedule_standard = train_loss/(i+1)
        scheduler.step(schedule_standard)
        print("epoch:{}/{} | trn loss:{:.4f} | val loss: {:.4f}".format(
            epoch + 1, epochs, train_loss /(i+1), val_loss /(j+1)))
        score_dict[epoch] = {'train':train_loss/(i+1), 'val':val_loss/(j+1)}

        checkpoint = {'loss': train_loss/(i+1),
                      'state_dict': model.state_dict(),
                      'optimizer':optimizer.state_dict()}
        torch.save(checkpoint, model_path+'{}_epoch.pth'.format(epoch))
    
    return model, score_dict

def eval_model(model, loader, device):
    model.eval()
    with torch.no_grad():
        mask_type = torch.float32 if model.n_classes == 1 else torch.long
        total_loss = 0
        iou_score = 0
        preds = []
        preds_thres = []
        labels = []

        for j, (images, masks) in enumerate(loader):
            imgs, true_masks = images, masks
            imgs = imgs.to(device = device, dtype = torch.float32)
            true_masks = true_masks.to(device = device, dtype = mask_type)
            masks_pred = model(imgs)

            if model.n_classes > 1:
                tot += F.cross_entropy(masks_pred, true_masks).item()
            else:
                criteriion = nn.BCEWithLogitsLoss()
                pred = masks_pred
                loss = criterion(masks_pred, true_masks)

                pred2 = torch.sigmoid(masks_pred)
                pred2 = (pred>0.5).float()

                pred = pred.cpu().numpy()
                pred2 = pred2.cpu().numpy()
                true_masks = true_masks.cpu().numpy()

                preds.append(pred)
                preds_thres.append(pred2)
                labels.append(true_masks)
                total_loss += loss.item()

                iou_score += compute_iou(pred2, true_masks)
    return np.vstach(preds), np.vstack(preds_thres), np.vstack(labels), (total_loss/(j+1), iou_score/(j+1))
