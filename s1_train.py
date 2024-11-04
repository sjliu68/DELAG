# -*- coding: utf-8 -*-
"""
Created on Wed Feb 21 10:24:57 2024

@author: skrisliu


nyc5
360 317    ---- 114120

"""

import numpy as np
import torch
from torch import nn, optim

import glob
import pandas as pd
import matplotlib.pyplot as plt
import time

import os
import argparse


parser = argparse.ArgumentParser(description='manual to this script')
parser.add_argument('--n1', type=int, default = 0)                 # adjust here to split image
parser.add_argument('--n2', type=int, default = 114120)  # 114120  # adjust here to split image
args = parser.parse_args()


site = 'nyc'   # change the site name here
year = '2023'  # change the year here


'''
Need the following files to run the train script
-- cloudmask, multiple files, named structured as  "nyc/cloudmask/y2023/cloud20230128.npy"
-- train data, multiple files, named structured as  "nyc/train/y2023/cloud20230128.npy"
-- coarse LST, single file, named structured as  "nyc/nyc2023lst.npy"
'''


#%% load data
mask = glob.glob(site+'/cloudmask/y'+year + '/*.npy')
mask = np.load(mask[0])
mask = np.ones(mask.shape)
mask = mask==1


## train
fps = glob.glob(site + '/train/y'+year+'/train*.npy')   # change here

train = []
for each in fps:
    train_ = np.load(each)
    train.append(train_)
    
train = np.concatenate(train)
train = np.float64(train)

train = train[train[:,0]>15000]   # remove values that are too small, roughly 200K (-72C)


#%% indexing, this multi-indexing is necessary for data with irregular shape (including NaN data)
idx = train[:,1]*mask.shape[1] + train[:,2]  # spatial index, each pixel has one
idx2 = np.unique(idx)  # spatial index, each pixel has one
idx3 = np.arange(idx2.shape[0]) # idx2[idx3[0]]
np.save(site+'/'+site+year+'-idx2.npy',idx2)


#%% add the coarse temperature as linear term
lintemp = np.load(site+'/'+site+year+'lst.npy') / 10  # change here
newtemp = np.zeros(train.shape[0])
for i in range(0,365):
    newtemp[train[:,3]==i+1] = lintemp[i]

newtemp = newtemp.reshape(-1,1)    
train = np.concatenate([train,idx.reshape(-1,1),newtemp],axis=-1)


### 
lintemp = lintemp
lintemp = lintemp.reshape(-1,1)


#%% normalize, original scale factor is *0.00341802 + 149.0 (K), -273.15 (degree C)
train[:,0] = (train[:,0] - 25000) / 10000  # LST
train[:,1] = train[:,1] / 1000 # xy index 
train[:,2] = train[:,2] / 1000 # xy index
train[:,3] = train[:,3] / 100  # doy 

train[:,5] = train[:,5]  # coarse LST

train[:,4] = idx  # index






#%% model
class CosNet(torch.nn.Module):
    def __init__(self):
        super(CosNet, self).__init__()
        self.w1 = torch.nn.Parameter(torch.randn(1))
        self.w1.requires_grad = True
        self.w2 = torch.nn.Parameter(torch.randn(1))
        self.w2.requires_grad = True
        self.w4 = torch.nn.Parameter(torch.randn(1))
        self.w4.requires_grad = True
        self.c1 = torch.nn.Parameter(torch.randn(1))
        self.c1.requires_grad = True
    
    def forward(self, x):
        out = self.w1 + self.w2*(torch.cos(1.7214206321039962*(x[:,0]-self.w4))) + self.c1*x[:,1]
        return out



#%% split x,y
y_train = train[:,0]  # LST
x_train = train[:,1:] # [x, y, doy, index, air temp]



#%% training
time1 = time.time()
ibatch = 0


# training range
starti = args.n1
endi = args.n2


# saving parameters here
# empty array, n, 4 parameters, 100 sets of parameters
save_params = np.zeros([endi-starti,4,100],dtype=np.float32)



## entering training
for repeat in range(1):
    for ibatch in range(starti,endi):
        if ibatch%100==0:
            print(ibatch,time.time()-time1)
            time1 = time.time()
        
        
        ### batch indexing
        batchidx = x_train[:,3] == idx2[idx3[ibatch]]
        x_batch = x_train[batchidx]
        y_batch = y_train[batchidx]
        
        
        x_batch = torch.Tensor(x_batch)
        y_batch = torch.Tensor(y_batch)
        
        
        # train, maximum re-train attempt=3
        for ntry in range(3):
            model = CosNet()
            
            #%
            criterion = nn.L1Loss()
            optimizer = optim.Adam(model.parameters(), lr=0.1)
            
            # training
            model.train()
            
            minn = 10
            count = 0
            # for i in range(1000):
            for i in range(500):
                if count>200: # if do not see a smaller loss over the past 200 epochs, break training
                    break
                count += 1
                optimizer.zero_grad()
                output = model(x_batch[:,[2,4]])
                loss = criterion(output, y_batch)
                loss.backward()
                optimizer.step()
                if loss.item()<minn:
                    minn = loss.item()
                    count = 0
            if loss.item()<1.5: # if loss small enough, no need to retrain. This value is empirical
                break
                
    
        
        #% ensemble, get the 100 sets of parameters over the next 200 epochs.
        model.train()
        count = 0
        for i in range(200):
            count += 1/2
            optimizer.zero_grad()
            output = model(x_batch[:,[2,4]])
            loss = criterion(output, y_batch)
            loss.backward()
            optimizer.step()
            
            if i%2==0: # every 2 epochs to get one set of parameters
                save_params[ibatch-starti,0,i//2] = model.w1.detach().numpy()[0]
                save_params[ibatch-starti,1,i//2] = model.w2.detach().numpy()[0]
                save_params[ibatch-starti,2,i//2] = model.w4.detach().numpy()[0]
                save_params[ibatch-starti,3,i//2] = model.c1.detach().numpy()[0]
                
        

#%% save model parameters
path = site + '/model'
isExist = os.path.exists(path)
if not isExist:
    os.makedirs(path)       
        
# save parameters
name = path + '/'+site + year+'-m-' + format(args.n1,'07d') +'-'+ format(args.n2,'07d') + '.npy'
np.save(name, save_params)











