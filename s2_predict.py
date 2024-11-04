# -*- coding: utf-8 -*-
"""
Created on Thu Aug 29 02:08:59 2024

@author: skrisliu

Create a folder named 'save100' before running this script
"""

import numpy as np
import torch
import glob
import matplotlib.pyplot as plt
import os


site = 'nyc'
year = '2023'


#%% shape of the image
mask = glob.glob(site + '/cloudmask/y' + year + '/*.npy')
mask = np.load(mask[0])
mask = np.ones(mask.shape)
mask = mask==1
size = mask


#%% coarse temperature
gtemp = np.load(site+'/'+site+year+'lst.npy')
gtemp = gtemp /  10
gtemp = gtemp.reshape(-1,1)
doy = np.arange(1,366) / 100
doy = doy.reshape(-1,1)

x_pre = np.concatenate([doy, gtemp], axis=-1)


#%% load model, predict, save 100
idx2 = np.load(site + '/' + site + year + '-idx2.npy')


if True:
    models = []
    knames = glob.glob(site + '/model/'+ site+year + '*.npy')
    kname = knames[0]
    for kname in knames:
        model = np.load(kname)
        models.append(model)
    model = np.concatenate(models,axis=0)
    

    ## iter
    day = 200
    modeln = 70
    for day in np.arange(365):
        print(day)
        
        newpath = site+'/save100/doy'+format(day+1,'03')
        if not os.path.exists(newpath):
            os.makedirs(newpath)
        for modeln in np.arange(100):
            modelx = model[:,:,modeln]
            out = modelx[:,0] + modelx[:,1]*np.cos(1.7214206321039962* (x_pre[day,0]-modelx[:,2]) ) + x_pre[day,1]*modelx[:,3]
            out = (out*10000+25000)*0.00341802 + 149.0 - 273.15  # this output is to degree C
            outim = out.reshape(size.shape[0],size.shape[1],1)
            
            # plt.imshow(outim)
            savep = site+'/save100/doy'+format(day+1,'03') + '/pre' + format(modeln,'03')
            np.save(savep, outim)


































