# -*- coding: utf-8 -*-
"""
Created on Thu Aug 29 02:16:48 2024

@author: skrisliu
"""

import numpy as np
import glob
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, mean_squared_error, root_mean_squared_error, r2_score
from scipy.stats import gaussian_kde
from scipy.stats import linregress, pearsonr
from  sklearn.linear_model import LinearRegression



site = 'nyc'
year = '2023'

date = '20230402'
doy = 92


#%%
fp = site + '/save100/doy' + format(doy,'03d') + '/pre'



#%%
ims = []
for i in range(100):
    im_ = np.load(fp + format(i,'03d') + '.npy')
    ims.append(im_)
    

#%%
ims = np.array(ims)
ims = np.squeeze(ims)

im_m = np.mean(ims,axis=0)
im_l = np.percentile(ims,2.5,axis=0)
im_u = np.percentile(ims,97.5,axis=0)



#%%
plt.imshow(im_m)
plt.show()





#%% Get cloud mask, im
mask = np.load(site+'/cloudmask/y'+year+'/cloud'+date+'.npy')
cloudmask = ~mask


# im
im_gt = glob.glob(site + '/order/y' + year + '/doy' + date + '*.npy')
im_gt = np.load(im_gt[0])[:,:,0]
im_gt = im_gt*0.00341802 + 149.0 - 273.15


#%%
m = {}

y_pre = im_m[cloudmask]
y_gt = im_gt[cloudmask]


res = linregress(y_gt,y_pre)
res2 = LinearRegression().fit(y_gt.reshape(-1,1),y_pre.reshape(-1,1))
m['mae'] = mean_absolute_error(y_gt, y_pre)
m['rmse'] = root_mean_squared_error(y_gt, y_pre)
m['bias'] = np.mean(y_pre) - np.mean(y_gt)
m['r'] = res.rvalue
m['r2'] = r2_score(y_gt, y_pre)


#%% Use K as unit, KDE will take a while to estimate dot density
y_gt2 = y_gt + 273.15
y_pre2 = y_pre + 273.15

# kde, show dot density, this will take a while
xy = np.vstack([y_gt2, y_pre2])
z = gaussian_kde(xy)(xy)
idx = z.argsort()
x, y, z = y_gt2[idx], y_pre2[idx], z[idx]



#%%
fig = plt.figure(figsize=(5,4),dpi=200)
plt.plot(np.arange(200,400),np.arange(200,400),'--',color='black',lw=1,alpha=0.7)
plt.scatter(x,y,s=30,c=z,alpha=0.10,cmap='cividis')
plt.ylim(270,330)
plt.xlim(270,330)
plt.xlabel('Original Landsat LST (K)')
plt.ylabel('Reconstructed Landsat LST (K)')

ax = plt.gca()
txt = 'rmse=' + format(m['rmse'],'.3f') + ' K' + '\nbias=' + format(m['bias'],'.3f') + ' K' + '\n' + r'R$^2$=' + format(m['r2'],'.3f') 
plt.text(0.04,0.96, txt,
         horizontalalignment='left',verticalalignment='top',
         transform = ax.transAxes)

plt.tight_layout(pad=0.2)
# plt.savefig('fig/scatterplot1.svg')  # need to create fig folder before saving image to disk
# plt.savefig('fig/scatterplot1.pdf')
# plt.savefig('fig/scatterplot1.png')
plt.show()


























