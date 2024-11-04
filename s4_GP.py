# -*- coding: utf-8 -*-
"""
Created on Thu Aug 29 02:16:48 2024

@author: skrisliu
"""

import os
import numpy as np
import glob
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, mean_squared_error, root_mean_squared_error, r2_score
from scipy.stats import gaussian_kde
from scipy.stats import linregress
from sklearn.linear_model import LinearRegression
import copy

import torch
import gpytorch
from torch.utils.data import TensorDataset, DataLoader
from datetime import datetime


site = 'nyc'
year = '2023'

date = '20230402'
doy = 92




FEATURE = 'TEMP'
ADDTRAIN = True
SINGLE = True
LINEARMEAN = True

batch_size = 2048



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

ims = np.transpose(ims,[1,2,0])
ims = np.float32(ims)



#%%
plt.imshow(im_m)
plt.show()





#%% Get cloud mask, im
mask = np.load(site+'/cloudmask/y'+year+'/cloud'+date+'.npy')


# im
im_gt = glob.glob(site + '/order/y' + year + '/doy' + date + '*.npy')
im_gt = np.load(im_gt[0])[:,:,0]
im_gt = im_gt*0.00341802 + 149.0 - 273.15
im_gt = np.float32(im_gt)


#%% Get Train
if FEATURE=='TEMP':
    # using past temperature as feature, need to have clean past temperature
    # this past temperature is precreated from all clean data
    im = np.load(site+'/clean/'+site+year+'.npy')
    im = im[:,:,:,0]
    
    # need to exclude the clean data from the date of prediction
    imdoy = np.load(site+'/clean/'+site+year+'doy.npy')
    im = im[imdoy!=doy,:]

    im = np.transpose(im, [1,2,0])
    im = im*0.00341802 + 149.0 - 273.15
    

# normalize features to 0-1
for i in range(im.shape[-1]):
    im[:,:,i] = ( im[:,:,i]-im[:,:,i].min() ) / ( im[:,:,i].max()-im[:,:,i].min() + 1e-6 )

im = np.float32(im)


#%% Get Train residual
y_gt = im_gt[mask]
y_pre = ims[mask,:]

### xy points
xy1 = np.zeros([mask.shape[0],mask.shape[1],1])
xy2 = np.zeros([mask.shape[0],mask.shape[1],1])
for i in range(xy1.shape[0]):
    for j in range(xy1.shape[1]):
        xy1[i,j] = i
        xy2[i,j] = j


xy1 = xy1.astype(np.float32)
xy2 = xy2.astype(np.float32)


# normalize spatial index to close 0-1
SCALE = np.max(xy1.shape)
xy1 = xy1/SCALE
xy2 = xy2/SCALE


im = np.concatenate([im,xy1,xy2],axis=-1)
IMZ = im.shape[-1]


#%%
x_tr = im[mask]
x_trs = np.zeros([x_tr.shape[0]*100,x_tr.shape[1]],dtype=np.float32)
y_trs = np.zeros([x_tr.shape[0]*100,],dtype=np.float32)


#%% ims[mask,0], im_gt[mask]
i = 0
for i in range(100):
    ygt = im_gt[mask]
    ypre = ims[mask,i]

    res = ygt-ypre
    
    x_trs[i*x_tr.shape[0]:(i+1)*x_tr.shape[0], :] = x_tr
    y_trs[i*x_tr.shape[0]:(i+1)*x_tr.shape[0], ] = res


x_tr = x_trs
y_tr = y_trs


if ADDTRAIN:
    x_tr2 = copy.deepcopy(x_tr)
    y_tr2 = copy.deepcopy(y_tr)



#%% Get Test Set
x_te = im[~mask]
y_te = im_gt[~mask] - im_m[~mask]



#%% linear regression parameters pre-loaded to GP model
reg = LinearRegression().fit(x_tr, y_tr)
pre1_lnearReg = reg.predict(x_te)

rr_raw = r2_score(y_te,pre1_lnearReg)

model0_params = reg.coef_
model0_params = model0_params.reshape(-1,1)
model0_params = np.float32(model0_params)


#%% Tensor
x_tr = torch.Tensor(x_tr).contiguous()
x_te = torch.Tensor(x_te).contiguous()
y_tr = torch.Tensor(y_tr).contiguous()
y_te = torch.Tensor(y_te).contiguous()


if torch.cuda.is_available():
    x_tr, y_tr, x_te, y_te = x_tr.cuda(), y_tr.cuda(), x_te.cuda(), y_te.cuda()
    if ADDTRAIN:
        x_tr2 = torch.Tensor(x_tr2).contiguous()
        y_tr2 = torch.Tensor(y_tr2).contiguous()


#%% dataloader
train_dataset = TensorDataset(x_tr, y_tr)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True) 

test_dataset = TensorDataset(x_te, y_te)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

if ADDTRAIN:
    train2_dataset = TensorDataset(x_tr2, y_tr2)
    train2_loader = DataLoader(train2_dataset, batch_size=batch_size, shuffle=False) 



#%% define network and GP
# there are two GP models here, GPS (spatial GP) and GP (all features). Using GP here


from gpytorch.models import ApproximateGP
from gpytorch.variational import CholeskyVariationalDistribution
from gpytorch.variational import VariationalStrategy
from gpytorch.means import Mean
from gpytorch.constraints import Interval


class LinearMean(Mean):
    def __init__(self, input_size, batch_shape=torch.Size(), bias=True):
        super().__init__()
        self.register_parameter(name="weights", parameter=torch.nn.Parameter(torch.randn(*batch_shape, input_size, 1)))
        if bias:
            self.register_parameter(name="bias", parameter=torch.nn.Parameter(torch.randn(*batch_shape, 1)))
        else:
            self.bias = None
        self.weights = torch.nn.Parameter(torch.from_numpy(model0_params))

    def forward(self, x):
        res = x.matmul(self.weights).squeeze(-1)
        if self.bias is not None:
            res = res + self.bias
        return res

# GPS
class GPS(ApproximateGP):
    def __init__(self, inducing_points,likelihood):
        variational_distribution = CholeskyVariationalDistribution(inducing_points.size(0))
        variational_strategy = VariationalStrategy(self, inducing_points, variational_distribution, learn_inducing_locations=True)
        super(GPS, self).__init__(variational_strategy)
        if LINEARMEAN:
            self.mean_module = LinearMean(input_size=IMZ)
        else:
            self.mean_module = gpytorch.means.ConstantMean()
        self.covar_module = gpytorch.kernels.ScaleKernel(
            gpytorch.kernels.MaternKernel(lengthscale_constraint=Interval(2/SCALE, 100/SCALE)))
        init_lengthscale = 10/SCALE
        self.covar_module.base_kernel.initialize(lengthscale=init_lengthscale)
        self.likelihood = likelihood
        

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x[:,-2:]) 
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)


class GP(ApproximateGP):
    def __init__(self, inducing_points,likelihood):
        variational_distribution = CholeskyVariationalDistribution(inducing_points.size(0))
        variational_strategy = VariationalStrategy(self, inducing_points, variational_distribution, learn_inducing_locations=True)
        super(GP, self).__init__(variational_strategy)
        if LINEARMEAN:
            self.mean_module = LinearMean(input_size=IMZ)
        else:
            self.mean_module = gpytorch.means.ConstantMean()
        if SINGLE:
            self.covar_module = gpytorch.kernels.ScaleKernel(
                gpytorch.kernels.MaternKernel(lengthscale_constraint=Interval(2/SCALE, 100/SCALE)))   # 2-100 pixel range
        else:
            self.covar_module = gpytorch.kernels.ScaleKernel(
                gpytorch.kernels.MaternKernel(ard_num_dims=IMZ,lengthscale_constraint=Interval(2/SCALE, 100/SCALE)))
        init_lengthscale = 10/SCALE
        self.covar_module.base_kernel.initialize(lengthscale=init_lengthscale)
        self.likelihood = likelihood
        

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x) 
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)




#%% build model
inducing_points = x_tr[:500, :]
likelihood = gpytorch.likelihoods.GaussianLikelihood()

model = GP(inducing_points=inducing_points,likelihood=likelihood)

if torch.cuda.is_available():
    model = model.cuda()
    likelihood = likelihood.cuda()



#%% training 1, learning rate=0.01
print('Enter Training #1')
num_epochs = 6   # for fast result
num_epochs = 30  # for best result

model.train()
likelihood.train()


lr = 0.01
optimizer = torch.optim.Adam(model.parameters(), lr=lr)

mll = gpytorch.mlls.VariationalELBO(likelihood, model, num_data=y_tr.size(0))

# training starts here
epochs_iter = np.arange(num_epochs)
for i in epochs_iter:
    printloss = 0
    count = 0
    for j, (x_batch, y_batch) in enumerate(train_loader):
        optimizer.zero_grad()
        output = model(x_batch)
        loss = -mll(output, y_batch)
        if SINGLE:
            print('Iter %d/%d - Loss: %.3f   lengthscale: %.3f   noise: %.3f' % (
                i + 1, num_epochs, loss.item(), 
                model.covar_module.base_kernel.lengthscale, 
                model.likelihood.noise.item()
            ))
        else:
            print(i + 1, num_epochs, 
                  format(loss.item(), '.3f'), 
                  format(model.covar_module.base_kernel.lengthscale[0].detach().cpu().numpy()[0], '.3f'), 
                  format(model.covar_module.base_kernel.lengthscale[0].detach().cpu().numpy()[-2], '.3f'), 
                  format(model.covar_module.base_kernel.lengthscale[0].detach().cpu().numpy()[-1], '.3f'),
                  format(model.likelihood.noise.item(), '.3f') )
        loss.backward()
        optimizer.step()
        printloss += loss.item()
        count += 1
    print(i,printloss/count)




#%% training 2, smaller learning rate=0.001
print('Enter Training #2')
num_epochs = 2   # for fast result
num_epochs = 10  # for best result
 
model.train()
likelihood.train()


lr = 0.001
optimizer = torch.optim.Adam(model.parameters(), lr=lr)

epochs_iter = np.arange(num_epochs)
for i in epochs_iter:
    printloss = 0
    count = 0
    for j, (x_batch, y_batch) in enumerate(train_loader):
        optimizer.zero_grad()
        output = model(x_batch)
        loss = -mll(output, y_batch)
        if SINGLE:
            print('Iter %d/%d - Loss: %.3f   lengthscale: %.3f   noise: %.3f' % (
                i + 1, num_epochs, loss.item(), 
                model.covar_module.base_kernel.lengthscale, 
                model.likelihood.noise.item()
            ))
        else:
            print(i + 1, num_epochs, 
                  format(loss.item(), '.3f'), 
                  format(model.covar_module.base_kernel.lengthscale[0].detach().cpu().numpy()[0], '.3f'), 
                  format(model.covar_module.base_kernel.lengthscale[0].detach().cpu().numpy()[-2], '.3f'), 
                  format(model.covar_module.base_kernel.lengthscale[0].detach().cpu().numpy()[-1], '.3f'),
                  format(model.likelihood.noise.item(), '.3f') )
        loss.backward()
        optimizer.step()
        printloss += loss.item()
        count += 1
    print(i,printloss/count)



#%%
now = datetime.now()
dt = now.strftime("%Y%m%d-%H%M%S")


path = site + '/modelgp'
isExist = os.path.exists(path)
if not isExist:
    os.makedirs(path)   

torch.save(model.state_dict(), site+'/modelgp/'+date+dt+'.pth')


#%% testing
model.eval()
likelihood.eval()
means = torch.tensor([0.])
lowers = torch.tensor([0.])
uppers = torch.tensor([0.])
with torch.no_grad():
    for x_batch, y_batch in test_loader:
        preds = model(x_batch)
        means = torch.cat([means, preds.mean.cpu()]) # only get the mean of the prediction
        
        # std
        lower, upper = preds.confidence_region()
        lower = lower.cpu()
        upper = upper.cpu()
        
        lowers = torch.cat([lowers, lower])
        uppers = torch.cat([uppers, upper])
        
means = means[1:]

# test summary
pp1 = y_te.cpu().numpy()
pp2 = means.numpy()

rr = r2_score(pp1,pp2)
mae = mean_absolute_error(pp1,pp2)
rmse = root_mean_squared_error(pp1,pp2)
print(mae,rmse,rr)   # 0.39721683 0.5546582 0.466806753945724


res = {}
res['pre1'] = [mae, rmse, rr]


#%%
ypre1 = im_m[~mask]
ypre2 = ypre1 + pp2

ypre2_l = im_l[~mask] + lowers[1:].cpu().numpy()
ypre2_u = im_u[~mask] + uppers[1:].cpu().numpy()


rr = r2_score(im_gt[~mask],ypre2)
mae = mean_absolute_error(im_gt[~mask],ypre2)
rmse = root_mean_squared_error(im_gt[~mask],ypre2)

res['pre2'] = [mae, rmse, rr]


rr = r2_score(im_gt[~mask],ypre1)
mae = mean_absolute_error(im_gt[~mask],ypre1)
rmse = root_mean_squared_error(im_gt[~mask],ypre1)
res['pre0'] = [mae, rmse, rr]


percs0 = np.sum(np.logical_and(im_l[~mask]<im_gt[~mask], im_u[~mask]>im_gt[~mask])) / ypre1.shape[0] *100
percs = np.sum(np.logical_and(ypre2_l<im_gt[~mask], ypre2_u>im_gt[~mask])) / ypre1.shape[0] *100

# checking 95% CI, ~93.12%
# the 1st is ATC only, the 2nd is ATC+GP
print('95% CI: ATC, ATC+GP')
print(percs0,percs)

print('ATC result [mae, rmse, rr]')
print(res['pre0'])

print('ATC + GP result [mae, rmse, rr]')
print(res['pre2'])


#%%
if True:
    path = 'tmp'
    isExist = os.path.exists(path)
    if not isExist:
        os.makedirs(path)   
    
    np.save('tmp/ypre1.npy',ypre1)
    np.save('tmp/ypre2.npy',pp2)
    
    np.save('tmp/ypre1_l.npy',im_l[~mask])
    np.save('tmp/ypre1_u.npy',im_u[~mask])

    np.save('tmp/ypre2_l.npy',lowers[1:].cpu().numpy())
    np.save('tmp/ypre2_u.npy',uppers[1:].cpu().numpy())
    
    np.save('tmp/ygt.npy',im_gt[~mask])
    






















