## DELAG

Deep Ensemble Learning with enhanced ATC and GP


### Requirements
Tested on Python 3.9, cuda=11.7, Intel i5 8500, 32 GB RAM, Nvidia 1080Ti 11GB
```
torch==1.13.1
gpytorch==1.11
numpy==1.26.4
```

### 

1. Run s1_train.py, train individual pixels and get 100 sets of ATC parameters 

2. Run s2_predict.py, get 100 predictions for ensemble learning 

3. Run s3_show_predict.py, show predictions and test performance before GP

4. Run s4_GP.py, train GP model for a specific date with partial observations. In the demo, date = '20230402', which doy = 92

