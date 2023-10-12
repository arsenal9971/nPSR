# Utilities from dave
import sys
import os
sys.path.append('../fno_utils/')

import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F

import matplotlib.pyplot as plt
from utilities3 import *

import operator
from functools import reduce
from functools import partial

from timeit import default_timer

from Adam import Adam

torch.manual_seed(0)
np.random.seed(0)
import gc

sys.path.append('../src/')
from importlib import reload
import utils
reload(utils)

width = 30
modes = 10
training_steps = 10000
learning_rate = 0.0025
scheduler_step = 100
scheduler_gamma = 0.5

root_path = '/scratch/Data/Fourier_Neural_Data/Shapenet_FNO_nonuniform_train/'
batch_size = 20
n_samples = 10000
sigma = 1.5

device = torch.device('cuda')
neuralpoisson = utils.NeuralPoisson(modes, modes, modes, width).cuda()

optimizer = Adam(model.parameters(), lr=learning_rate, weight_decay=1e-4)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=scheduler_step, gamma=scheduler_gamma)

checkpoint_path = '../checkpoints/checkpoints_n_samples_'+str(n_samples)

# Start training
best_loss = 10.20
if 0:
    model.load_state_dict(torch.load(checkpoint_path))
    model.eval()

# Training loop
myloss = LpLoss(size_average=False)
train_mse = 0
train_l2 = 0
S = T_in = T = 64 
for i in range(10000):
    model.train()
    # Loading data
    voxels_batch, divergence_tensor_batch = loader_train.generate_data()
    
    voxels_batch = torch.Tensor(voxels_batch).cuda()
    divergence_tensor_batch = torch.Tensor(divergence_tensor_batch).cuda()
    
    div_normalizer = UnitGaussianNormalizer(divergence_tensor_batch)
    divergence_tensor_batch = div_normalizer.encode(divergence_tensor_batch)
    
    vox_normalizer = UnitGaussianNormalizer(voxels_batch)
    voxels_batch = vox_normalizer.encode(voxels_batch)
    
    divergence_tensor_batch_resh = divergence_tensor_batch.reshape(batch_size,S,S,1,T_in).repeat([1,1,1,T,1])
    
    optimizer.zero_grad()
    pred = model(divergence_tensor_batch_resh).view(batch_size, S, S, T)

    mse = F.mse_loss(pred, voxels_batch, reduction='mean')
    
    voxels_batch = vox_normalizer.decode(voxels_batch)
    pred = div_normalizer.decode(pred)
    l2 = myloss(pred.view(batch_size, -1), voxels_batch.view(batch_size, -1))
    
    gc.collect()
    torch.cuda.empty_cache()
    
    print('Training step ', i, ' of ', 100000, ' with loss ', l2.cpu().detach().numpy())
    
    if l2 < best_loss:
        print("Saving checkpoints")
        torch.save(model.state_dict(), checkpoint_path)
        best_loss = l2
        
    l2.backward()

    optimizer.step()
    scheduler.step()
    gc.collect()
    torch.cuda.empty_cache()