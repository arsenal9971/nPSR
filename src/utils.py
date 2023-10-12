import sys
sys.path.append('../fno_utils/')

from scipy.ndimage.interpolation import rotate

import ipywidgets as ipyw
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

from skimage.filters import threshold_otsu, threshold_minimum, threshold_triangle

import os
import odl

import scipy

import open3d as o3d

import binvox_rw
from skimage import measure
import random

from torch import Tensor


from collections import defaultdict
import torch.nn.functional as F

import trimesh
import mcubes


from typing import Optional
import numpy as np
import pycolmap
import plotly.graph_objects as go

################################################################
# 3d fourier layers
################################################################

class SpectralConv3d(nn.Module):
    def __init__(self, in_channels, out_channels, modes1, modes2, modes3):
        super(SpectralConv3d, self).__init__()

        """
        3D Fourier layer. It does FFT, linear transform, and Inverse FFT.    
        """

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.modes1 = modes1 #Number of Fourier modes to multiply, at most floor(N/2) + 1
        self.modes2 = modes2
        self.modes3 = modes3

        self.scale = (1 / (in_channels * out_channels))
        self.weights1 = nn.Parameter(self.scale * torch.rand(in_channels, out_channels, self.modes1, self.modes2, self.modes3, dtype=torch.cfloat))
        self.weights2 = nn.Parameter(self.scale * torch.rand(in_channels, out_channels, self.modes1, self.modes2, self.modes3, dtype=torch.cfloat))
        self.weights3 = nn.Parameter(self.scale * torch.rand(in_channels, out_channels, self.modes1, self.modes2, self.modes3, dtype=torch.cfloat))
        self.weights4 = nn.Parameter(self.scale * torch.rand(in_channels, out_channels, self.modes1, self.modes2, self.modes3, dtype=torch.cfloat))

    # Complex multiplication
    def compl_mul3d(self, input, weights):
        # (batch, in_channel, x,y,t ), (in_channel, out_channel, x,y,t) -> (batch, out_channel, x,y,t)
        return torch.einsum("bixyz,ioxyz->boxyz", input, weights)

    def forward(self, x):
        batchsize = x.shape[0]
        #Compute Fourier coeffcients up to factor of e^(- something constant)
        x_ft = torch.fft.rfftn(x, dim=[-3,-2,-1])

        # Multiply relevant Fourier modes
        out_ft = torch.zeros(batchsize, self.out_channels, x.size(-3), x.size(-2), x.size(-1)//2 + 1, dtype=torch.cfloat, device=device)
        out_ft[:, :, :self.modes1, :self.modes2, :self.modes3] = \
            self.compl_mul3d(x_ft[:, :, :self.modes1, :self.modes2, :self.modes3], self.weights1)
        out_ft[:, :, -self.modes1:, :self.modes2, :self.modes3] = \
            self.compl_mul3d(x_ft[:, :, -self.modes1:, :self.modes2, :self.modes3], self.weights2)
        out_ft[:, :, :self.modes1, -self.modes2:, :self.modes3] = \
            self.compl_mul3d(x_ft[:, :, :self.modes1, -self.modes2:, :self.modes3], self.weights3)
        out_ft[:, :, -self.modes1:, -self.modes2:, :self.modes3] = \
            self.compl_mul3d(x_ft[:, :, -self.modes1:, -self.modes2:, :self.modes3], self.weights4)

        #Return to physical space
        x = torch.fft.irfftn(out_ft, s=(x.size(-3), x.size(-2), x.size(-1)))
        return x
    
class NeuralPoisson(nn.Module):
    def __init__(self, modes1, modes2, modes3, width):
        super(NeuralPoisson, self).__init__()

        """
        The overall network. It contains 4 layers of the Fourier layer.
        1. Lift the input to the desire channel dimension by self.fc0 .
        2. 4 layers of the integral operators u' = (W + K)(u).
            W defined by self.w; K defined by self.conv .
        3. Project from the channel space to the output space by self.fc1 and self.fc2 .
        
        input: the solution of the first 10 timesteps + 3 locations (u(1, x, y), ..., u(10, x, y),  x, y, t). It's a constant function in time, except for the last index.
        input shape: (batchsize, x=64, y=64, t=40, c=13)
        output: the solution of the next 40 timesteps
        output shape: (batchsize, x=64, y=64, t=40, c=1)
        """

        self.modes1 = modes1
        self.modes2 = modes2
        self.modes3 = modes3
        self.width = width
        self.padding = 6 # pad the domain if input is non-periodic
        self.fc0 = nn.Linear(67, self.width)
        # input channel is 12: the solution of the first 10 timesteps + 3 locations (u(1, x, y), ..., u(10, x, y),  x, y, t)

        self.conv0 = SpectralConv3d(self.width, self.width, self.modes1, self.modes2, self.modes3)
        self.conv1 = SpectralConv3d(self.width, self.width, self.modes1, self.modes2, self.modes3)
        self.conv2 = SpectralConv3d(self.width, self.width, self.modes1, self.modes2, self.modes3)
        self.conv3 = SpectralConv3d(self.width, self.width, self.modes1, self.modes2, self.modes3)
        self.w0 = nn.Conv3d(self.width, self.width, 1)
        self.w1 = nn.Conv3d(self.width, self.width, 1)
        self.w2 = nn.Conv3d(self.width, self.width, 1)
        self.w3 = nn.Conv3d(self.width, self.width, 1)
        self.bn0 = torch.nn.BatchNorm3d(self.width)
        self.bn1 = torch.nn.BatchNorm3d(self.width)
        self.bn2 = torch.nn.BatchNorm3d(self.width)
        self.bn3 = torch.nn.BatchNorm3d(self.width)

        self.fc1 = nn.Linear(self.width, 128)
        self.fc2 = nn.Linear(128, 1)

    def forward(self, x):
        grid = self.get_grid(x.shape, device)
        x = torch.cat((x, grid), dim=-1)
        x = self.fc0(x)
        x = x.permute(0, 4, 1, 2, 3)
        x = F.pad(x, [0,self.padding]) # pad the domain if input is non-periodic

        x1 = self.conv0(x)
        x2 = self.w0(x)
        x = x1 + x2
        x = F.gelu(x)

        x1 = self.conv1(x)
        x2 = self.w1(x)
        x = x1 + x2
        x = F.gelu(x)

        x1 = self.conv2(x)
        x2 = self.w2(x)
        x = x1 + x2
        x = F.gelu(x)

        x1 = self.conv3(x)
        x2 = self.w3(x)
        x = x1 + x2

        x = x[..., :-self.padding]
        x = x.permute(0, 2, 3, 4, 1) # pad the domain if input is non-periodic
        x = self.fc1(x)
        x = F.gelu(x)
        x = self.fc2(x)
        return x

    def get_grid(self, shape, device):
        batchsize, size_x, size_y, size_z = shape[0], shape[1], shape[2], shape[3]
        gridx = torch.tensor(np.linspace(0, 1, size_x), dtype=torch.float)
        gridx = gridx.reshape(1, size_x, 1, 1, 1).repeat([batchsize, 1, size_y, size_z, 1])
        gridy = torch.tensor(np.linspace(0, 1, size_y), dtype=torch.float)
        gridy = gridy.reshape(1, 1, size_y, 1, 1).repeat([batchsize, size_x, 1, size_z, 1])
        gridz = torch.tensor(np.linspace(0, 1, size_z), dtype=torch.float)
        gridz = gridz.reshape(1, 1, 1, size_z, 1).repeat([batchsize, size_x, size_y, 1, 1])
        return torch.cat((gridx, gridy, gridz), dim=-1).to(device)
    
class FNO3d_batchnorm(nn.Module):
    def __init__(self, modes1, modes2, modes3, width):
        super(FNO3d_batchnorm, self).__init__()

        """
        The overall network. It contains 4 layers of the Fourier layer.
        1. Lift the input to the desire channel dimension by self.fc0 .
        2. 4 layers of the integral operators u' = (W + K)(u).
            W defined by self.w; K defined by self.conv .
        3. Project from the channel space to the output space by self.fc1 and self.fc2 .
        
        input: the solution of the first 10 timesteps + 3 locations (u(1, x, y), ..., u(10, x, y),  x, y, t). It's a constant function in time, except for the last index.
        input shape: (batchsize, x=64, y=64, t=40, c=13)
        output: the solution of the next 40 timesteps
        output shape: (batchsize, x=64, y=64, t=40, c=1)
        """

        self.modes1 = modes1
        self.modes2 = modes2
        self.modes3 = modes3
        self.width = width
        self.padding = 6 # pad the domain if input is non-periodic
        self.fc0 = nn.Linear(67, self.width)
        # input channel is 12: the solution of the first 10 timesteps + 3 locations (u(1, x, y), ..., u(10, x, y),  x, y, t)

        self.conv0 = SpectralConv3d(self.width, self.width, self.modes1, self.modes2, self.modes3)
        self.conv1 = SpectralConv3d(self.width, self.width, self.modes1, self.modes2, self.modes3)
        self.conv2 = SpectralConv3d(self.width, self.width, self.modes1, self.modes2, self.modes3)
        self.conv3 = SpectralConv3d(self.width, self.width, self.modes1, self.modes2, self.modes3)
        self.w0 = nn.Conv3d(self.width, self.width, 1)
        self.w1 = nn.Conv3d(self.width, self.width, 1)
        self.w2 = nn.Conv3d(self.width, self.width, 1)
        self.w3 = nn.Conv3d(self.width, selfgen_batch_exampl.width, 1)
        self.bn0 = torch.nn.BatchNorm3d(self.width)
        self.bn1 = torch.nn.BatchNorm3d(self.width)
        self.bn2 = torch.nn.BatchNorm3d(self.width)
        self.bn3 = torch.nn.BatchNorm3d(self.width)

        self.fc1 = nn.Linear(self.width, 128)
        self.fc2 = nn.Linear(128, 1)

    def forward(self, x):
        grid = self.get_grid(x.shape, device)
        x = torch.cat((x, grid), dim=-1)
        x = self.fc0(x)
        x = x.permute(0, 4, 1, 2, 3)
        x = F.pad(x, [0,self.padding]) # pad the domain if input is non-periodic

        x1 = self.conv0(x)
        x2 = self.w0(x)
        x = x1 + x2
        x = F.gelu(x)
        x = self.bn0(x)

        x1 = self.conv1(x)
        x2 = self.w1(x)
        x = x1 + x2
        x = F.gelu(x)
        x = self.bn1(x)

        x1 = self.conv2(x)
        x2 = self.w2(x)
        x = x1 + x2
        x = F.gelu(x)
        x = self.bn2(x)

        x1 = self.conv3(x)
        x2 = self.w3(x)
        x = x1 + x2
        x = self.bn3(x)

        x = x[..., :-self.padding]
        x = x.permute(0, 2, 3, 4, 1) # pad the domain if input is non-periodic
        x = self.fc1(x)
        x = F.gelu(x)
        x = self.fc2(x)
        return x

    def get_grid(self, shape, device):
        batchsize, size_x, size_y, size_z = shape[0], shape[1], shape[2], shape[3]
        gridx = torch.tensor(np.linspace(0, 1, size_x), dtype=torch.float)
        gridx = gridx.reshape(1, size_x, 1, 1, 1).repeat([batchsize, 1, size_y, size_z, 1])
        gridy = torch.tensor(np.linspace(0, 1, size_y), dtype=torch.float)
        gridy = gridy.reshape(1, 1, size_y, 1, 1).repeat([batchsize, size_x, 1, size_z, 1])
        gridz = torch.tensor(np.linspace(0, 1, size_z), dtype=torch.float)
        gridz = gridz.reshape(1, 1, 1, size_z, 1).repeat([batchsize, size_x, size_y, 1, 1])
        return torch.cat((gridx, gridy, gridz), dim=-1).to(device)

class FNO3d_dropout(nn.Module):
    def __init__(self, modes1, modes2, modes3, width):
        super(FNO3d_dropout, self).__init__()

        """
        The overall network. It contains 4 layers of the Fourier layer.
        1. Lift the input to the desire channel dimension by self.fc0 .
        2. 4 layers of the integral operators u' = (W + K)(u).
            W defined by self.w; K defined by self.conv .
        3. Project from the channel space to the output space by self.fc1 and self.fc2 .
        
        input: the solution of the first 10 timesteps + 3 locations (u(1, x, y), ..., u(10, x, y),  x, y, t). It's a constant function in time, except for the last index.
        input shape: (batchsize, x=64, y=64, t=40, c=13)
        output: the solution of the next 40 timesteps
        output shape: (batchsize, x=64, y=64, t=40, c=1)
        """

        self.modes1 = modes1
        self.modes2 = modes2
        self.modes3 = modes3
        self.width = width
        self.padding = 6 # pad the domain if input is non-periodic
        self.fc0 = nn.Linear(67, self.width)
        # input channel is 12: the solution of the first 10 timesteps + 3 locations (u(1, x, y), ..., u(10, x, y),  x, y, t)
        
        self.dropout = torch.nn.Dropout(p = 0.5)
        self.conv0 = SpectralConv3d(self.width, self.width, self.modes1, self.modes2, self.modes3)
        self.conv1 = SpectralConv3d(self.width, self.width, self.modes1, self.modes2, self.modes3)
        self.conv2 = SpectralConv3d(self.width, self.width, self.modes1, self.modes2, self.modes3)
        self.conv3 = SpectralConv3d(self.width, self.width, self.modes1, self.modes2, self.modes3)
        self.w0 = nn.Conv3d(self.width, self.width, 1)
        self.w1 = nn.Conv3d(self.width, self.width, 1)
        self.w2 = nn.Conv3d(self.width, self.width, 1)
        self.w3 = nn.Conv3d(self.width, self.width, 1)
        self.bn0 = torch.nn.BatchNorm3d(self.width)
        self.bn1 = torch.nn.BatchNorm3d(self.width)
        self.bn2 = torch.nn.BatchNorm3d(self.width)
        self.bn3 = torch.nn.BatchNorm3d(self.width)

        self.fc1 = nn.Linear(self.width, 128)
        self.fc2 = nn.Linear(128, 1)

    def forward(self, x):
        grid = self.get_grid(x.shape, device)
        x = torch.cat((x, grid), dim=-1)
        x = self.fc0(x)
        x = x.permute(0, 4, 1, 2, 3)
        x = F.pad(x, [0,self.padding]) # pad the domain if input is non-periodic

        x1 = self.conv0(x)
        x2 = self.w0(x)
        x = x1 + x2
        x = F.gelu(x)
        x = self.dropout(x)

        x1 = self.conv1(x)
        x2 = self.w1(x)
        x = x1 + x2
        x = F.gelu(x)
        x = self.dropout(x)

        x1 = self.conv2(x)
        x2 = self.w2(x)
        x = x1 + x2
        x = F.gelu(x)
        x = self.dropout(x)

        x1 = self.conv3(x)
        x2 = self.w3(x)
        x = x1 + x2

        x = x[..., :-self.padding]
        x = x.permute(0, 2, 3, 4, 1) # pad the domain if input is non-periodic
        x = self.fc1(x)
        x = F.gelu(x)
        x = self.fc2(x)
        return x

    def get_grid(self, shape, device):
        batchsize, size_x, size_y, size_z = shape[0], shape[1], shape[2], shape[3]
        gridx = torch.tensor(np.linspace(0, 1, size_x), dtype=torch.float)
        gridx = gridx.reshape(1, size_x, 1, 1, 1).repeat([batchsize, 1, size_y, size_z, 1])
        gridy = torch.tensor(np.linspace(0, 1, size_y), dtype=torch.float)
        gridy = gridy.reshape(1, 1, size_y, 1, 1).repeat([batchsize, size_x, 1, size_z, 1])
        gridz = torch.tensor(np.linspace(0, 1, size_z), dtype=torch.float)
        gridz = gridz.reshape(1, 1, 1, size_z, 1).repeat([batchsize, size_x, size_y, 1, 1])
        return torch.cat((gridx, gridy, gridz), dim=-1).to(device)
       

def gen_axis_angle():
    axis_rand = np.random.randint(0,3)
    angle_rand = np.random.randint(0,4)

    if axis_rand==0:
        axis = (1,0)
    elif axis_rand==1:
        axis = (2,0)
    elif axis_rand==2:
        axis = (2,1)

    if angle_rand==0:
        angle = 0
    elif angle_rand==1:
        angle = 90
    elif angle_rand==2:
        angle = 180
    elif angle_rand==3:
        angle = 270
    return axis, angle

class ShapeNet_FNO_DataLoader(object):
    def __init__(self, root_path, mode='train', batch_size = 20, noise = False, augment=False):
        self.root_path = root_path
        self.mode = 'train'
        self.batch_size = batch_size
        self.noise = noise
        self.augment = augment
        if mode == 'train':
            self.data_path = self.root_path+'/Shapenet_FNO_train/'
            self.files_name = np.array(np.array([x for x in 
                                                 os.walk(self.data_path)])[0][2])
        else:
            self.data_path = self.root_path+'/Shapenet_FNO_test/'
            self.files_name = np.array(np.array([x for x in 
                                                 os.walk(self.data_path)])[0][2])
    def generate_data(self):
        voxels_batch = []
        divergence_tensor_batch = []
        for i in range(self.batch_size):
            j = np.random.randint(len(self.files_name))
            file_name = self.files_name[j]
            voxels_divergence = np.load(self.data_path+file_name)
            voxels = voxels_divergence['voxels']
            divergence_tensor = voxels_divergence['divergence_tensor']
            if self.noise:
                divergence_tensor+= np.random.rand(divergence_tensor.shape[0],
                                                   divergence_tensor.shape[1],
                                                   divergence_tensor.shape[2])*np.mean(np.abs(divergence_tensor))*0.05
            if self.augment:
                # Generate axis and angle for random rotation
                axis, angle = gen_axis_angle()
                voxels = rotate(voxels, angle, axis, reshape=False)
                divergence_tensor = rotate(divergence_tensor, angle, axis, reshape=False)
            voxels_batch.append(voxels)
            divergence_tensor_batch.append(divergence_tensor)
        return np.array(voxels_batch), np.array(divergence_tensor_batch)
    
class ShapeNet_FNO_guassmall_DataLoader(object):
    def __init__(self, root_path, mode='train', batch_size = 20, sigma=2, noise = False, augment=False):
        self.root_path = root_path
        self.mode = 'train'
        self.batch_size = batch_size
        self.noise = noise
        self.augment = augment
        if mode == 'train':
            self.data_path = self.root_path+'/Shapenet_FNO_gaussmall_train-'+str(sigma)+'/'
            self.files_name = np.array(np.array([x for x in 
                                                 os.walk(self.data_path)])[0][2])
        else:
            self.data_path = self.root_path+'/Shapenet_FNO_gaussmall_test-'+str(sigma)+'/'
            self.files_name = np.array(np.array([x for x in 
                                                 os.walk(self.data_path)])[0][2])
    def generate_data(self):
        voxels_batch = []
        divergence_tensor_batch = []
        for i in range(self.batch_size):
            j = np.random.randint(len(self.files_name))
            file_name = self.files_name[j]
            voxels_divergence = np.load(self.data_path+file_name)
            voxels = voxels_divergence['voxels']
            divergence_tensor = voxels_divergence['divergence_tensor']
            if self.noise:
                divergence_tensor+= np.random.rand(divergence_tensor.shape[0],
                                                   divergence_tensor.shape[1],
                                                   divergence_tensor.shape[2])*np.mean(np.abs(divergence_tensor))*0.05
            if self.augment:
                # Generate axis and angle for random rotation
                axis, angle = gen_axis_angle()
                voxels = rotate(voxels, angle, axis, reshape=False)
                divergence_tensor = rotate(divergence_tensor, angle, axis, reshape=False)
            voxels_batch.append(voxels)
            divergence_tensor_batch.append(divergence_tensor)
        return np.array(voxels_batch), np.array(divergence_tensor_batch)
    
class ShapeNet_FNO_nonuniform_dense_DataLoader(object):
    def __init__(self, root_path, mode='train', batch_size = 20, n_samples=30000, sigma=0.8):
        self.root_path = root_path
        self.mode = 'train'
        self.batch_size = batch_size
        if mode == 'train':
            self.data_path = self.root_path+'/Shapenet_FNO_nonuniform_nsamples_'+str(n_samples)+'_sigma_'+str(sigma)+'_train/'
            self.files_name = np.array(np.array([x for x in 
                                                 os.walk(self.data_path)])[0][2])
        else:
            self.data_path = self.root_path+'/Shapenet_FNO_nonuniform_nsamples_'+str(n_samples)+'_sigma_'+str(sigma)+'_test/'
            self.files_name = np.array(np.array([x for x in 
                                                 os.walk(self.data_path)])[0][2])
    def generate_data(self):
        voxels_batch = []
        divergence_tensor_batch = []
        for i in range(self.batch_size):
            j = np.random.randint(len(self.files_name))
            file_name = self.files_name[j]
            voxels_divergence = np.load(self.data_path+file_name)
            voxels = voxels_divergence['voxels']
            divergence_tensor = voxels_divergence['divergence_tensor']
            voxels_batch.append(voxels)
            divergence_tensor_batch.append(divergence_tensor)
        return np.array(voxels_batch), np.array(divergence_tensor_batch)
    
class ShapeNet_FNO_nonuniform_DataLoader(object):
    def __init__(self, root_path, mode='train', batch_size = 20):
        self.root_path = root_path
        self.mode = 'train'
        self.batch_size = batch_size
        if mode == 'train':
            self.data_path = self.root_path+'/Shapenet_FNO_nonuniform_train/'
            self.files_name = np.array(np.array([x for x in 
                                                 os.walk(self.data_path)])[0][2])
        else:
            self.data_path = self.root_path+'/Shapenet_FNO_nonuniform_test/'
            self.files_name = np.array(np.array([x for x in 
                                                 os.walk(self.data_path)])[0][2])
    def generate_data(self):
        voxels_batch = []
        divergence_tensor_batch = []
        for i in range(self.batch_size):
            j = np.random.randint(len(self.files_name))
            file_name = self.files_name[j]
            voxels_divergence = np.load(self.data_path+file_name)
            voxels = voxels_divergence['voxels']
            divergence_tensor = voxels_divergence['divergence_tensor']
            voxels_batch.append(voxels)
            divergence_tensor_batch.append(divergence_tensor)
        return np.array(voxels_batch), np.array(divergence_tensor_batch)

class ShapeNet_FNO_inward_DataLoader(object):
    def __init__(self, root_path, mode='train', batch_size = 20, noise = False, augment=False):
        self.root_path = root_path
        self.mode = 'train'
        self.batch_size = batch_size
        self.noise = noise
        self.augment = augment
        if mode == 'train':
            self.data_path = self.root_path+'/Shapenet_FNO_inward_train/'
            self.files_name = np.array(np.array([x for x in 
                                                 os.walk(self.data_path)])[0][2])
        else:
            self.data_path = self.root_path+'/Shapenet_FNO_inward_test/'
            self.files_name = np.array(np.array([x for x in 
                                                 os.walk(self.data_path)])[0][2])
    def generate_data(self):
        voxels_batch = []
        divergence_tensor_batch = []
        for i in range(self.batch_size):
            j = np.random.randint(len(self.files_name))
            file_name = self.files_name[j]
            voxels_divergence = np.load(self.data_path+file_name)
            voxels = voxels_divergence['voxels']
            divergence_tensor = voxels_divergence['divergence_tensor']
            if self.noise:
                divergence_tensor+= np.random.rand(divergence_tensor.shape[0],
                                                   divergence_tensor.shape[1],
                                                   divergence_tensor.shape[2])*np.mean(np.abs(divergence_tensor))*0.05
            if self.augment:
                # Generate axis and angle for random rotation
                axis, angle = gen_axis_angle()
                voxels = rotate(voxels, angle, axis, reshape=False)
                divergence_tensor = rotate(divergence_tensor, angle, axis, reshape=False)
            voxels_batch.append(voxels)
            divergence_tensor_batch.append(divergence_tensor)
        return np.array(voxels_batch), np.array(divergence_tensor_batch)
    
def normal2divergence(points, normals):
    grid_size = 64;
    data = np.round(points)

    field_V = np.zeros((grid_size, grid_size, grid_size, 3)) #Divergence of \overrightarrow{V}
    density_tensor_binary = np.zeros((grid_size, grid_size, grid_size)) # Density of points, so we can normalize
    for i in range(data.shape[0]):
        density_tensor_binary[round(data[i,0]), round(data[i,1]), round(data[i,2])] += 1

    density_tensor = scipy.ndimage.filters.gaussian_filter(np.pad(density_tensor_binary,((5,), (5,), (5,))), 2) #Smoothed density
    for i in range(data.shape[0]):
        field_V[round(data[i,0]), round(data[i,1]), round(data[i,2])] += normals[i] / density_tensor[round(data[i,0]) + 5, round(data[i,1]) + 5, round(data[i,2]) + 5]

    divergence_tensor = np.zeros(field_V.shape[:3])
    for i in range(3):
        divergence_tensor += np.roll(field_V, 1, axis = i)[:,:,:,i] - field_V[:,:,:,i]
    divergence_tensor = scipy.ndimage.filters.gaussian_filter(divergence_tensor,2)
    return divergence_tensor

class ShapeNet_FNO_no_smooth_DataLoader(object):
    def __init__(self, root_path, mode='train', batch_size = 20, noise = False):
        self.root_path = root_path
        self.mode = 'train'
        self.batch_size = batch_size
        self.noise = noise
        if mode == 'train':
            self.data_path = self.root_path+'/Shapenet_FNO_no_smooth_train/'
            self.files_name = np.array(np.array([x for x in 
                                                 os.walk(self.data_path)])[0][2])
        else:
            self.data_path = self.root_path+'/Shapenet_FNO_no_smooth_test/'
            self.files_name = np.array(np.array([x for x in 
                                                 os.walk(self.data_path)])[0][2])
    def generate_data(self):
        voxels_batch = []
        divergence_tensor_batch = []
        for i in range(self.batch_size):
            j = np.random.randint(len(self.files_name))
            file_name = self.files_name[j]
            voxels_divergence = np.load(self.data_path+file_name)
            voxels = voxels_divergence['voxels']
            divergence_tensor = voxels_divergence['divergence_tensor']
            if self.noise:
                divergence_tensor+= np.random.rand(divergence_tensor.shape[0],
                                                   divergence_tensor.shape[1],
                                                   divergence_tensor.shape[2])*np.mean(np.abs(divergence_tensor))*0.05
            voxels_batch.append(voxels)
            divergence_tensor_batch.append(divergence_tensor)
        return np.array(voxels_batch), np.array(divergence_tensor_batch)
    
class ShapeNet_FNO_density2divergence_DataLoader(object):
    def __init__(self, root_path, mode='train', batch_size = 20, noise = False):
        self.root_path = root_path
        self.mode = 'train'
        self.batch_size = batch_size
        self.noise = noise
        if mode == 'train':
            self.data_path = self.root_path+'/Shapenet_FNO_density2divergence_train/'
            self.files_name = np.array(np.array([x for x in 
                                                 os.walk(self.data_path)])[0][2])
        else:
            self.data_path = self.root_path+'/Shapenet_FNO_density2divergence_test/'
            self.files_name = np.array(np.array([x for x in 
                                                 os.walk(self.data_path)])[0][2])
    def generate_data(self):
        density_tensor_batch = []
        divergence_tensor_batch = []
        for i in range(self.batch_size):
            j = np.random.randint(len(self.files_name))
            file_name = self.files_name[j]
            density_divergence = np.load(self.data_path+file_name)
            density_tensor = density_divergence['density_tensor']
            divergence_tensor = density_divergence['divergence_tensor']
            if self.noise:
                density_tensor+= np.random.rand(density_tensor.shape[0],
                                                   density_tensor.shape[1],
                                                   density_tensor.shape[2])*np.mean(np.abs(density_tensor))*0.05
            density_tensor_batch.append(density_tensor)
            divergence_tensor_batch.append(divergence_tensor)
        return np.array(divergence_tensor_batch), np.array(density_tensor_batch)

def generate_voxel_divergence(path, subsampled = True):
    with open(path, 'rb') as f:
        voxels = (binvox_rw.read_as_3d_array(f).data.astype(int))
        if subsampled:
            voxels = voxels[::2,::2,::2]

    verts, faces, normals, values = measure.marching_cubes(voxels, 0)

    if subsampled:
        n_sample = 5000
    else:
        n_sample = 10000
    dense_indices = list([i for i in range(len(np.asarray(verts)))])
    sparse_indices = np.array(random.choices(dense_indices,k = n_sample))

    sparse_points = verts[sparse_indices]
    sparse_normals = normals[sparse_indices]

    grid_size = voxels.shape[0];
    data = np.round(sparse_points)

    field_V = np.zeros((grid_size, grid_size, grid_size, 3)) #Divergence of \overrightarrow{V}
    density_tensor_binary = np.zeros((grid_size, grid_size, grid_size)) # Density of points, so we can normalize
    for i in range(data.shape[0]):
        density_tensor_binary[round(data[i,0]), round(data[i,1]), round(data[i,2])] += 1

    density_tensor = scipy.ndimage.filters.gaussian_filter(np.pad(density_tensor_binary,((5,), (5,), (5,))), 2) #Smoothed density
    for i in range(data.shape[0]):
        field_V[round(data[i,0]), round(data[i,1]), round(data[i,2])] += sparse_normals[i] / density_tensor[round(data[i,0]) + 5, round(data[i,1]) + 5, round(data[i,2]) + 5]

    divergence_tensor = np.zeros(field_V.shape[:3])
    for i in range(3):
        divergence_tensor += np.roll(field_V, 1, axis = i)[:,:,:,i] - field_V[:,:,:,i]
    divergence_tensor = scipy.ndimage.filters.gaussian_filter(divergence_tensor,2)
    return voxels, divergence_tensor, verts, sparse_points, sparse_normals, path

def gen_data_train(data_path, raw_data_path, batch_size):
    files_data_name = np.array(np.array([x for x in os.walk(data_path)])[0][2])
    divergence_tensor_batch = []
    divergence_tensor_noise_batch = []
    voxels_batch = []
    voxels_binseg_batch = []
    sparse_points_batch = []
    sparse_normals_batch = []
    paths = []
    for i in range(batch_size):
        j = np.random.randint(len(files_data_name))
        file_data_name = files_data_name[j]
        file_raw_data_name = file_data_name.split('.')[0]+'.binvox'
        
        voxels, divergence_tensor, _, sparse_points, sparse_normals, path = generate_voxel_divergence(raw_data_path+
                                                                                    file_raw_data_name)
        
        voxels_binseg = np.zeros([2]+list(voxels.shape))
        voxels_binseg[0] = (voxels==0).astype(int)
        voxels_binseg[1] = (voxels!=0).astype(int)
        divergence_tensor_noise = divergence_tensor+np.random.rand(
            divergence_tensor.shape[0], 
            divergence_tensor.shape[1], 
            divergence_tensor.shape[2])*np.mean(np.abs(divergence_tensor))*0.05
        voxels_batch.append(voxels)
        voxels_binseg_batch.append(voxels_binseg)
        divergence_tensor_batch.append(divergence_tensor)
        divergence_tensor_noise_batch.append(divergence_tensor_noise)
        sparse_points_batch.append(sparse_points)
        sparse_normals_batch.append(sparse_normals)
        paths.append(path)
    return np.array(divergence_tensor_batch), np.array(divergence_tensor_noise_batch), np.array(voxels_batch), np.array(voxels_binseg_batch), np.array(sparse_points_batch), np.array(sparse_normals_batch), paths


def compute_divergence_tensor(points, normals, grid_size = 5000 , sigma = 2):
    data = np.round(points)

    field_V = np.zeros((grid_size, grid_size, grid_size, 3)) #Divergence of \overrightarrow{V}
    density_tensor_binary = np.zeros((grid_size, grid_size, grid_size)) # Density of points, so we can normalize
    for i in range(data.shape[0]):
        density_tensor_binary[round(data[i,0]), round(data[i,1]), round(data[i,2])] += 1

    density_tensor = scipy.ndimage.filters.gaussian_filter(np.pad(density_tensor_binary,((5,), (5,), (5,))), sigma) #Smoothed density
    for i in range(data.shape[0]):
        field_V[round(data[i,0]), round(data[i,1]), round(data[i,2])] += normals[i] / density_tensor[round(data[i,0]) + 5, round(data[i,1]) + 5, round(data[i,2]) + 5]

    divergence_tensor = np.zeros(field_V.shape[:3])
    for i in range(3):
        divergence_tensor += np.roll(field_V, 1, axis = i)[:,:,:,i] - field_V[:,:,:,i]
    divergence_tensor = scipy.ndimage.filters.gaussian_filter(divergence_tensor,sigma)
    return divergence_tensor

def gen_data_uniform_realistic(data_path, raw_data_path, batch_size):
    files_data_name = np.array(np.array([x for x in os.walk(data_path)])[0][2])
    divergence_tensor_uniform_batch = []
    divergence_tensor_realistic_batch = []
    voxels_batch = []
    points_uniform_batch = []
    normals_uniform_batch = []
    points_realistic_batch = []
    normals_realistic_batch = []
    paths = []
    for i in range(batch_size):
        j = np.random.randint(len(files_data_name))
        file_data_name = files_data_name[j]
        file_raw_data_name = file_data_name.split('.')[0]+'.binvox'
        path = raw_data_path+file_raw_data_name
        with open(path, 'rb') as f:
            voxels = (binvox_rw.read_as_3d_array(f).data.astype(int))
            voxels = voxels[::2,::2,::2]

        verts, faces, normals, values = measure.marching_cubes(voxels, 0)
        n_samples = 5000
        
        ## Computing uniform sparse sampling
        dense_indices = list([i for i in range(len(np.asarray(verts)))])
        sparse_indices = np.array(random.choices(dense_indices,k = n_samples))

        points_uniform = verts[sparse_indices]
        normals_uniform = normals[sparse_indices]
        
        ## Computing realistic sparse sampling
        mesh = trimesh.base.Trimesh(verts, faces, None, normals)
        points_realistic, idxs = mesh.sample(n_samples, return_index = True)

        # compute the barycentric coordinates of each sample
        bary = trimesh.triangles.points_to_barycentric(triangles=mesh.triangles[idxs], points=points_realistic)

        # interpolate vertex normals from barycentric coordinates
        normals_realistic = trimesh.unitize((mesh.vertex_normals[mesh.faces[idxs]] 
                                  *trimesh.unitize(bary).reshape((-1, 3, 1))).sum(axis=1))
        
        # Compute the associated divergence tensor
        grid_size = voxels.shape[0]
        divergence_tensor_uniform = compute_divergence_tensor(points_uniform, normals_uniform, grid_size)
        divergence_tensor_realistic = compute_divergence_tensor(points_realistic, normals_realistic, grid_size)
        
        divergence_tensor_uniform_batch.append(divergence_tensor_uniform)
        divergence_tensor_realistic_batch.append(divergence_tensor_realistic)
        voxels_batch.append(voxels)
        points_uniform_batch.append(points_uniform)
        normals_uniform_batch.append(normals_uniform)
        points_realistic_batch.append(points_realistic)
        normals_realistic_batch.append(normals_realistic)
        paths.append(path)
    return (np.array(divergence_tensor_uniform_batch), np.array(divergence_tensor_realistic_batch),
            np.array(voxels_batch), np.array(points_uniform_batch), np.array(normals_uniform_batch),
            np.array(points_realistic_batch), np.array(normals_realistic_batch), paths)

def gen_data_sparse_dense(data_path, raw_data_path, batch_size, n_samples_sparse, 
                          n_samples_dense, sigma_sparse, sigma_dense):
    files_data_name = np.array(np.array([x for x in os.walk(data_path)])[0][2])
    divergence_tensor_sparse_batch = []
    divergence_tensor_dense_batch = []
    voxels_batch = []
    points_sparse_batch = []
    normals_sparse_batch = []
    points_dense_batch = []
    normals_dense_batch = []
    paths = []
    for i in range(batch_size):
        j = np.random.randint(len(files_data_name))
        file_data_name = files_data_name[j]
        file_raw_data_name = file_data_name.split('.')[0]+'.binvox'
        path = raw_data_path+file_raw_data_name
        with open(path, 'rb') as f:
            voxels = (binvox_rw.read_as_3d_array(f).data.astype(int))
            voxels = voxels[::2,::2,::2]
        
        # Generate mesh 
        verts, faces, normals, values = measure.marching_cubes(voxels, 0)
        mesh = trimesh.base.Trimesh(verts, faces, None, normals)
        
        # Generating the sparse sampling
        points_sparse, idxs_sparse = mesh.sample(n_samples_sparse, return_index = True)
        
        bary_sparse = trimesh.triangles.points_to_barycentric(triangles=mesh.triangles[idxs_sparse], points=points_sparse)
        normals_sparse = trimesh.unitize((mesh.vertex_normals[mesh.faces[idxs_sparse]] 
                                  *trimesh.unitize(bary_sparse).reshape((-1, 3, 1))).sum(axis=1))
        
        # Generating the dense sampling
        points_dense, idxs_dense = mesh.sample(n_samples_dense, return_index = True)
        
        bary_dense = trimesh.triangles.points_to_barycentric(triangles=mesh.triangles[idxs_dense], 
                                                             points=points_dense)
        normals_dense = trimesh.unitize((mesh.vertex_normals[mesh.faces[idxs_dense]] 
                                  *trimesh.unitize(bary_dense).reshape((-1, 3, 1))).sum(axis=1))
    
        
        # Compute the associated divergence tensor
        grid_size = voxels.shape[0]
        divergence_tensor_sparse = compute_divergence_tensor(points_sparse, normals_sparse, grid_size, sigma_sparse)
        divergence_tensor_dense = compute_divergence_tensor(points_dense, normals_dense, grid_size, sigma_dense)
        
        divergence_tensor_sparse_batch.append(divergence_tensor_sparse)
        divergence_tensor_dense_batch.append(divergence_tensor_dense)
        voxels_batch.append(voxels)
        points_sparse_batch.append(points_sparse)
        normals_sparse_batch.append(normals_sparse)
        points_dense_batch.append(points_dense)
        normals_dense_batch.append(normals_dense)
        paths.append(path)
    return (np.array(divergence_tensor_sparse_batch), np.array(divergence_tensor_dense_batch),
            np.array(voxels_batch), np.array(points_sparse_batch), np.array(normals_sparse_batch),
            np.array(points_dense_batch), np.array(normals_dense_batch), paths)

def FNO_predict_no_vox(divergence_tensor_batch, model, batch_size):
    # To torch tensor and reshape
    divergence_tensor_batch = torch.Tensor(divergence_tensor_batch).cuda()
    
    div_normalizer = UnitGaussianNormalizer(divergence_tensor_batch)
    divergence_tensor_batch = div_normalizer.encode(divergence_tensor_batch)
    
    T = T_in = S = divergence_tensor_batch.shape[1]
    divergence_tensor_batch_resh = divergence_tensor_batch.reshape(batch_size,S,S,1,T_in).repeat([1,1,1,T,1])
    
    pred = model(divergence_tensor_batch_resh).view(batch_size, S, S, T)
    gc.collect()
    torch.cuda.empty_cache()
    
    pred = div_normalizer.decode(pred)
    gc.collect()
    torch.cuda.empty_cache()

    divergence_tensor_batch_cpu = divergence_tensor_batch.cpu().numpy()
    pred_cpu = pred.detach().cpu().numpy()
    
    del divergence_tensor_batch
    del pred
    gc.collect()
    torch.cuda.empty_cache()
    
    return pred_cpu, divergence_tensor_batch_cpu

def FNO_predict_vox(divergence_tensor_batch, voxels_batch, model, batch_size):
    # To torch tensor and reshape
    divergence_tensor_batch = torch.Tensor(divergence_tensor_batch).cuda()
    voxels_batch = torch.Tensor(voxels_batch).cuda()
    
    div_normalizer = UnitGaussianNormalizer(divergence_tensor_batch)
    divergence_tensor_batch = div_normalizer.encode(divergence_tensor_batch)
    
    vox_normalizer = UnitGaussianNormalizer(voxels_batch)
    voxels_batch = vox_normalizer.encode(voxels_batch)
    
    T = T_in = S = divergence_tensor_batch.shape[1]
    divergence_tensor_batch_resh = divergence_tensor_batch.reshape(batch_size,S,S,1,T_in).repeat([1,1,1,T,1])
    
    pred = model(divergence_tensor_batch_resh).view(batch_size, S, S, T)
    gc.collect()
    torch.cuda.empty_cache()
    
    voxels_batch = vox_normalizer.decode(voxels_batch)
    pred = vox_normalizer.decode(pred)
    gc.collect()
    torch.cuda.empty_cache()
    
    divergence_tensor_batch_cpu = divergence_tensor_batch.cpu().numpy()
    voxels_batch_cpu = voxels_batch.cpu().numpy()
    pred_cpu = pred.detach().cpu().numpy()
    
    del divergence_tensor_batch_resh
    del divergence_tensor_batch
    del pred
    del voxels_batch
    gc.collect()
    torch.cuda.empty_cache()
    
    return pred_cpu, divergence_tensor_batch_cpu, voxels_batch_cpu

def FNO_predict(divergence_tensor_batch,  model, batch_size):
    # To torch tensor and reshape
    divergence_tensor_batch = torch.Tensor(divergence_tensor_batch).cuda()
    
    div_normalizer = UnitGaussianNormalizer(divergence_tensor_batch)
    divergence_tensor_batch = div_normalizer.encode(divergence_tensor_batch)
    
    T = T_in = S = divergence_tensor_batch.shape[1]
    divergence_tensor_batch_resh = divergence_tensor_batch.reshape(batch_size,S,S,1,T_in).repeat([1,1,1,T,1])
    
    pred = model(divergence_tensor_batch_resh).view(batch_size, S, S, T)
    gc.collect()
    torch.cuda.empty_cache()
    
    divergence_tensor_batch_cpu = divergence_tensor_batch.cpu().numpy()
    pred_cpu = pred.detach().cpu().numpy()
    
    del divergence_tensor_batch_resh
    del divergence_tensor_batch
    del pred
    gc.collect()
    torch.cuda.empty_cache()
    
    return pred_cpu, divergence_tensor_batch_cpu

def FNO_predict_l2_div(divergence_tensor_batch,  model, batch_size):
    # To torch tensor and reshape    
    divergence_tensor_batch = torch.Tensor(divergence_tensor_batch).cuda()
    
    div_normalizer = UnitGaussianNormalizer(divergence_tensor_batch)
    divergence_tensor_batch = div_normalizer.encode(divergence_tensor_batch)
    
    T = T_in = S = divergence_tensor_batch.shape[1]
    divergence_tensor_batch_resh = divergence_tensor_batch.reshape(batch_size,S,S,1,T_in).repeat([1,1,1,T,1])
    
    pred = model(divergence_tensor_batch_resh).view(batch_size, S, S, T)
    gc.collect()
    torch.cuda.empty_cache()
    
    pred = div_normalizer.decode(pred)
    gc.collect()
    torch.cuda.empty_cache()
    
    divergence_tensor_batch_cpu = divergence_tensor_batch.cpu().numpy()
    pred_cpu = F.sigmoid(pred).detach().cpu().numpy()
    
    del divergence_tensor_batch_resh
    del divergence_tensor_batch
    del div_normalizer
    del pred
    gc.collect()
    torch.cuda.empty_cache()
    
    return pred_cpu, divergence_tensor_batch_cpu

def FNO_predict_l2_no_div(divergence_tensor_batch,  model, batch_size):
    # To torch tensor and reshape    
    divergence_tensor_batch = torch.Tensor(divergence_tensor_batch).cuda()
    
    T = T_in = S = divergence_tensor_batch.shape[1]
    divergence_tensor_batch_resh = divergence_tensor_batch.reshape(batch_size,S,S,1,T_in).repeat([1,1,1,T,1])
    
    pred = model(divergence_tensor_batch_resh).view(batch_size, S, S, T)
    gc.collect()
    torch.cuda.empty_cache()
    
    divergence_tensor_batch_cpu = divergence_tensor_batch.cpu().numpy()
    pred_cpu = F.sigmoid(pred).detach().cpu().numpy()
    
    del divergence_tensor_batch_resh
    del divergence_tensor_batch
    del pred
    gc.collect()
    torch.cuda.empty_cache()
    
    return pred_cpu, divergence_tensor_batch_cpu

def UFNO_predict(divergence_tensor_batch,  model, batch_size, T):
    # To torch tensor and reshape    
    divergence_tensor_batch = torch.Tensor(divergence_tensor_batch).cuda()
    
    div_normalizer = UnitGaussianNormalizer(divergence_tensor_batch)
    divergence_tensor_batch = div_normalizer.encode(divergence_tensor_batch)
    
    divergence_tensor_batch_resh = divergence_tensor_batch[:,:,:,:,np.newaxis].repeat([1,1,1,1,T])
    
    pred = model(divergence_tensor_batch_resh)
    gc.collect()
    torch.cuda.empty_cache()
    
    pred = div_normalizer.decode(pred)
    gc.collect()
    torch.cuda.empty_cache()
    
    divergence_tensor_batch_cpu = divergence_tensor_batch.cpu().numpy()
    pred_cpu = F.sigmoid(pred).detach().cpu().numpy()
    
    del divergence_tensor_batch_resh
    del divergence_tensor_batch
    del div_normalizer
    del pred
    gc.collect()
    torch.cuda.empty_cache()
    
    return pred_cpu, divergence_tensor_batch_cpu

def FNO_predict_binseg_div(divergence_tensor_batch,  model, batch_size):
    # To torch tensor and reshape    
    divergence_tensor_batch_norm = np.repeat(divergence_tensor_batch[:,np.newaxis,:,:,:],2,axis=1)
    divergence_tensor_batch_norm[:,0,:,:,:] = 1-divergence_tensor_batch
    
    divergence_tensor_batch = torch.Tensor(divergence_tensor_batch).cuda()
    
    div_normalizer = UnitGaussianNormalizer(divergence_tensor_batch)
    divergence_tensor_batch = div_normalizer.encode(divergence_tensor_batch)
   
    T = T_in = S = divergence_tensor_batch.shape[1]
    divergence_tensor_batch_resh = divergence_tensor_batch.reshape(batch_size,S,S,1,T_in).repeat([1,1,1,T,1])
    
    pred = model(divergence_tensor_batch_resh).view(batch_size, S, S, T,2).permute(0,4,1,2,3)
    gc.collect()
    torch.cuda.empty_cache()
    
    pred_normalizer = UnitGaussianNormalizer(torch.Tensor(divergence_tensor_batch_norm).cuda())
    pred = pred_normalizer.decode(pred)
    gc.collect()
    torch.cuda.empty_cache()
    
    divergence_tensor_batch_cpu = divergence_tensor_batch.cpu().numpy()
    pred_cpu = F.sigmoid(pred).detach().cpu().numpy()
    
    del divergence_tensor_batch_resh
    del divergence_tensor_batch
    del pred
    del div_normalizer
    del pred_normalizer
    gc.collect()
    torch.cuda.empty_cache()
    
    return pred_cpu, divergence_tensor_batch_cpu

def FNO_predict_l2_div_norepdim(divergence_tensor_batch,  model, batch_size):
    # To torch tensor and reshape    
    divergence_tensor_batch = torch.Tensor(divergence_tensor_batch).cuda()
    
    div_normalizer = UnitGaussianNormalizer(divergence_tensor_batch)
    divergence_tensor_batch = div_normalizer.encode(divergence_tensor_batch)
        
    pred = model(divergence_tensor_batch)
    gc.collect()
    torch.cuda.empty_cache()
    
    pred = div_normalizer.decode(pred)
    gc.collect()
    torch.cuda.empty_cache()
    
    divergence_tensor_batch_cpu = divergence_tensor_batch.cpu().numpy()
    pred_cpu = F.sigmoid(pred).detach().cpu().numpy()
    
    del divergence_tensor_batch
    del pred
    del div_normalizer
    gc.collect()
    torch.cuda.empty_cache()
    
    return pred_cpu, divergence_tensor_batch_cpu

def FNO_predict_binseg_div_norepdim(divergence_tensor_batch,  model, batch_size):
    # To torch tensor and reshape    
    divergence_tensor_batch_norm = np.repeat(divergence_tensor_batch[:,np.newaxis,:,:,:],2,axis=1)
    divergence_tensor_batch_norm[:,0,:,:,:] = 1-divergence_tensor_batch
    
    divergence_tensor_batch = torch.Tensor(divergence_tensor_batch).cuda()
    
    div_normalizer = UnitGaussianNormalizer(divergence_tensor_batch)
    divergence_tensor_batch = div_normalizer.encode(divergence_tensor_batch)
    
    pred = model(divergence_tensor_batch).permute(0,4,1,2,3)
    gc.collect()
    torch.cuda.empty_cache()
    
    pred_normalizer = UnitGaussianNormalizer(torch.Tensor(divergence_tensor_batch_norm).cuda())
    pred = pred_normalizer.decode(pred)
    gc.collect()
    torch.cuda.empty_cache()
    
    divergence_tensor_batch_cpu = divergence_tensor_batch.cpu().numpy()
    pred_cpu = F.sigmoid(pred).detach().cpu().numpy()
    
    del divergence_tensor_batch
    del pred
    del div_normalizer
    del pred_normalizer
    gc.collect()
    torch.cuda.empty_cache()
    
    return pred_cpu, divergence_tensor_batch_cpu


def FNO_predict_l2_no_norm(divergence_tensor_batch,  model, batch_size):
    # To torch tensor and reshape    
    divergence_tensor_batch = torch.Tensor(divergence_tensor_batch).cuda()
    
    T = T_in = S = divergence_tensor_batch.shape[1]
    divergence_tensor_batch_resh = divergence_tensor_batch.reshape(batch_size,S,S,1,T_in).repeat([1,1,1,T,1])
    
    pred = model(divergence_tensor_batch_resh).view(batch_size, S, S, T)
    gc.collect()
    torch.cuda.empty_cache()
    
    divergence_tensor_batch_cpu = divergence_tensor_batch.cpu().numpy()
    pred_cpu = F.sigmoid(pred).detach().cpu().numpy()
    
    del divergence_tensor_batch_resh
    del divergence_tensor_batch
    del pred
    gc.collect()
    torch.cuda.empty_cache()
    
    return pred_cpu, divergence_tensor_batch_cpu

def FNO_predict_binseg_no_norm(divergence_tensor_batch,  model, batch_size):
    # To torch tensor and reshape    
    divergence_tensor_batch = torch.Tensor(divergence_tensor_batch).cuda()
   
    T = T_in = S = divergence_tensor_batch.shape[1]
    divergence_tensor_batch_resh = divergence_tensor_batch.reshape(batch_size,S,S,1,T_in).repeat([1,1,1,T,1])
    
    pred = model(divergence_tensor_batch_resh).view(batch_size, S, S, T,2).permute(0,4,1,2,3)
    gc.collect()
    torch.cuda.empty_cache()
    
    divergence_tensor_batch_cpu = divergence_tensor_batch.cpu().numpy()
    pred_cpu = F.sigmoid(pred).detach().cpu().numpy()
    
    del divergence_tensor_batch_resh
    del divergence_tensor_batch
    del pred
    gc.collect()
    torch.cuda.empty_cache()
    
    return pred_cpu, divergence_tensor_batch_cpu

def FNO_binseg_predict_vox(divergence_tensor_batch,voxels_batch,  model_binseg, batch_size):
    # To torch tensor and reshape
    divergence_tensor_batch = torch.Tensor(divergence_tensor_batch).cuda()
    voxels_batch = torch.Tensor(voxels_batch).cuda()
    
    div_normalizer = UnitGaussianNormalizer(divergence_tensor_batch)
    divergence_tensor_batch = div_normalizer.encode(divergence_tensor_batch)
    
    vox_normalizer = UnitGaussianNormalizer(voxels_batch)
    voxels_batch = vox_normalizer.encode(voxels_batch)

    T = T_in = S = divergence_tensor_batch.shape[1]
    divergence_tensor_batch_resh = divergence_tensor_batch.reshape(batch_size,S,S,1,T_in).repeat([1,1,1,T,1])
    
    pred = model_binseg(divergence_tensor_batch_resh).view(batch_size, S, S, T,2).permute(0,4,1,2,3)
    gc.collect()
    torch.cuda.empty_cache()
    
    voxels_batch = vox_normalizer.decode(voxels_batch)
    pred = vox_normalizer.decode(pred)
    gc.collect()
    torch.cuda.empty_cache()
    
    divergence_tensor_batch_cpu = divergence_tensor_batch.cpu().numpy()
    pred_cpu = F.sigmoid(pred).detach().cpu().numpy()
    voxels_batch_cpu = voxels_batch.cpu().numpy()
    
    del divergence_tensor_batch_resh
    del divergence_tensor_batch
    del pred
    del voxels_batch
    gc.collect()
    torch.cuda.empty_cache()
    
    return pred_cpu, divergence_tensor_batch_cpu, voxels_batch_cpu

class Laplacian(odl.Operator):
    def __init__(self, space):
        super(Laplacian, self).__init__(
            domain=space, range=space, linear=True)

    def _call(self, x):
        return scipy.ndimage.laplace(x)

    @property
    def adjoint(self):
        return self

def classical_poisson_solver(divergence_tensor):
    space = odl.space.NumpyTensorSpace(divergence_tensor.shape, dtype = np.float)
    mask = space.element(divergence_tensor)
    mask_op = odl.operator.default_ops.MultiplyOperator(mask)
    A = Laplacian(space)
    lin_ops = [A, mask_op]
    a = 0.2
    b = 0.00005
    g = space.element(divergence_tensor)
    # Create functionals for the l2 distance and l1 norm.
    g_funcs = [odl.solvers.L2NormSquared(space).translated(g), a * odl.solvers.L2NormSquared(space)]
    f = odl.solvers.functional.default_functionals.ZeroFunctional(space)
    
    # Find scaling constants so that the solver converges.
    # See the douglas_rachford_pd documentation for more information.
    opnorm_A = odl.power_method_opnorm(A, xstart=g)
    #opnorm_B = odl.power_method_opnorm(B)
    opnorm_mask = 1
    sigma = [1 / opnorm_A ** 2, 1 / opnorm_mask ** 2]
    tau = 1.0
    
    # Solve using the Douglas-Rachford Primal-Dual method
    x = space.zero()
    odl.solvers.douglas_rachford_pd(x, f, g_funcs, lin_ops,
                                    tau=tau, sigma=sigma, niter=1000)
    
    pred_poisson_classic = x.asarray() >= 0
    return pred_poisson_classic

def write_binvox(voxel_model, fp):
    """ Write binary binvox format.

    Note that when saving a model in sparse (coordinate) format, it is first
    converted to dense format.

    Doesn't check if the model is 'sane'.

    """
    if voxel_model.data.ndim==2:
        # TODO avoid conversion to dense
        dense_voxel_data = sparse_to_dense(voxel_model.data, voxel_model.dims)
    else:
        dense_voxel_data = voxel_model.data

    fp.write('#binvox 1\n'.encode('ascii'))
    line = 'dim '+' '.join(map(str, voxel_model.dims))+'\n'
    fp.write(line.encode('ascii'))
    line = 'translate '+' '.join(map(str, voxel_model.translate))+'\n'
    fp.write(line.encode('ascii'))
    line = 'scale '+str(voxel_model.scale)+'\n'
    fp.write(line.encode('ascii'))
    fp.write('data\n'.encode('ascii'))
    if not voxel_model.axis_order in ('xzy', 'xyz'):
        raise ValueError('Unsupported voxel model axis order')

    if voxel_model.axis_order=='xzy':
        voxels_flat = dense_voxel_data.flatten()
    elif voxel_model.axis_order=='xyz':
        voxels_flat = np.transpose(dense_voxel_data, (0, 2, 1)).flatten()

    # keep a sort of state machine for writing run length encoding
    state = voxels_flat[0]
    ctr = 0
    for c in voxels_flat:
        if c==state:
            ctr += 1
            # if ctr hits max, dump
            if ctr==255:
                fp.write(state.tobytes())
                fp.write(ctr.to_bytes(1, byteorder='little'))
                ctr = 0
        else:
            # if switch state, dump
            if ctr > 0:
                fp.write(state.tobytes())
                fp.write(ctr.to_bytes(1, byteorder='little'))
            state = c
            ctr = 1
    # flush out remainders
    if ctr > 0:
        fp.write(state.tobytes())
        fp.write(ctr.to_bytes(1, byteorder='little'))
        
def save_points(points, n_sample, path,name):
    pcd = o3d.geometry.PointCloud()
    dense_indices = list([i for i in range(len(np.asarray(points)))])
    sparse_indices = np.array(random.choices(dense_indices,k = n_sample))
    pcd.points = o3d.utility.Vector3dVector(points[sparse_indices])
    o3d.io.write_point_cloud(path+'/'+name+'.ply', pcd)
        
def save_prediction(pred, n_sample, path,  name):
    # Saving as pointclouds using marching cubes
    verts, faces, normals, _ = measure.marching_cubes(pred, 0)

    dense_indices = list([i for i in range(len(np.asarray(verts)))])
    sparse_indices = np.array(random.choices(dense_indices,k = n_sample))

    sparse_points = verts[sparse_indices]

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(verts)
    o3d.io.write_point_cloud(path+'/'+name+'.ply', pcd)
    print("Pointcloud .ply saved...")

    # Saving binvox
    pred_bool = pred.astype(bool)
    voxel_model = binvox_rw.Voxels(pred_bool, list(pred_bool.shape), 
                              translate = [0.0,0.0,0.0], scale = 1.,
                                axis_order='xyz')


    fp = open(path+'/'+name+'.binvox', "wb")
    write_binvox(voxel_model, fp)
    fp.close
    print("Voxel array .binvox saved...")

    # Saving mesh as stl
    mesh = trimesh.base.Trimesh(verts, faces, None, normals)
    mesh.export(path+'/'+name+'.stl');
    print("Mesh .stl saved...")

    #Saving mesh as 
    vertices, triangles = mcubes.marching_cubes(pred, 0)
    mcubes.export_mesh(vertices, triangles, path+'/'+name+'.dae', name)
    print("Mesh .dae saved...")
    
    smoothed_pred = mcubes.smooth(pred)
    vertices, triangles = mcubes.marching_cubes(smoothed_pred, 0)
    mcubes.export_mesh(vertices, triangles, path+'/'+name+'_smoothed.dae', name+'_smoothed')
    print("Smoothed .dae saved...")
    
def gen_data_from_ply(pcd_path, batch_size, noise = False):
    pcd = o3d.io.read_point_cloud(pcd_path)
    pcd.normals = o3d.utility.Vector3dVector(np.zeros((1, 3)))
    pcd.estimate_normals()
    pcd.orient_normals_consistent_tangent_plane(100)

    points = np.asarray(pcd.points)
    normals = np.asarray(pcd.normals)

    divergence_tensor = normal2divergence(points, normals)
    
    if noise:
        divergence_tensor+= np.random.rand(divergence_tensor.shape[0], divergence_tensor.shape[1], 
                                           divergence_tensor.shape[2])*np.mean(np.abs(divergence_tensor))*0.05
    return np.repeat(divergence_tensor[np.newaxis], batch_size, 0)

class ShapeNet_FNO_binseg_DataLoader(object):
    def __init__(self, root_path, mode='train', batch_size = 20, noise = False):
        self.root_path = root_path
        self.mode = 'train'
        self.batch_size = batch_size
        self.noise = noise
        if mode == 'train':
            self.data_path = self.root_path+'/Shapenet_FNO_train/'
            self.files_name = np.array(np.array([x for x in 
                                                 os.walk(self.data_path)])[0][2])
        else:
            self.data_path = self.root_path+'/Shapenet_FNO_test/'
            self.files_name = np.array(np.array([x for x in 
                                                 os.walk(self.data_path)])[0][2])
    def generate_data(self):
        voxels_batch = []
        divergence_tensor_batch = []
        for i in range(self.batch_size):
            j = np.random.randint(len(self.files_name))
            file_name = self.files_name[j]
            voxels_divergence = np.load(self.data_path+file_name)
            voxels = voxels_divergence['voxels']
            voxels_semseg = np.zeros([2]+list(voxels.shape))
            voxels_semseg[0] = (voxels==0).astype(int)
            voxels_semseg[1] = (voxels!=0).astype(int)
            divergence_tensor = voxels_divergence['divergence_tensor']
            if self.noise:
                divergence_tensor+= np.random.rand(divergence_tensor.shape[0],
                                                   divergence_tensor.shape[1],
                                                   divergence_tensor.shape[2])*np.mean(np.abs(divergence_tensor))*0.05
            voxels_batch.append(voxels_semseg)
            divergence_tensor_batch.append(divergence_tensor)
        return np.array(voxels_batch), np.array(divergence_tensor_batch)
    
class ShapeNet_FNO_subsampled_DataLoader(object):
    def __init__(self, root_path, mode='train', batch_size = 20, noise=False):
        self.root_path = root_path
        self.mode = 'train'
        self.batch_size = batch_size
        self.noise = noise
        if mode == 'train':
            self.data_path = self.root_path+'/Shapenet_FNO_subsampled_train/'
            self.files_name = np.array(np.array([x for x in 
                                                 os.walk(self.data_path)])[0][2])
        else:
            self.data_path = self.root_path+'/Shapenet_FNO_subsampled_test/'
            self.files_name = np.array(np.array([x for x in 
                                                 os.walk(self.data_path)])[0][2])
    def generate_data(self):
        voxels_batch = []
        divergence_tensor_batch = []
        for i in range(self.batch_size):
            j = np.random.randint(len(self.files_name))
            file_name = self.files_name[j]
            voxels_divergence = np.load(self.data_path+file_name)
            voxels = voxels_divergence['voxels']
            divergence_tensor = voxels_divergence['divergence_tensor']
            if self.noise:
                divergence_tensor+= np.random.rand(divergence_tensor.shape[0],
                                                   divergence_tensor.shape[1],
                                                   divergence_tensor.shape[2])*np.mean(np.abs(divergence_tensor))*0.05
            voxels_batch.append(voxels)
            divergence_tensor_batch.append(divergence_tensor)
        return np.array(voxels_batch), np.array(divergence_tensor_batch)
    
class ShapeNet_FNO_no_normals_DataLoader(object):
    def __init__(self, root_path, mode='train', batch_size = 20, noise = False):
        self.root_path = root_path
        self.mode = 'train'
        self.batch_size = batch_size
        self.noise = noise
        if mode == 'train':
            self.data_path = self.root_path+'/Shapenet_FNO_no_normals_train/'
            self.files_name = np.array(np.array([x for x in 
                                                 os.walk(self.data_path)])[0][2])
        else:
            self.data_path = self.root_path+'/Shapenet_FNO_no_normals_test/'
            self.files_name = np.array(np.array([x for x in 
                                                 os.walk(self.data_path)])[0][2])
    def generate_data(self):
        voxels_batch = []
        density_tensor_batch = []
        for i in range(self.batch_size):
            j = np.random.randint(len(self.files_name))
            file_name = self.files_name[j]
            voxels_density = np.load(self.data_path+file_name)
            voxels = voxels_density['voxels']
            density_tensor = voxels_density['density_tensor']
            if self.noise:
                density_tensor+= np.random.rand(density_tensor.shape[0],
                                                   density_tensor.shape[1],
                                                   density_tensor.shape[2])*np.mean(np.abs(density_tensor))*0.05
            voxels_batch.append(voxels)
            density_tensor_batch.append(density_tensor)
        return np.array(voxels_batch), np.array(density_tensor_batch)
    
    
class FNO3d_binseg(nn.Module):
    def __init__(self, modes1, modes2, modes3, width):
        super(FNO3d_binseg, self).__init__()

        """
        The overall network. It contains 4 layers of the Fourier layer.
        1. Lift the input to the desire channel dimension by self.fc0 .
        2. 4 layers of the integral operators u' = (W + K)(u).
            W defined by self.w; K defined by self.conv .
        3. Project from the channel space to the output space by self.fc1 and self.fc2 .
        
        input: the solution of the first 10 timesteps + 3 locations (u(1, x, y), ..., u(10, x, y),  x, y, t). It's a constant function in time, except for the last index.
        input shape: (batchsize, x=64, y=64, t=40, c=13)
        output: the solution of the next 40 timesteps
        output shape: (batchsize, x=64, y=64, t=40, c=1)
        """

        self.modes1 = modes1
        self.modes2 = modes2
        self.modes3 = modes3
        self.width = width
        self.padding = 6 # pad the domain if input is non-periodic
        self.fc0 = nn.Linear(67, self.width)
        # input channel is 12: the solution of the first 10 timesteps + 3 locations (u(1, x, y), ..., u(10, x, y),  x, y, t)

        self.conv0 = SpectralConv3d(self.width, self.width, self.modes1, self.modes2, self.modes3)
        self.conv1 = SpectralConv3d(self.width, self.width, self.modes1, self.modes2, self.modes3)
        self.conv2 = SpectralConv3d(self.width, self.width, self.modes1, self.modes2, self.modes3)
        self.conv3 = SpectralConv3d(self.width, self.width, self.modes1, self.modes2, self.modes3)
        self.w0 = nn.Conv3d(self.width, self.width, 1)
        self.w1 = nn.Conv3d(self.width, self.width, 1)
        self.w2 = nn.Conv3d(self.width, self.width, 1)
        self.w3 = nn.Conv3d(self.width, self.width, 1)
        self.bn0 = torch.nn.BatchNorm3d(self.width)
        self.bn1 = torch.nn.BatchNorm3d(self.width)
        self.bn2 = torch.nn.BatchNorm3d(self.width)
        self.bn3 = torch.nn.BatchNorm3d(self.width)

        self.fc1 = nn.Linear(self.width, 128)
        self.fc2 = nn.Linear(128, 2)

    def forward(self, x):
        grid = self.get_grid(x.shape, device)
        x = torch.cat((x, grid), dim=-1)
        x = self.fc0(x)
        x = x.permute(0, 4, 1, 2, 3)
        x = F.pad(x, [0,self.padding]) # pad the domain if input is non-periodic

        x1 = self.conv0(x)
        x2 = self.w0(x)
        x = x1 + x2
        x = F.gelu(x)

        x1 = self.conv1(x)
        x2 = self.w1(x)
        x = x1 + x2
        x = F.gelu(x)

        x1 = self.conv2(x)
        x2 = self.w2(x)
        x = x1 + x2
        x = F.gelu(x)

        x1 = self.conv3(x)
        x2 = self.w3(x)
        x = x1 + x2

        x = x[..., :-self.padding]
        x = x.permute(0, 2, 3, 4, 1) # pad the domain if input is non-periodic
        x = self.fc1(x)
        x = F.gelu(x)
        x = self.fc2(x)
        return x

    def get_grid(self, shape, device):
        batchsize, size_x, size_y, size_z = shape[0], shape[1], shape[2], shape[3]
        gridx = torch.tensor(np.linspace(0, 1, size_x), dtype=torch.float)
        gridx = gridx.reshape(1, size_x, 1, 1, 1).repeat([batchsize, 1, size_y, size_z, 1])
        gridy = torch.tensor(np.linspace(0, 1, size_y), dtype=torch.float)
        gridy = gridy.reshape(1, 1, size_y, 1, 1).repeat([batchsize, size_x, 1, size_z, 1])
        gridz = torch.tensor(np.linspace(0, 1, size_z), dtype=torch.float)
        gridz = gridz.reshape(1, 1, 1, size_z, 1).repeat([batchsize, size_x, size_y, 1, 1])
        return torch.cat((gridx, gridy, gridz), dim=-1).to(device)

class FNO3d_norepdim(nn.Module):
    def __init__(self, modes1, modes2, modes3, width):
        super(FNO3d_norepdim, self).__init__()

        """
        The overall network. It contains 4 layers of the Fourier layer.
        1. Lift the input to the desire channel dimension by self.fc0 .
        2. 4 layers of the integral operators u' = (W + K)(u).
            W defined by self.w; K defined by self.conv .
        3. Project from the channel space to the output space by self.fc1 and self.fc2 .
        
        input: the solution of the first 10 timesteps + 3 locations (u(1, x, y), ..., u(10, x, y),  x, y, t). It's a constant function in time, except for the last index.
        input shape: (batchsize, x=64, y=64, t=40, c=13)
        output: the solution of the next 40 timesteps
        output shape: (batchsize, x=64, y=64, t=40, c=1)
        """

        self.modes1 = modes1
        self.modes2 = modes2
        self.modes3 = modes3
        self.width = width
        self.padding = 6 # pad the domain if input is non-periodic
        self.fc0 = nn.Linear(4, self.width)
        # input channel is 12: the solution of the first 10 timesteps + 3 locations (u(1, x, y), ..., u(10, x, y),  x, y, t)

        self.conv0 = SpectralConv3d(self.width, self.width, self.modes1, self.modes2, self.modes3)
        self.conv1 = SpectralConv3d(self.width, self.width, self.modes1, self.modes2, self.modes3)
        self.conv2 = SpectralConv3d(self.width, self.width, self.modes1, self.modes2, self.modes3)
        self.conv3 = SpectralConv3d(self.width, self.width, self.modes1, self.modes2, self.modes3)
        self.w0 = nn.Conv3d(self.width, self.width, 1)
        self.w1 = nn.Conv3d(self.width, self.width, 1)
        self.w2 = nn.Conv3d(self.width, self.width, 1)
        self.w3 = nn.Conv3d(self.width, self.width, 1)
        self.bn0 = torch.nn.BatchNorm3d(self.width)
        self.bn1 = torch.nn.BatchNorm3d(self.width)
        self.bn2 = torch.nn.BatchNorm3d(self.width)
        self.bn3 = torch.nn.BatchNorm3d(self.width)

        self.fc1 = nn.Linear(self.width, 128)
        self.fc2 = nn.Linear(128, 1)

    def forward(self, x):
        grid = self.get_grid(x.shape, device)
        x = torch.cat((x[:,:,:,:,np.newaxis], grid), dim=-1)
        x = self.fc0(x)
        x = x.permute(0, 4, 1, 2, 3)
        x = F.pad(x, [0,self.padding]) # pad the domain if input is non-periodic

        x1 = self.conv0(x)
        x2 = self.w0(x)
        x = x1 + x2
        x = F.gelu(x)

        x1 = self.conv1(x)
        x2 = self.w1(x)
        x = x1 + x2
        x = F.gelu(x)

        x1 = self.conv2(x)
        x2 = self.w2(x)
        x = x1 + x2
        x = F.gelu(x)

        x1 = self.conv3(x)
        x2 = self.w3(x)
        x = x1 + x2

        x = x[..., :-self.padding]
        x = x.permute(0, 2, 3, 4, 1) # pad the domain if input is non-periodic
        x = self.fc1(x)
        x = F.gelu(x)
        x = self.fc2(x)
        return x[:,:,:,:,0]

    def get_grid(self, shape, device):
        batchsize, size_x, size_y, size_z = shape[0], shape[1], shape[2], shape[3]
        gridx = torch.tensor(np.linspace(0, 1, size_x), dtype=torch.float)
        gridx = gridx.reshape(1, size_x, 1, 1, 1).repeat([batchsize, 1, size_y, size_z, 1])
        gridy = torch.tensor(np.linspace(0, 1, size_y), dtype=torch.float)
        gridy = gridy.reshape(1, 1, size_y, 1, 1).repeat([batchsize, size_x, 1, size_z, 1])
        gridz = torch.tensor(np.linspace(0, 1, size_z), dtype=torch.float)
        gridz = gridz.reshape(1, 1, 1, size_z, 1).repeat([batchsize, size_x, size_y, 1, 1])
        return torch.cat((gridx, gridy, gridz), dim=-1).to(device)
    
class FNO3d_binseg_norepdim(nn.Module):
    def __init__(self, modes1, modes2, modes3, width):
        super(FNO3d_binseg_norepdim, self).__init__()

        """
        The overall network. It contains 4 layers of the Fourier layer.
        1. Lift the input to the desire channel dimension by self.fc0 .
        2. 4 layers of the integral operators u' = (W + K)(u).
            W defined by self.w; K defined by self.conv .
        3. Project from the channel space to the output space by self.fc1 and self.fc2 .
        
        input: the solution of the first 10 timesteps + 3 locations (u(1, x, y), ..., u(10, x, y),  x, y, t). It's a constant function in time, except for the last index.
        input shape: (batchsize, x=64, y=64, t=40, c=13)
        output: the solution of the next 40 timesteps
        output shape: (batchsize, x=64, y=64, t=40, c=1)
        """

        self.modes1 = modes1
        self.modes2 = modes2
        self.modes3 = modes3
        self.width = width
        self.padding = 6 # pad the domain if input is non-periodic
        self.fc0 = nn.Linear(4, self.width)
        # input channel is 12: the solution of the first 10 timesteps + 3 locations (u(1, x, y), ..., u(10, x, y),  x, y, t)

        self.conv0 = SpectralConv3d(self.width, self.width, self.modes1, self.modes2, self.modes3)
        self.conv1 = SpectralConv3d(self.width, self.width, self.modes1, self.modes2, self.modes3)
        self.conv2 = SpectralConv3d(self.width, self.width, self.modes1, self.modes2, self.modes3)
        self.conv3 = SpectralConv3d(self.width, self.width, self.modes1, self.modes2, self.modes3)
        self.w0 = nn.Conv3d(self.width, self.width, 1)
        self.w1 = nn.Conv3d(self.width, self.width, 1)
        self.w2 = nn.Conv3d(self.width, self.width, 1)
        self.w3 = nn.Conv3d(self.width, self.width, 1)
        self.bn0 = torch.nn.BatchNorm3d(self.width)
        self.bn1 = torch.nn.BatchNorm3d(self.width)
        self.bn2 = torch.nn.BatchNorm3d(self.width)
        self.bn3 = torch.nn.BatchNorm3d(self.width)

        self.fc1 = nn.Linear(self.width, 128)
        self.fc2 = nn.Linear(128, 2)

    def forward(self, x):
        grid = self.get_grid(x.shape, device)
        x = torch.cat((x[:,:,:,:,np.newaxis], grid), dim=-1)
        x = self.fc0(x)
        x = x.permute(0, 4, 1, 2, 3)
        x = F.pad(x, [0,self.padding]) # pad the domain if input is non-periodic

        x1 = self.conv0(x)
        x2 = self.w0(x)
        x = x1 + x2
        x = F.gelu(x)

        x1 = self.conv1(x)
        x2 = self.w1(x)
        x = x1 + x2
        x = F.gelu(x)

        x1 = self.conv2(x)
        x2 = self.w2(x)
        x = x1 + x2
        x = F.gelu(x)

        x1 = self.conv3(x)
        x2 = self.w3(x)
        x = x1 + x2

        x = x[..., :-self.padding]
        x = x.permute(0, 2, 3, 4, 1) # pad the domain if input is non-periodic
        x = self.fc1(x)
        x = F.gelu(x)
        x = self.fc2(x)
        return x

    def get_grid(self, shape, device):
        batchsize, size_x, size_y, size_z = shape[0], shape[1], shape[2], shape[3]
        gridx = torch.tensor(np.linspace(0, 1, size_x), dtype=torch.float)
        gridx = gridx.reshape(1, size_x, 1, 1, 1).repeat([batchsize, 1, size_y, size_z, 1])
        gridy = torch.tensor(np.linspace(0, 1, size_y), dtype=torch.float)
        gridy = gridy.reshape(1, 1, size_y, 1, 1).repeat([batchsize, size_x, 1, size_z, 1])
        gridz = torch.tensor(np.linspace(0, 1, size_z), dtype=torch.float)
        gridz = gridz.reshape(1, 1, 1, size_z, 1).repeat([batchsize, size_x, size_y, 1, 1])
        return torch.cat((gridx, gridy, gridz), dim=-1).to(device)
    
def dice_coeff(input: Tensor, target: Tensor, reduce_batch_first: bool = False, epsilon=1e-6):
    # Average of Dice coefficient for all batches, or for a single mask
    assert input.size() == target.size()
    if input.dim() == 2 and reduce_batch_first:
        raise ValueError(f'Dice: asked to reduce batch but got tensor without batch dimension (shape {input.shape})')

    if input.dim() == 2 or reduce_batch_first:
        inter = torch.dot(input.reshape(-1), target.reshape(-1))
        sets_sum = torch.sum(input) + torch.sum(target)
        if sets_sum.item() == 0:
            sets_sum = 2 * inter

        return (2 * inter + epsilon) / (sets_sum + epsilon)
    else:
        # compute and average metric for each batch element
        dice = 0
        for i in range(input.shape[0]):
            dice += dice_coeff(input[i, ...], target[i, ...])
        return dice / input.shape[0]


def multiclass_dice_coeff(input: Tensor, target: Tensor, reduce_batch_first: bool = False, epsilon=1e-6):
    # Average of Dice coefficient for all classes
    assert input.size() == target.size()
    dice = 0
    for channel in range(input.shape[1]):
        dice += dice_coeff(input[:, channel, ...], target[:, channel, ...], reduce_batch_first, epsilon)

    return dice / input.shape[1]


def dice_loss(input: Tensor, target: Tensor, multiclass: bool = False):
    # Dice loss (objective to minimize) between 0 and 1
    assert input.size() == target.size()
    fn = multiclass_dice_coeff if multiclass else dice_coeff
    return 1 - fn(input, target, reduce_batch_first=True)


def binseg_loss(pred, target, ce_weight=0.5):
    target = torch.argmax(target,dim=1)
    weight = torch.Tensor(np.array([1.0,10.0])).to(device)
    criterion = torch.nn.CrossEntropyLoss(weight)
    ce = criterion(pred,target)    
    dice = dice_loss(F.softmax(pred, dim=1).float(),
                     F.one_hot(target.to(torch.int64), 2).permute(0, 4, 1, 2,3).float(),
                                       multiclass=True)
    loss = ce_weight*ce+(1-ce_weight)*dice

    return loss
from skimage.filters import threshold_otsu, threshold_minimum, threshold_triangle

def binarize(pred, threshold):
    thresh = threshold(pred)
    return (pred>thresh).astype(int)

class ImageSliceViewer3D:
    """ 
    ImageSliceViewer3D is for viewing volumetric image slices in jupyter or
    ipython notebooks. 
    
    User can interactively change the slice plane selection for the image and 
    the slice plane being viewed. 

    Argumentss:
    Volume = 3D input image
    figsize = default(8,8), to set the size of the figure
    cmap = default('plasma'), string for the matplotlib colormap. You can find 
    more matplotlib colormaps on the following link:
    https://matplotlib.org/users/colormaps.html
    
    """
    
    def __init__(self, volume, figsize=(8,8), cmap='plasma'):
        self.volume = volume
        self.figsize = figsize
        self.cmap = cmap
        self.v = [np.min(volume), np.max(volume)]
        
        # Call to select slice plane
        ipyw.interact(self.view_selection, view=ipyw.RadioButtons(
            options=['x-y','y-z', 'z-x'], value='x-y', 
            description='Slice plane selection:', disabled=False,
            style={'description_width': 'initial'}))
    
    def view_selection(self, view):
        # Transpose the volume to orient according to the slice plane selection
        orient = {"y-z":[1,2,0], "z-x":[2,0,1], "x-y": [0,1,2]}
        self.vol = np.transpose(self.volume, orient[view])
        maxZ = self.vol.shape[2] - 1
        
        # Call to view a slice within the selected slice plane
        ipyw.interact(self.plot_slice, 
            z=ipyw.IntSlider(min=0, max=maxZ, step=1, continuous_update=False, 
            description='Image Slice:'))
        
    def plot_slice(self, z):
        # Plot slice for the given plane and slice
        self.fig = plt.figure(figsize=self.figsize)
        plt.axis("off")
        plt.imshow(self.vol[:,:,z], cmap=plt.get_cmap(self.cmap), 
            vmin=self.v[0], vmax=self.v[1])
        
class ShapeNet_FNO_sparse2dense_DataLoader(object):
    def __init__(self, root_path, n_samples, sigma, mode='train', batch_size = 20):
        self.root_path = root_path
        self.n_samples = n_samples
        self.sigma = sigma
        self.mode = 'train'
        self.batch_size = batch_size
        if mode == 'train':
            self.data_path = self.root_path+'/Shapenet_FNO_train_nsamples_'+str(n_samples)+'_sigma_'+str(sigma)+'/'
            self.files_name = np.array(np.array([x for x in 
                                                 os.walk(self.data_path)])[0][2])
        else:
            self.data_path = self.root_path+'/Shapenet_FNO_test_nsamples_'+str(n_samples)+'_sigma_'+str(sigma)+'/'
            self.files_name = np.array(np.array([x for x in 
                                                 os.walk(self.data_path)])[0][2])
    def generate_data(self):
        voxels_batch = []
        divergence_tensor_batch = []
        for i in range(self.batch_size):
            j = np.random.randint(len(self.files_name))
            file_name = self.files_name[j]
            voxels_divergence = np.load(self.data_path+file_name)
            voxels = voxels_divergence['voxels']
            divergence_tensor = voxels_divergence['divergence_tensor']
            voxels_batch.append(voxels)
            divergence_tensor_batch.append(divergence_tensor)
        return np.array(voxels_batch), np.array(divergence_tensor_batch)

def gen_data_sparse2dense(data_path, raw_data_path, n_samples, sigma = 2, lowres = True, batch_size = 2):
    files_data_name = np.array(np.array([x for x in os.walk(data_path)])[0][2])
    divergence_tensor_batch = []
    voxels_batch = []
    points_batch = []
    normals_batch = []
    path_batch = []
    for i in range(batch_size):
        j = np.random.randint(len(files_data_name))
        file_data_name = files_data_name[j]
        path = raw_data_path+file_data_name.replace('.npz', '.binvox') 
        with open(path, 'rb') as f:
            voxels = (binvox_rw.read_as_3d_array(f).data.astype(int))
        if lowres:
            voxels = voxels[::2,::2,::2]

        verts, faces, normals, values = measure.marching_cubes(voxels, 0)

        # Create the mesh
        mesh = trimesh.base.Trimesh(verts, faces, None, normals)
        # Sampling the points
        sampled_points, sampled_indices = mesh.sample(n_samples, return_index = True)
        # compute the barycentric coordinates of each sample
        bary = trimesh.triangles.points_to_barycentric(triangles=mesh.triangles[sampled_indices], points=sampled_points)

        # interpolate vertex normals from barycentric coordinates
        sampled_normals = trimesh.unitize((mesh.vertex_normals[mesh.faces[sampled_indices]] 
                                  *trimesh.unitize(bary).reshape((-1, 3, 1))).sum(axis=1))

        grid_size = voxels.shape[0];
        data = np.round(sampled_points)

        field_V = np.zeros((grid_size, grid_size, grid_size, 3)) #Divergence of \overrightarrow{V}
        density_tensor_binary = np.zeros((grid_size, grid_size, grid_size)) # Density of points, so we can normalize
        for i in range(data.shape[0]):
            density_tensor_binary[round(data[i,0]), round(data[i,1]), round(data[i,2])] += 1

        density_tensor = scipy.ndimage.filters.gaussian_filter(np.pad(density_tensor_binary,((5,), (5,), (5,))), sigma) #Smoothed density
        for i in range(data.shape[0]): 
            field_V[round(data[i,0]), round(data[i,1]), round(data[i,2])] += sampled_normals[i] / density_tensor[round(data[i,0]) + 5, round(data[i,1]) + 5, round(data[i,2]) + 5]

        divergence_tensor = np.zeros(field_V.shape[:3])
        for i in range(3):
            divergence_tensor += np.roll(field_V, 1, axis = i)[:,:,:,i] - field_V[:,:,:,i]
        divergence_tensor = scipy.ndimage.filters.gaussian_filter(divergence_tensor,sigma)
        divergence_tensor_batch.append(divergence_tensor)
        voxels_batch.append(voxels)
        points_batch.append(sampled_points)
        normals_batch.append(sampled_normals)
        path_batch.append(path)
        
    return divergence_tensor_batch, voxels_batch, points_batch, normals_batch, path_batch

    
def visualize_pred(pred):
    verts, faces, normals, _ = measure.marching_cubes(pred, 0)
    mesh = trimesh.base.Trimesh(verts, faces, None, normals).as_open3d
    o3d.visualization.draw_plotly([mesh])
    
def save_prediction_mesh(pred, n_sample, path,  name):
    # Saving as pointclouds using marching cubes
    verts, faces, normals, _ = measure.marching_cubes(pred, 0)

    # Saving mesh as stl
    mesh = trimesh.base.Trimesh(verts, faces, None, normals)
    mesh.export(path+'/'+name+'.ply');
    
def save_results(voxels, pred_bin, points, path, n_samples, sigma, lowres=True):
    if lowres:
        saving_path = './results_n_samples_'+str(n_samples)+'_sigma_'+str(sigma)+'_lowres/'+path.split('/')[-1].split('.')[0]+'/'
    else:
        saving_path = './results_n_samples_'+str(n_samples)+'_sigma_'+str(sigma)+'_highres/'+path.split('/')[-1].split('.')[0]+'/'
    if not os.path.exists(saving_path):
        os.makedirs(saving_path)
    save_points(points, 1000, saving_path, 'measurements')
    save_prediction_mesh(voxels, n_samples, saving_path, 'gt')
    save_prediction_mesh(pred_bin, n_samples, saving_path, 'pred')
    print("Saved results")
    
def save_results_benchmarks(voxels, pred_bin, points, path):
    n_samples = len(points)
    saving_path = './output/shapenet'+str(int(np.ceil(n_samples/1000)))+'K/' 
    if not os.path.exists(saving_path):
        os.makedirs(saving_path)
    save_points(points, int(np.ceil(n_samples/50)), saving_path, path.split('/')[-1].replace('.binvox','')+'_measurements')
    save_prediction_mesh(voxels, n_samples, saving_path,path.split('/')[-1].replace('.binvox','')+'_gt')
    save_prediction_mesh(pred_bin, n_samples, saving_path, path.split('/')[-1].replace('.binvox','')+'_Neuralpoisson')
    print("Saved results")

def Neuralpoisson_predict_highres(divergence_tensor_batch,model, batch_size):
    divergence_tensor_batch = torch.Tensor(divergence_tensor_batch).cuda()

    div_normalizer = UnitGaussianNormalizer(divergence_tensor_batch)
    divergence_tensor_batch = div_normalizer.encode(divergence_tensor_batch)

    T = 64
    divergence_tensor_batch_resh = divergence_tensor_batch[:,:,:,:,np.newaxis].repeat([1,1,1,1,T])
    
    pred = model(divergence_tensor_batch_resh)[:,:,:,:,0]
    gc.collect()
    torch.cuda.empty_cache()
    
    pred = div_normalizer.decode(pred)
    gc.collect()
    torch.cuda.empty_cache()
    
    divergence_tensor_batch_cpu = divergence_tensor_batch.cpu().numpy()
    pred_cpu = F.sigmoid(pred).detach().cpu().numpy()
    
    del divergence_tensor_batch_resh
    del divergence_tensor_batch
    del div_normalizer
    del pred
    gc.collect()
    torch.cuda.empty_cache()
    return pred_cpu, divergence_tensor_batch_cpu

def gen_batch_example(path, n_samples, sigma=2, lowres = True):
    divergence_tensor_batch = []
    voxels_batch = []
    points_batch = []
    normals_batch = []
    path_batch = []
    with open(path, 'rb') as f:
            voxels = (binvox_rw.read_as_3d_array(f).data.astype(int))
    if lowres:
        voxels = voxels[::2,::2,::2]

    verts, faces, normals, values = measure.marching_cubes(voxels, 0)

    # Create the mesh
    mesh = trimesh.base.Trimesh(verts, faces, None, normals)
    # Sampling the points
    sampled_points, sampled_indices = mesh.sample(n_samples, return_index = True)
    # compute the barycentric coordinates of each sample
    bary = trimesh.triangles.points_to_barycentric(triangles=mesh.triangles[sampled_indices], points=sampled_points)

    # interpolate vertex normals from barycentric coordinates
    sampled_normals = trimesh.unitize((mesh.vertex_normals[mesh.faces[sampled_indices]] 
                              *trimesh.unitize(bary).reshape((-1, 3, 1))).sum(axis=1))

    grid_size = voxels.shape[0];
    data = np.round(sampled_points)

    field_V = np.zeros((grid_size, grid_size, grid_size, 3)) #Divergence of \overrightarrow{V}
    density_tensor_binary = np.zeros((grid_size, grid_size, grid_size)) # Density of points, so we can normalize
    for i in range(data.shape[0]):
        density_tensor_binary[round(data[i,0]), round(data[i,1]), round(data[i,2])] += 1

    density_tensor = scipy.ndimage.filters.gaussian_filter(np.pad(density_tensor_binary,((5,), (5,), (5,))), sigma) #Smoothed density
    for i in range(data.shape[0]): 
        field_V[round(data[i,0]), round(data[i,1]), round(data[i,2])] += sampled_normals[i] / density_tensor[round(data[i,0]) + 5, round(data[i,1]) + 5, round(data[i,2]) + 5]

    divergence_tensor = np.zeros(field_V.shape[:3])
    for i in range(3):
        divergence_tensor += np.roll(field_V, 1, axis = i)[:,:,:,i] - field_V[:,:,:,i]
    divergence_tensor = scipy.ndimage.filters.gaussian_filter(divergence_tensor,sigma)
    for i in range(2):
        divergence_tensor_batch.append(divergence_tensor)
        voxels_batch.append(voxels)
        points_batch.append(sampled_points)
        normals_batch.append(sampled_normals)
        path_batch.append(path)
    return divergence_tensor_batch, voxels_batch, points_batch, normals_batch, path_batch

def gen_batch_from_mesh(path2mesh, n_samples, sigma = 2, grid_size = 256):
    mesh_batch = []
    divergence_tensor_batch = []
    points_batch = []
    normals_batch = []
    path_batch = []
    # Loading mesh
    mesh = trimesh.load(path2mesh)
    # Sampling the points
    sampled_points, sampled_indices = mesh.sample(n_samples, return_index = True)
    # compute the barycentric coordinates of each sample
    bary = trimesh.triangles.points_to_barycentric(triangles=mesh.triangles[sampled_indices], points=sampled_points)

    # interpolate vertex normals from barycentric coordinates
    sampled_normals = trimesh.unitize((mesh.vertex_normals[mesh.faces[sampled_indices]] 
                              *trimesh.unitize(bary).reshape((-1, 3, 1))).sum(axis=1))
    
    sampled_points = (grid_size-1)*(sampled_points-sampled_points.min())/(sampled_points.max()-sampled_points.min())

    data = np.round(sampled_points)

    field_V = np.zeros((grid_size, grid_size, grid_size, 3)) #Divergence of \overrightarrow{V}
    density_tensor_binary = np.zeros((grid_size, grid_size, grid_size)) # Density of points, so we can normalize
    for i in range(data.shape[0]):
        density_tensor_binary[round(data[i,0]), round(data[i,1]), round(data[i,2])] += 1

    density_tensor = scipy.ndimage.filters.gaussian_filter(np.pad(density_tensor_binary,((5,), (5,), (5,))), sigma) #Smoothed density
    for i in range(data.shape[0]): 
        field_V[round(data[i,0]), round(data[i,1]), round(data[i,2])] += sampled_normals[i] / density_tensor[round(data[i,0]) + 5, round(data[i,1]) + 5, round(data[i,2]) + 5]

    divergence_tensor = np.zeros(field_V.shape[:3])
    for i in range(3):
        divergence_tensor += np.roll(field_V, 1, axis = i)[:,:,:,i] - field_V[:,:,:,i]
    divergence_tensor = scipy.ndimage.filters.gaussian_filter(divergence_tensor,sigma)
    for i in range(2):
        mesh_batch.append(mesh)
        divergence_tensor_batch.append(divergence_tensor)
        points_batch.append(sampled_points)
        normals_batch.append(sampled_normals)
        path_batch.append(path2mesh)
    return divergence_tensor_batch, mesh_batch,  points_batch, normals_batch, path_batch


## Utilities for 3D visualization
def to_homogeneous(points):
    pad = np.ones((points.shape[:-1]+(1,)), dtype=points.dtype)
    return np.concatenate([points, pad], axis=-1)


def init_figure(height: int = 800) -> go.Figure:
    """Initialize a 3D figure."""
    fig = go.Figure()
    axes = dict(
        visible=False,
        showbackground=False,
        showgrid=False,
        showline=False,
        showticklabels=True,
        autorange=True,
    )
    fig.update_layout(
        template="plotly_dark",
        height=height,
        scene_camera=dict(
            eye=dict(x=0., y=-.1, z=-2),
            up=dict(x=0, y=-1., z=0),
            projection=dict(type="orthographic")),
        scene=dict(
            xaxis=axes,
            yaxis=axes,
            zaxis=axes,
            aspectmode='data',
            dragmode='orbit',
        ),
        margin=dict(l=0, r=0, b=0, t=0, pad=0),
        legend=dict(
            orientation="h",
            yanchor="top",
            y=0.99,
            xanchor="left",
            x=0.1
        ),
    )
    return fig


def plot_points(
        fig: go.Figure,
        pts: np.ndarray,
        color: str = 'rgba(255, 0, 0, 1)',
        ps: int = 2,
        colorscale: Optional[str] = None,
        name: Optional[str] = None):
    """Plot a set of 3D points."""
    x, y, z = pts.T
    tr = go.Scatter3d(
        x=x, y=y, z=z, mode='markers', name=name, legendgroup=name,
        marker=dict(
            size=ps, color=color, line_width=0.0, colorscale=colorscale))
    fig.add_trace(tr)


def plot_camera(
        fig: go.Figure,
        R: np.ndarray,
        t: np.ndarray,
        K: np.ndarray,
        color: str = 'rgb(0, 0, 255)',
        name: Optional[str] = None,
        legendgroup: Optional[str] = None,
        size: float = 1.0):
    """Plot a camera frustum from pose and intrinsic matrix."""
    W, H = K[0, 2]*2, K[1, 2]*2
    corners = np.array([[0, 0], [W, 0], [W, H], [0, H], [0, 0]])
    if size is not None:
        image_extent = max(size * W / 1024.0, size * H / 1024.0)
        world_extent = max(W, H) / (K[0, 0] + K[1, 1]) / 0.5
        scale = 0.5 * image_extent / world_extent
    else:
        scale = 1.0
    corners = to_homogeneous(corners) @ np.linalg.inv(K).T
    corners = (corners / 2 * scale) @ R.T + t

    x, y, z = corners.T
    rect = go.Scatter3d(
        x=x, y=y, z=z, line=dict(color=color), legendgroup=legendgroup,
        name=name, marker=dict(size=0.0001), showlegend=False)
    fig.add_trace(rect)

    x, y, z = np.concatenate(([t], corners)).T
    i = [0, 0, 0, 0]
    j = [1, 2, 3, 4]
    k = [2, 3, 4, 1]

    pyramid = go.Mesh3d(
        x=x, y=y, z=z, color=color, i=i, j=j, k=k,
        legendgroup=legendgroup, name=name, showlegend=False)
    fig.add_trace(pyramid)
    triangles = np.vstack((i, j, k)).T
    vertices = np.concatenate(([t], corners))
    tri_points = np.array([
        vertices[i] for i in triangles.reshape(-1)
    ])

    x, y, z = tri_points.T
    pyramid = go.Scatter3d(
        x=x, y=y, z=z, mode='lines', legendgroup=legendgroup,
        name=name, line=dict(color=color, width=1), showlegend=False)
    fig.add_trace(pyramid)


def plot_camera_colmap(
        fig: go.Figure,
        image: pycolmap.Image,
        camera: pycolmap.Camera,
        name: Optional[str] = None,
        **kwargs):
    """Plot a camera frustum from PyCOLMAP objects"""
    plot_camera(
        fig,
        image.rotmat().T,
        image.projection_center(),
        camera.calibration_matrix(),
        name=name or str(image.image_id),
        **kwargs)


def plot_cameras(
        fig: go.Figure,
        reconstruction: pycolmap.Reconstruction,
        **kwargs):
    """Plot a camera as a cone with camera frustum."""
    for image_id, image in reconstruction.images.items():
        plot_camera_colmap(
            fig, image, reconstruction.cameras[image.camera_id], **kwargs)


def plot_reconstruction(
        fig: go.Figure,
        rec: pycolmap.Reconstruction,
        max_reproj_error: float = 6.0,
        color: str = 'rgb(0, 0, 255)',
        name: Optional[str] = None,
        min_track_length: int = 2,
        points: bool = True,
        cameras: bool = True,
        cs: float = 1.0):
    # Filter outliers
    bbs = rec.compute_bounding_box(0.001, 0.999)
    # Filter points, use original reproj error here
    xyzs = [p3D.xyz for _, p3D in rec.points3D.items() if (
                            (p3D.xyz >= bbs[0]).all() and
                            (p3D.xyz <= bbs[1]).all() and
                            p3D.error <= max_reproj_error and
                            p3D.track.length() >= min_track_length)]
    if points:
        plot_points(fig, np.array(xyzs), color=color, ps=1, name=name)
    if cameras:
        plot_cameras(fig, rec, color=color, legendgroup=name, size=cs)