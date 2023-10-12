import torch
import numpy as np
import torch.nn as nn

import torch.nn.functional as F

import matplotlib.pyplot as plt
from tqdm import tqdm
from time import sleep

import numpy as np
import open3d as o3d
import scipy
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import odl
import mcubes
import binvox_rw
from skimage import measure
import trimesh

import sys
import os

def generate_voxel_divergence_nonuniform(path, n_samples, sigma = 2, lowres = True):
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
    return voxels, divergence_tensor

def export_sampled_divergence(root_path, output_path, files, n_samples, sigma, lowres = True):
    # Creating output_path if it does not exists yet
    if not os.path.exists(output_path):
        os.makedirs(output_path)
    for i in tqdm(range(len(files))):
        sleep(3)
        file = files[i]
        file_path = root_path+file.replace('.npz','.binvox')
        voxels, divergence_tensor = generate_voxel_divergence_nonuniform(file_path, n_samples, sigma, lowres)
        np.savez(output_path+'/'+file.split('.')[0]+'.npz', voxels=voxels, divergence_tensor=divergence_tensor)

root_path = '../data/'# This is the folder to all the shapenet .binvox files
path_train_data = '/scratch/Data/Fourier_Neural_Data/Shapenet_FNO_nonuniform_train/'
path_test_data = '/scratch/Data/Fourier_Neural_Data/Shapenet_FNO_nonuniform_test/'

files_train = np.array(np.array([x for x in os.walk(path_train_data)])[0][2])
files_test = np.array(np.array([x for x in os.walk(path_test_data)])[0][2])
        
files = files_train
n_samples = 50000
sigma = 0.7
output_path = '/scratch/Data/Fourier_Neural_Data/Sparse2Dense/Shapenet_FNO_train_nsamples_'+str(n_samples)+'_sigma_'+str(sigma)

export_sampled_divergence(root_path, output_path, files, n_samples, sigma)