# Code in this file is based on the following project:
# https://github.com/autonomousvision/shape_as_points
# Associated with the paper: 
# Peng, Songyou, et al. "Shape as points: A differentiable poisson solver." Advances in Neural Information Processing Systems 34 (2021): 13032-13044.

import torch
import torch.nn as nn
import numpy as np
import torch.fft

def fftfreqs(res, dtype=torch.float32, exact=True):
    """
    Helper function to return frequency tensors
    :param res: n_dims int tuple of number of frequency modes
    :return:
    """

    n_dims = len(res)
    freqs = []
    for dim in range(n_dims - 1):
        r_ = res[dim]
        freq = np.fft.fftfreq(r_, d=1/r_)
        freqs.append(torch.tensor(freq, dtype=dtype))
    r_ = res[-1]
    if exact:
        freqs.append(torch.tensor(np.fft.rfftfreq(r_, d=1/r_), dtype=dtype))
    else:
        freqs.append(torch.tensor(np.fft.rfftfreq(r_, d=1/r_)[:-1], dtype=dtype))
    omega = torch.meshgrid(freqs)
    omega = list(omega)
    omega = torch.stack(omega, dim=-1)

    return omega

def img(x, deg=1): # imaginary of tensor (assume last dim: real/imag)
    """
    multiply tensor x by i ** deg
    """
    deg %= 4
    if deg == 0:
        res = x
    elif deg == 1:
        res = x[..., [1, 0]]
        res[..., 0] = -res[..., 0]
    elif deg == 2:
        res = -x
    elif deg == 3:
        res = x[..., [1, 0]]
        res[..., 1] = -res[..., 1]
    return res

def grid_interp(grid, pts, batched=True):
    """
    :param grid: tensor of shape (batch, *size, in_features)
    :param pts: tensor of shape (batch, num_points, dim) within range (0, 1)
    :return values at query points
    """
    if not batched:
        grid = grid.unsqueeze(0)
        pts = pts.unsqueeze(0)
    dim = pts.shape[-1]
    bs = grid.shape[0]
    size = torch.tensor(grid.shape[1:-1]).to(grid.device).type(pts.dtype)
    cubesize = 1.0 / size
    
    ind0 = torch.floor(pts / cubesize).long()  # (batch, num_points, dim)
    ind1 = torch.fmod(torch.ceil(pts / cubesize), size).long() # periodic wrap-around
    ind01 = torch.stack((ind0, ind1), dim=0) # (2, batch, num_points, dim)
    tmp = torch.tensor([0,1],dtype=torch.long)
    com_ = torch.stack(torch.meshgrid(tuple([tmp] * dim)), dim=-1).view(-1, dim)
    dim_ = torch.arange(dim).repeat(com_.shape[0], 1) # (2**dim, dim)
    ind_ = ind01[com_, ..., dim_]   # (2**dim, dim, batch, num_points)
    ind_n = ind_.permute(2, 3, 0, 1) # (batch, num_points, 2**dim, dim)
    ind_b = torch.arange(bs).expand(ind_n.shape[1], ind_n.shape[2], bs).permute(2, 0, 1) # (batch, num_points, 2**dim)
    # latent code on neighbor nodes
    if dim == 2:
        lat = grid.clone()[ind_b, ind_n[..., 0], ind_n[..., 1]] # (batch, num_points, 2**dim, in_features)
    else:
        lat = grid.clone()[ind_b, ind_n[..., 0], ind_n[..., 1], ind_n[..., 2]] # (batch, num_points, 2**dim, in_features)

    # weights of neighboring nodes
    xyz0 = ind0.type(cubesize.dtype) * cubesize        # (batch, num_points, dim)
    xyz1 = (ind0.type(cubesize.dtype) + 1) * cubesize  # (batch, num_points, dim)
    xyz01 = torch.stack((xyz0, xyz1), dim=0) # (2, batch, num_points, dim)
    pos = xyz01[com_, ..., dim_].permute(2,3,0,1)   # (batch, num_points, 2**dim, dim)
    pos_ = xyz01[1-com_, ..., dim_].permute(2,3,0,1)   # (batch, num_points, 2**dim, dim)
    pos_ = pos_.type(pts.dtype)
    dxyz_ = torch.abs(pts.unsqueeze(-2) - pos_) / cubesize # (batch, num_points, 2**dim, dim)
    weights = torch.prod(dxyz_, dim=-1, keepdim=False) # (batch, num_points, 2**dim)
    query_values = torch.sum(lat * weights.unsqueeze(-1), dim=-2)  # (batch, num_points, in_features)
    if not batched:
        query_values = query_values.squeeze(0)
        
    return query_values

class DPSR_grid(nn.Module):
    def __init__(self, res, scale=True, shift=True):
        """
        :param res: tuple of output field resolution. eg., (128,128)
        :param sig: degree of gaussian smoothing
        """
        super(DPSR_grid, self).__init__()
        self.res = res
        self.dim = len(res)
        self.denom = np.prod(res)
        # self.G.requires_grad = False # True, if we also make sig a learnable parameter
        self.omega = fftfreqs(res, dtype=torch.float32)
        self.scale = scale
        self.shift = shift
        
    def forward(self, ras_p, screen):
        ras_s = torch.fft.rfftn(ras_p, dim=(2,3,4))
        ras_s = ras_s.permute(*tuple([0]+list(range(2, self.dim+1))+[self.dim+1, 1]))
        N_ = ras_s[..., None] # [b, dim0, dim1, dim2/2+1, n_dim, 1]
        print(f'N {N_[0].requires_grad}')

        omega = fftfreqs(self.res, dtype=torch.float32).unsqueeze(-1) # [dim0, dim1, dim2/2+1, n_dim, 1]
        omega *= 2 * np.pi  # normalize frequencies
        omega = omega.to(ras_p.device)
        print(f'omega {omega.requires_grad}')
        DivN = torch.sum(-img(torch.view_as_real(N_[..., 0])) * omega, dim=-2)
        
        Lap = -torch.sum(omega**2, -2) # [dim0, dim1, dim2/2+1, 1]
        Phi = DivN / (Lap+1e-6) # [b, dim0, dim1, dim2/2+1, 2]  
        Phi = Phi.permute(*tuple([list(range(1,self.dim+2)) + [0]]))  # [dim0, dim1, dim2/2+1, 2, b] 
        Phi[tuple([0] * self.dim)] = 0
        Phi = Phi.permute(*tuple([[self.dim+1] + list(range(self.dim+1))]))  # [b, dim0, dim1, dim2/2+1, 2]
        
        phi = torch.fft.irfftn(torch.view_as_complex(Phi), s=self.res, dim=(1,2,3))
        
        print(f'phi {omega.requires_grad}')
        if self.shift or self.scale:
            # ensure values at points are zero
            if self.shift: # offset points to have mean of 0
                offset = torch.mean(phi[screen >= 0.1], dim=-1)  # [b,] 
                phi -= offset
            if self.scale:
                phi /= torch.mean(torch.abs(phi))
        return phi

def DPSR_grid2(ras_p, screen=None, shift=False, scale=False):
    res = list(ras_p.shape[-3:])
    dim = 3
    
    ras_s = torch.fft.rfftn(ras_p, dim=(2,3,4))
    ras_s = ras_s.permute(*tuple([0]+list(range(2, dim+1))+[dim+1, 1]))
    N_ = ras_s[..., None] # [b, dim0, dim1, dim2/2+1, n_dim, 1]
    omega = fftfreqs(res, dtype=torch.float32).unsqueeze(-1) # [dim0, dim1, dim2/2+1, n_dim, 1]
    omega *= 2 * np.pi  # normalize frequencies
    omega = omega.to(ras_s.device)

    DivN = torch.sum(-img(torch.view_as_real(N_[..., 0])) * omega, dim=-2)

    Lap = -torch.sum(omega**2, -2) # [dim0, dim1, dim2/2+1, 1]
    Phi = DivN / (Lap+1e-6) # [b, dim0, dim1, dim2/2+1, 2]  
    Phi = Phi.permute(*tuple([list(range(1,dim+2)) + [0]]))  # [dim0, dim1, dim2/2+1, 2, b] 
    Phi[tuple([0] * dim)] = 0
    Phi = Phi.permute(*tuple([[dim+1] + list(range(dim+1))]))  # [b, dim0, dim1, dim2/2+1, 2]

    phi = torch.fft.irfftn(torch.view_as_complex(Phi), s=res, dim=(1,2,3))

    if shift or scale:
        # ensure values at points are zero
        if shift: # offset points to have mean of 0
            #offset = torch.sum(phi * (screen >= 0.1)) / torch.sum(screen >= 0.1)  # [b,] #torch.mean(phi[screen >= 0.1], dim=-1)
            #screen_binary = torch.heaviside(screen-0.1, torch.tensor([0.0]))
            #offset = torch.sum(phi * screen_binary) / torch.sum(screen_binary)  # [b,] #torch.mean(phi[screen >= 0.1], dim=-1)
            offset = torch.mean(phi * screen)  # [b,] #torch.mean(phi[screen >= 0.1], dim=-1)
            phi -= offset
        if scale:
            return phi / torch.max(torch.abs(phi))#* torch.sum(phi * phi) / torch.sum(phi * phi * phi * phi)
    return phi