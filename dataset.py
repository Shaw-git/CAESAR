import os
from glob import glob
import json
import threading

import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F
import torchvision.transforms as T

import numpy as np
from scipy.ndimage import zoom
from tqdm import tqdm
import torch
from torch.utils.data import Dataset
import torch.nn.functional as F
import torchvision.transforms as T
from copy import deepcopy
import numpy as np

import torch
import torch.nn.functional as F
import math


def center_crop(x, tshape):
    _, _,_, H, W = x.shape
    
    target_h, target_w = tshape
    start_h = (H - target_h) // 2
    start_w = (W - target_w) // 2
    end_h = start_h + target_h
    end_w = start_w + target_w
    return x[..., start_h:end_h, start_w:end_w]

def downsampling_data(data, zoom_factors):
    """Apply cubic interpolation-based downsampling or upsampling."""
    return zoom(data, zoom_factors, order=3)  # order=3 = cubic interpolation


def block_hw(data, block_size=(256, 256)):
    V, S, T, H, W = data.shape
    h_block, w_block = block_size

    # Compute padding for height to nearest multiple of h_block
    H_target = math.ceil(H / h_block) * h_block
    dh = H_target - H
    top, down = dh // 2, dh - dh // 2
    n_h = H_target // h_block

    # Compute padding for width to nearest multiple of w_block
    W_target = math.ceil(W / w_block) * w_block
    dw = W_target - W
    left, right = dw // 2, dw - dw // 2
    n_w = W_target // w_block
    
    V1, S1, T1, H1, W1 = data.shape
    data = data.view(V1*S1,T1, H1, W1)
    # Apply reflection padding (order: W_left, W_right, H_top, H_bottom)
    data = F.pad(data, (left, right, top, down), mode='reflect')
    data = data.view(V1, S1, T1,*data.shape[-2:])
    # Update shape after padding
    V, S, T, H_p, W_p = data.shape

    # Reshape and split along H and W into blocks
    data = data.reshape(V, S, T, n_h, h_block, n_w, w_block)
    data = data.permute(0, 1, 3, 5, 2, 4, 6)  # [V, S, n_h, n_w, T, h_block, w_block]
    data = data.reshape(V, S * n_h * n_w, T, h_block, w_block)

    padding = (top, down, left, right)
    return data, (n_h, n_w, padding)


def deblock_hw(data, n_h, n_w, padding):
    V, S_blk, T, h_block, w_block = data.shape
    top, down, left, right = padding

    S_orig = S_blk // (n_h * n_w)

    # Reshape to original split
    data = data.reshape(V, S_orig, n_h, n_w, T, h_block, w_block)
    data = data.permute(0, 1, 4, 2, 5, 3, 6)
    data = data.reshape(V, S_orig, T, n_h * h_block, n_w * w_block)

    # Remove padding
    H_p, W_p = n_h * h_block, n_w * w_block
    H = H_p - top - down
    W = W_p - left - right

    data = data[:, :, :, top:top+H, left:left+W]

    return data



def normalize_data(data, norm_type, axis):
    """
    Normalize data according to the specified normalization type.

    Args:
        data (np.ndarray): Input data array.
        norm_type (str): Type of normalization ('std', 'min_max', 'mean_range').
        axis (tuple or int): Axis or axes along which to compute statistics.

    Returns:
        tuple: (normalized_data, var_mean, var_scale)
    """
    norm_type = norm_type.lower()

    if not isinstance(data, np.ndarray):
        raise TypeError("Input data must be a numpy ndarray.")

    if norm_type == "std":
        var_mean = np.mean(data, axis=axis, keepdims=True)
        var_scale = np.std(data, axis=axis, keepdims=True)
        normalized_data = (data - var_mean) / var_scale

    elif norm_type == "min_max":
        var_min = np.min(data, axis=axis, keepdims=True)
        var_max = np.max(data, axis=axis, keepdims=True)
        var_range = var_max - var_min

        var_mean = var_min + var_range / 2
        var_scale = var_range / 2

        normalized_data = (data - var_mean) / var_scale

    elif norm_type == "mean_range":
        var_mean = np.mean(data, axis=axis, keepdims=True)
        var_max = np.max(data, axis=axis, keepdims=True)
        var_min = np.min(data, axis=axis, keepdims=True)
        var_scale = var_max - var_min

        normalized_data = (data - var_mean) / var_scale

    else:
        raise NotImplementedError(f"Normalization type '{norm_type}' is not implemented.")

    return normalized_data, var_mean, var_scale


class BaseDataset(Dataset):
    def __init__(self, args):
        args = deepcopy(args)
        # Universal configs
        self.data_path      = args["data_path"]
        
        self.dataset_name   = args.get("name", "Customized Dataset")
        
        self.variable_idx   = args.get("variable_idx")
        self.section_range  = args.get("section_range")
        self.frame_range    = args.get("frame_range")
        
        self.n_frame        = args["n_frame"]
        
        self.resolution     = args.get("resolution", None)
        
        self.train_size     = args.get("train_size", 256)
        self.inst_norm      = args.get("inst_norm", True)
        self.augment_type   = args.get("augment_type", {})
        self.norm_type      = args.get("norm_type", "mean_range")
        
        self.train_mode     = args.get("train")
        
        self.test_size      = args.get("test_size", (256,256))
        self.n_overlap      = args.get("n_overlap", 0)
        self.downsampling   = args.get("downsampling", 1)
        
        
        
        
        self.random_crop = T.RandomCrop(size=(self.train_size, self.train_size))

        if "downsample" in self.augment_type:
            self.max_downsample = self.augment_type["downsample"]
        elif "randsample" in self.augment_type:
            self.max_downsample = self.augment_type["randsample"]
        else:
            self.max_downsample = 1

        self.enble_ds = True
        
    def apply_augments(self,data):
        
        if ("downsample" in self.augment_type) and self.enble_ds:
            data = self.apply_downsampling(data, step=self.augment_type["downsample"])
        elif ("randsample" in self.augment_type) and self.enble_ds:
            step = torch.randint(1, self.augment_type["randsample"] + 1, (1,)).item()
            data = self.apply_downsampling(data, step=step)
            
        return data

    def apply_padding_or_crop(self, data):
        cur_size = data.shape[-1]
        if self.train_size > cur_size:
            pad_size = self.train_size - cur_size
            pad_left = pad_size // 2
            pad_right = pad_size - pad_left
            data = F.pad(data[None], (pad_left, pad_right, pad_left, pad_right), mode='reflect')[0]
        elif self.train_size < cur_size:
            data = self.random_crop(data)
        return data

    def apply_inst_norm(self, data, return_norm=False):
        if self.norm_type == "mean_range":
            offset = torch.mean(data).view([1,1,1])
            scale = data.max() - data.min()            
            assert scale != 0, "Scale is zero."
            data = (data - offset) / scale
            
            offset = offset.view([1,1,1])
            scale = scale.view([1,1,1])
            

        elif self.norm_type == "min_max":
            dmin = data.min()
            dmax = data.max()
            offset = (dmax + dmin) / 2
            scale = (dmax - dmin) / 2
            assert scale != 0, "Scale is zero."
            data = (data - offset) / scale
            
            offset = offset.view([1,1,1])
            scale = scale.view([1,1,1])
            

        elif self.norm_type == "mean_range_hw":
            offset = torch.mean(data, dim=(-2, -1), keepdim=True)
            scale = torch.amax(data, dim=(-2, -1), keepdim=True) - torch.amin(data, dim=(-2, -1), keepdim=True)
            assert torch.all(scale != 0), "Scale is zero."
            data = (data - offset) / scale

        else:
            raise NotImplementedError(f"Normalization type {self.norm_type} not implemented.")

        if return_norm:
            return data, offset, scale
        else:
            return data

        
    
class ScientificDataset(BaseDataset):
    def __init__(self, args):
        super().__init__(args)

        print(f"*************** Loading {self.dataset_name} ***************")
        
        data = self.load_dataset(self.data_path, self.variable_idx, self.section_range, self.frame_range)
        
        self.shape_org = data.shape
        
        self.delta_t = self.n_frame - self.n_overlap
        T = data.shape[2]
        self.t_samples = math.ceil((T - self.n_frame) / self.delta_t) + 1
        
        pad_T = (self.t_samples - 1) * self.delta_t + self.n_frame - T
        self.pad_T = pad_T
        
        if pad_T > 0:
            tail_frames = data[:, :, -pad_T:]
            tail_frames = tail_frames[:, :, ::-1] 
            data = np.concatenate([data, tail_frames], axis=2)
        #self.shape = data.shape
            
        if not self.inst_norm:
            assert(self.norm_type != "mean_range_hw")
            data, var_offset, var_scale = normalize_data(data, self.norm_type, axis=(1, 2, 3, 4))
            self.var_offset, self.var_scale = torch.FloatTensor(var_offset), torch.FloatTensor(var_scale)
        data = torch.FloatTensor(data)
        
        if not self.train_mode:
            data, self.block_info = block_hw(data, self.test_size)
            print("Testing Data Shape",self.shape_org)
        self.shape = data.shape

        self.data_input = data 
        self.visble_length = self.update_length()
        
        
    def original_data(self,):
        data =  self.data_input
        if not self.train_mode:
            data = deblock_hw(data, *self.block_info)
        if not self.inst_norm:
            data = data * self.var_scale + self.var_offset
        return data
    
    def input_data(self,):
        data = self.original_data()
        data = data[:, :, :self.shape[2]-self.pad_T, :, :]
        return data
        
    def recons_data(self, recons_data):
        return recons_data[:, :, :self.shape[2]-self.pad_T, :, :]
            
    def load_dataset(self, data_path, variable_idx, section_range, frame_range):
        frame_range   = slice(None) if frame_range   is None else slice(frame_range[0], frame_range[1])
        section_range = slice(None) if section_range is None else slice(section_range[0], section_range[1])

        with np.load(data_path) as npzfile:
            data = npzfile["data"][variable_idx, section_range, frame_range]
        
        if self.resolution is not None:
            data = center_crop(data, self.resolution)
        self.dtype = data.dtype
        data = data.astype(np.float32)
        return data
    
    def deblocking_hw(self,data):
        return deblock_hw(data, *self.block_info)

    def update_length(self):
        self.dataset_length = self.shape[0] * self.shape[1] * self.t_samples
        return self.dataset_length

    def __len__(self):
        return self.visble_length

    def post_processing(self, data, var_idx, is_training):
        if is_training:
            data = self.apply_augments(data)
            data = self.apply_padding_or_crop(data)

        if self.inst_norm:
            data, offset, scale = self.apply_inst_norm(data, True)
        else:
            offset = self.var_offset[var_idx].view(1,1,1)
            scale = self.var_scale[var_idx].view(1,1,1)

        data_dict = {"input": data[None], "offset": offset[None], "scale": scale[None]}
        return data_dict

    def __getitem__(self, idx):
        idx = idx % self.dataset_length
        idx0 = idx // (self.shape[1] * self.t_samples)
        idx1 = (idx // self.t_samples) % self.shape[1]
        idx2 = idx % self.t_samples

        start_t = idx2 * self.delta_t
        end_t = start_t + self.n_frame

        data = self.data_input[idx0, idx1, start_t:end_t]
        data = self.post_processing(data, idx0, self.train_mode)
        data["index"] = [idx0, idx1, start_t,end_t]
        return data

