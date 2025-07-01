import math
import torch
import numpy as np
from compressai.entropy_models import GaussianConditional
from compressai.entropy_models.entropy_models_vbr import EntropyModelVbr


def _build_indexes(size):
    dims = len(size)
    N = size[0]
    C = size[1]

    view_dims = np.ones((dims,), dtype=np.int64)
    view_dims[1] = -1
    indexes = torch.arange(C).view(*view_dims)
    indexes = indexes.int()

    return indexes.repeat(N, 1, *size[2:])

def _extend_ndims(tensor, n):
    return tensor.reshape(-1, *([1] * n)) if n > 0 else tensor.reshape(-1)

    
class RangeCoder:
    def __init__(self, scale_bound=0.1, max_scale=20, scale_steps=128, 
                 _quantized_cdf = None, _cdf_length= None, _offset= None, medians = None,
                 device="cpu"):
        
        self.device = device
        self.gaussian = GaussianConditional(None, scale_bound=scale_bound)
        
        lower = self.gaussian.lower_bound_scale.bound.item()
        scale_table = torch.exp(torch.linspace(math.log(lower), math.log(max_scale), steps=scale_steps))
        self.gaussian.update_scale_table(scale_table)
        self.gaussian.update()
        
        self.entropy_model = EntropyModelVbr()
        
        self.entropy_model._quantized_cdf = _quantized_cdf
        self.entropy_model._cdf_length = _cdf_length
        self.entropy_model._offset = _offset
        
        self.medians = medians
        

    def compress(self, latent, mean, scale):
        latent, mean, scale = latent.to(self.device), mean.to(self.device), scale.to(self.device)
        indexes = self.gaussian.build_indexes(scale.clamp(min=0.1))
        strings = self.gaussian.compress(latent, indexes, means=mean)
        return strings
    
    def compress_hyperlatent(self, latent, qs= None):
        indexes = _build_indexes(latent.size())
        spatial_dims = len(latent.size()) - 2
        medians = _extend_ndims(self.medians, spatial_dims)
        medians = medians.expand(latent.size(0), *([-1] * (spatial_dims + 1)))
        strings = self.entropy_model.compress(latent, indexes, medians, qs)
        
        return strings
        
    
    def decompress_hyperlatent(self, strings, size, qs= None):
        size = size[-2:]
        output_size = (len(strings), self.entropy_model._quantized_cdf.size(0), *size[-2:])
        indexes = _build_indexes(output_size).to(self.entropy_model._quantized_cdf.device)
        medians = _extend_ndims(self.medians, len(size))
        medians = medians.expand(len(strings), *([-1] * (len(size) + 1)))
        return self.entropy_model.decompress(strings, indexes, medians.dtype, medians, qs)
    

    def decompress(self, strings, mean, scale):
        mean, scale = mean.to(self.device), scale.to(self.device)
        indexes = self.gaussian.build_indexes(scale.clamp(min=0.1))
        decoded_latent = self.gaussian.decompress(strings, indexes, means=mean)
        return decoded_latent
    
