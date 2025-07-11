import torch.nn as nn
import math
import torch
import torch.nn.functional as F
import numpy as np
from .utils import exists, LowerBound
from einops import rearrange

from compressai._CXX import pmf_to_quantized_cdf as _pmf_to_quantized_cdf



def pmf_to_quantized_cdf(pmf, precision = 16):
    cdf = _pmf_to_quantized_cdf(pmf.tolist(), precision)
    cdf = torch.IntTensor(cdf)
    return cdf


class Residual(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, x, *args, **kwargs):
        return self.fn(x, *args, **kwargs) + x


class SinusoidalPosEmb(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, x):
        device = x.device
        half_dim = self.dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
        emb = x[:, None] * emb[None, :]
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)
        return emb


class Upsample(nn.Module):
    def __init__(self, dim_in, dim_out=None, d3 = False):
        super().__init__()
        if dim_out is None:
            dim_out = dim_in
            
        self.conv = nn.ConvTranspose3d(dim_in, dim_out, 4, 2, 1) if d3 else nn.ConvTranspose2d(dim_in, dim_out, 4, 2, 1)

    def forward(self, x):
        return self.conv(x)


class Downsample(nn.Module):
    def __init__(self, dim_in, dim_out=None, stride=2,  d3 = False):
        super().__init__()
        if dim_out is None:
            dim_out = dim_in
        self.conv = nn.Conv3d(dim_in, dim_out, 3, stride, 1) if d3 else nn.Conv2d(dim_in, dim_out, 3, 2, 1)

    def forward(self, x):
        return self.conv(x)


class LayerNorm(nn.Module):
    def __init__(self, dim, d3=False, eps=1e-5):
        super().__init__()
        self.eps = eps
        shape = (1, dim, 1, 1, 1) if d3 else (1, dim, 1, 1) 
        self.g = nn.Parameter(torch.ones(*shape))
        self.b = nn.Parameter(torch.zeros(*shape))

    def forward(self, x):
        var = torch.var(x, dim=1, unbiased=False, keepdim=True)
        mean = torch.mean(x, dim=1, keepdim=True)
        
        return (x - mean) / (var + self.eps).sqrt() * self.g + self.b


class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.fn = fn
        self.norm = LayerNorm(dim)

    def forward(self, x):
        x = self.norm(x)
        return self.fn(x)





class Block(nn.Module):
    def __init__(self, dim, dim_out, large_filter=False, d3=False):
        super().__init__()
        conv_layer = nn.Conv3d if d3 else nn.Conv2d
        self.block = nn.Sequential(
            conv_layer(dim, dim_out, 7 if large_filter else 3, padding=3 if large_filter else 1), 
            LayerNorm(dim_out, d3=d3), 
            nn.ReLU()
        )

    def forward(self, x):
        return self.block(x)
    


class ResnetBlock(nn.Module):
    def __init__(self, dim, dim_out, time_emb_dim=None, large_filter=False, d3=False):
        super().__init__()
        
        conv_layer = nn.Conv3d if d3 else nn.Conv2d
        
        self.mlp = (
            nn.Sequential(nn.LeakyReLU(0.2), nn.Linear(time_emb_dim, dim_out))
            if exists(time_emb_dim)
            else None
        )

        self.block1 = Block(dim, dim_out, large_filter, d3=d3)
        self.block2 = Block(dim_out, dim_out, d3=d3)
        self.res_conv = conv_layer(dim, dim_out, 1) if dim != dim_out else nn.Identity()

    def forward(self, x, time_emb=None):
        h = self.block1(x)

        if exists(time_emb):
            h = h + self.mlp(time_emb)[:, :, None, None]

        h = self.block2(h)
        return h + self.res_conv(x)
    

class CALayer(nn.Module):
    def __init__(self, channel, reduction=4, d3 = False):
        super(CALayer, self).__init__()
        # global average pooling: feature --> point
        self.avg_pool = nn.AdaptiveAvgPool3d(1) if d3 else nn.AdaptiveAvgPool2d(1)
        conv_layer = nn.Conv3d if d3 else nn.Conv2d
        # feature channel downscale and upscale --> channel weight
        self.conv_du = nn.Sequential(
                conv_layer(channel, channel // reduction, 1, padding=0, bias=True),
                nn.ReLU(inplace=True),
                conv_layer(channel // reduction, channel, 1, padding=0, bias=True),
                nn.Sigmoid()
        )

    def forward(self, x):
        y = self.avg_pool(x)
   
        y = self.conv_du(y)
        return x * y
    
class ResnetBlockAtten(nn.Module):
    def __init__(self, dim, dim_out, time_emb_dim=None, large_filter=False, d3=False, atten_reduction = 4):
        super().__init__()
        conv_layer = nn.Conv3d if d3 else nn.Conv2d
        
        self.res_block = ResnetBlock(dim, dim_out, time_emb_dim, large_filter, d3)
        self.atten_block = CALayer(dim_out, atten_reduction, d3)
        self.res_conv = conv_layer(dim, dim_out, 1) if dim != dim_out else nn.Identity()

    def forward(self, x, time_emb=None):
        h = self.res_block(x)
        h = self.atten_block(h)
        return h + self.res_conv(x)
    
class ChannelShuffle(nn.Module):
    def __init__(self, scale_factor = 2):
        super().__init__()
        self.scale_factor = scale_factor
        
    def forward(self,inputs):

        batch_size, channels, *in_dims = inputs.size()
        # in_depth, in_height, in_width = dims
        channels //= self.scale_factor ** len(in_dims)
        
        out_dims = [dim * self.scale_factor for dim in in_dims]
  
        
        if len(in_dims)==3:
            input_view = inputs.contiguous().view(batch_size, channels, self.scale_factor, self.scale_factor, self.scale_factor, *in_dims)
            shuffle_out = input_view.permute(0, 1, 5, 2, 6, 3, 7, 4).contiguous()
        else:                                         #  0      1          2                    3                  4       5
            input_view = inputs.contiguous().view(batch_size, channels, self.scale_factor, self.scale_factor,  *in_dims)
            shuffle_out = input_view.permute(0, 1, 4, 2, 5, 3).contiguous()

        return shuffle_out.view(batch_size, channels, *out_dims)
    
    
# class MultiScaleResnetBlock(nn.Module):
#     def __init__(self, dim, dim_out, time_emb_dim=None, large_filter=False, d3=False):
#         super().__init__()
        
#         conv_layer = nn.Conv3d if d3 else nn.Conv2d
        
#         self.mlp = (
#             nn.Sequential(nn.LeakyReLU(0.2), nn.Linear(time_emb_dim, dim_out))
#             if exists(time_emb_dim)
#             else None
#         )
        
#         self.shuffle_up = ChannelShuffle(2)
        
#         self.merge_up = 
        

#         self.block1 = Block(dim, dim_out, large_filter, d3=d3)
#         self.down1 = nn.Sequential(Downsample(dim, dim_out, d3 = d3),nn.LeakyReLU(0.2))
        
#         self.block2 = Block(dim_out, dim_out, d3=d3)
#         self.down2 = nn.Sequential(Downsample(dim_out, dim_out, d3 = d3),nn.LeakyReLU(0.2))
        
#         self.res_conv = conv_layer(dim, dim_out, 1) if dim != dim_out else nn.Identity()

#     def forward(self, x, time_emb=None):
#         h = self.block1(x)
        
#         h2 = self.down1(h)
        
#         h2_up = self.shuffle_up(h2)
        
#         h4 = self.down2(h2)
        
#         h4_up = self.shuffle_up(h4)
        
        
        
        
        

#         if exists(time_emb):
#             h = h + self.mlp(time_emb)[:, :, None, None]

#         h = self.block2(h)
#         return h + self.res_conv(x)


class LinearAttention(nn.Module):
    def __init__(self, dim, heads=1, dim_head=None):
        super().__init__()
        if dim_head is None:
            dim_head = dim
        self.scale = dim_head ** -0.5
        self.heads = heads
        hidden_dim = dim_head * heads
        self.to_qkv = nn.Conv2d(dim, hidden_dim * 3, 1, bias=False)
        self.to_out = nn.Conv2d(hidden_dim, dim, 1)

    def forward(self, x):
        b, c, h, w = x.shape
        qkv = self.to_qkv(x).chunk(3, dim=1)
        q, k, v = map(lambda t: rearrange(t, "b (h c) x y -> b h c (x y)", h=self.heads), qkv)
        q = q * self.scale

        k = k.softmax(dim=-1)
        context = torch.einsum("b h d n, b h e n -> b h d e", k, v)

        out = torch.einsum("b h d e, b h d n -> b h e n", context, q)
        out = rearrange(out, "b h c (x y) -> b (h c) x y", h=self.heads, x=h, y=w)
        return self.to_out(out)


class LearnedSinusoidalPosEmb(nn.Module):
    def __init__(self, dim):
        super().__init__()
        assert (dim % 2) == 0
        half_dim = dim // 2
        self.weights = nn.Parameter(torch.randn(half_dim))

    def forward(self, x):
        x = rearrange(x, 'b -> b 1')
        freqs = x * rearrange(self.weights, 'd -> 1 d') * 2 * math.pi
        fouriered = torch.cat((freqs.sin(), freqs.cos()), dim = -1)
        fouriered = torch.cat((x, fouriered), dim = -1)
        return fouriered

class ImprovedSinusoidalPosEmb(nn.Module):
    """ following @crowsonkb 's lead with random (learned optional) sinusoidal pos emb """
    """ https://github.com/crowsonkb/v-diffusion-jax/blob/master/diffusion/models/danbooru_128.py#L8 """

    def __init__(self, dim, is_random = False):
        super().__init__()
        assert (dim % 2) == 0
        half_dim = dim // 2
        self.weights = nn.Parameter(torch.randn(half_dim), requires_grad = not is_random)

    def forward(self, x):
        x = rearrange(x, 'b -> b 1')
        freqs = x * rearrange(self.weights, 'd -> 1 d') * 2 * math.pi
        fouriered = torch.cat((freqs.sin(), freqs.cos()), dim = -1)
        fouriered = torch.cat((x, fouriered), dim = -1)
        return fouriered


class VBRCondition(nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.scale = nn.Conv2d(input_dim, output_dim, 1)
        self.shift = nn.Conv2d(input_dim, output_dim, 1)

    def forward(self, input, cond):
        cond = cond.reshape(-1, 1, 1, 1)
        scale = self.scale(cond)
        shift = self.shift(cond)
        return input * scale + shift


class GDN(nn.Module):
    """Generalized divisive normalization layer.
    y[i] = x[i] / sqrt(beta[i] + sum_j(gamma[j, i] * x[j]))
    """
    def __init__(self, ch, inverse=False, beta_min=1e-6, gamma_init=.1, reparam_offset=2**-18):
        super(GDN, self).__init__()
        self.inverse = inverse
        self.beta_min = beta_min
        self.gamma_init = gamma_init
        self.reparam_offset = reparam_offset

        self.build(ch)

    def build(self, ch):
        self.pedestal = self.reparam_offset**2
        self.beta_bound = (self.beta_min + self.reparam_offset**2)**.5
        self.gamma_bound = self.reparam_offset

        # Create beta param
        beta = torch.sqrt(torch.ones(ch) + self.pedestal)
        self.beta = nn.Parameter(beta)

        # Create gamma param
        eye = torch.eye(ch)
        g = self.gamma_init * eye
        g = g + self.pedestal
        gamma = torch.sqrt(g)

        self.gamma = nn.Parameter(gamma)
        self.pedestal = self.pedestal

    def forward(self, inputs):
        unfold = False
        if inputs.dim() == 5:
            unfold = True
            bs, ch, d, w, h = inputs.size()
            inputs = inputs.view(bs, ch, d * w, h)

        _, ch, _, _ = inputs.size()

        # Beta bound and reparam
        beta = LowerBound.apply(self.beta, self.beta_bound)
        beta = beta**2 - self.pedestal

        # Gamma bound and reparam
        gamma = LowerBound.apply(self.gamma, self.gamma_bound)
        gamma = gamma**2 - self.pedestal
        gamma = gamma.view(ch, ch, 1, 1)

        # Norm pool calc
        norm_ = nn.functional.conv2d(inputs**2, gamma, beta)
        norm_ = torch.sqrt(norm_)

        # Apply norm
        if self.inverse:
            outputs = inputs * norm_
        else:
            outputs = inputs / norm_

        if unfold:
            outputs = outputs.view(bs, ch, d, w, h)
        return outputs


class GDN1(GDN):
    def forward(self, inputs):
        unfold = False
        if inputs.dim() == 5:
            unfold = True
            bs, ch, d, w, h = inputs.size()
            inputs = inputs.view(bs, ch, d * w, h)

        _, ch, _, _ = inputs.size()

        # Beta bound and reparam
        beta = LowerBound.apply(self.beta, self.beta_bound)
        beta = beta ** 2 - self.pedestal

        # Gamma bound and reparam
        gamma = LowerBound.apply(self.gamma, self.gamma_bound)
        gamma = gamma ** 2 - self.pedestal
        gamma = gamma.view(ch, ch, 1, 1)

        # Norm pool calc
        norm_ = nn.functional.conv2d(torch.abs(inputs), gamma, beta)
        # norm_ = torch.sqrt(norm_)

        # Apply norm
        if self.inverse:
            outputs = inputs * norm_
        else:
            outputs = inputs / norm_

        if unfold:
            outputs = outputs.view(bs, ch, d, w, h)
        return outputs


class PriorFunction(nn.Module):
    #  A Custom Function described in Balle et al 2018. https://arxiv.org/pdf/1802.01436.pdf
    __constants__ = ['bias', 'in_features', 'out_features']

    def __init__(self, parallel_dims, in_features, out_features, scale, bias=True):
        super(PriorFunction, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = nn.Parameter(torch.Tensor(parallel_dims, 1, 1, in_features, out_features))
        if bias:
            self.bias = nn.Parameter(torch.Tensor(parallel_dims, 1, 1, 1, out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters(scale)

    def reset_parameters(self, scale):
        nn.init.constant_(self.weight, scale)
        if self.bias is not None:
            nn.init.uniform_(self.bias, -0.5, 0.5)

    def forward(self, input, detach=False):
        # input shape (channel, batch_size, in_features)
        # print("prior function input shape:", input.shape) torch.Size([64, 16, 4, 4, 1])
        if detach:
            return torch.matmul(input, F.softplus(self.weight.detach())) + self.bias.detach()
        return torch.matmul(input, F.softplus(self.weight)) + self.bias

    def extra_repr(self):
        return 'in_features={}, out_features={}, bias={}'.format(self.in_features, self.out_features, self.bias
                                                                 is not None)

class a_module(nn.Module):
    def __init__(self, channels, dims):
        super().__init__()
        self.param = nn.Parameter(torch.zeros(channels, 1, 1, 1, dims))
    def forward(self,):
        return self.param
        
class FlexiblePrior(nn.Module):
    '''
        A prior model described in Balle et al 2018 Appendix 6.1 https://arxiv.org/pdf/1802.01436.pdf
        return the boxshape likelihood
    '''
    def __init__(self, channels=256, dims=[3, 3, 3], init_scale=10., convert_module = False):
        super(FlexiblePrior, self).__init__()
        dims = [1] + dims + [1]
        self.chain_len = len(dims) - 1
        scale = init_scale**(1 / self.chain_len)
        h_b = []
        for i in range(self.chain_len):
            init = np.log(np.expm1(1 / scale / dims[i + 1]))
            h_b.append(PriorFunction(channels, dims[i], dims[i + 1], init))
        self.affine = nn.ModuleList(h_b)
        self.convert_module = convert_module
        if self.convert_module:
            self.a = nn.ModuleList([a_module(channels, dims[i + 1]) for i in range(self.chain_len - 1)])
        else:
            self.a = nn.ParameterList([nn.Parameter(torch.zeros(channels, 1, 1, 1, dims[i + 1])) for i in range(self.chain_len - 1)])

        # optimize the medians to fix the offset issue
        self._medians = nn.Parameter(torch.zeros(1, channels, 1, 1))
        # self.register_buffer('_medians', torch.zeros(1, channels, 1, 1))

    @property
    def medians(self):
        return self._medians.detach()
    
    def get_a(self,i):
        if self.convert_module:
            return self.a[i]()
        else:
            return self.a[i]
            

    def cdf(self, x, logits=True, detach=False):
        x = x.transpose(0, 1).unsqueeze(-1)  # C, N, H, W, 1
        if detach:
            for i in range(self.chain_len - 1):
                x = self.affine[i](x, detach)
                x = x + torch.tanh(self.get_a(i).detach()) * torch.tanh(x)
            if logits:
                return self.affine[-1](x, detach).squeeze(-1).transpose(0, 1)
            return torch.sigmoid(self.affine[-1](x, detach)).squeeze(-1).transpose(0, 1)

        # not detached
        for i in range(self.chain_len - 1):
            x = self.affine[i](x)
            x = x + torch.tanh(self.get_a(i)) * torch.tanh(x)
        if logits:
            return self.affine[-1](x).squeeze(-1).transpose(0, 1)
        return torch.sigmoid(self.affine[-1](x)).squeeze(-1).transpose(0, 1)

    def pdf(self, x):
        cdf = self.cdf(x, False)
        jac = torch.ones_like(cdf)
        pdf = torch.autograd.grad(cdf, x, grad_outputs=jac)[0]
        return pdf

    def get_extraloss(self):
        target = 0
        logits = self.cdf(self._medians, detach=True)
        extra_loss = torch.abs(logits - target).sum()
        return extra_loss

    def likelihood(self, x, min=1e-9):
        lower = self.cdf(x - 0.5, True)
        upper = self.cdf(x + 0.5, True)
        sign = -torch.sign(lower + upper).detach()
        upper = torch.sigmoid(upper * sign)
        lower = torch.sigmoid(lower * sign)
        return LowerBound.apply(torch.abs(upper - lower), min)

    def icdf(self, xi, method='bisection', max_iterations=1000, tol=1e-9, **kwargs):
        if method == 'bisection':
            init_interval = [-1, 1]
            left_endpoints = torch.ones_like(xi) * init_interval[0]
            right_endpoints = torch.ones_like(xi) * init_interval[1]

            def f(z):
                return self.cdf(z, logits=False, detach=True) - xi

            while True:
                if (f(left_endpoints) < 0).all():
                    break
                else:
                    left_endpoints = left_endpoints * 2
            while True:
                if (f(right_endpoints) > 0).all():
                    break
                else:
                    right_endpoints = right_endpoints * 2

            for i in range(max_iterations):
                mid_pts = 0.5 * (left_endpoints + right_endpoints)
                mid_vals = f(mid_pts)
                pos = mid_vals > 0
                non_pos = torch.logical_not(pos)
                neg = mid_vals < 0
                non_neg = torch.logical_not(neg)
                left_endpoints = left_endpoints * non_neg.float() + mid_pts * neg.float()
                right_endpoints = right_endpoints * non_pos.float() + mid_pts * pos.float()
                if (torch.logical_and(non_pos, non_neg)).all() or torch.min(right_endpoints - left_endpoints) <= tol:
                    print(f'bisection terminated after {i} its')
                    break

            return mid_pts
        else:
            raise NotImplementedError

    def sample(self, img, shape):
        uni = torch.rand(shape, device=img.device)
        return self.icdf(uni)
    
    def _pmf_to_cdf(self, pmf, tail_mass, pmf_length, max_length):
        cdf = torch.zeros(
            (len(pmf_length), max_length + 2), dtype=torch.int32, device=pmf.device
        )
        for i, p in enumerate(pmf):
            prob = torch.cat((p[: pmf_length[i]], tail_mass[i]), dim=0)
            _cdf = pmf_to_quantized_cdf(prob, 16)
            cdf[i, : _cdf.size(0)] = _cdf
        return cdf
    
            
    def _update(self, N=30, cdf_precision=16):
        with torch.no_grad():
            device = self._medians.device

            medians = self.medians[0, :, 0, 0]  # (C,) torch.float32

            minima = torch.full_like(medians, -N)  # (C,) torch.float32
            maxima = torch.full_like(medians, N)   # (C,) torch.float32

            pmf_length = (maxima - minima + 1)     # (C,) torch.float32
            pmf_length = pmf_length.to(torch.long) # (C,) torch.int64

            max_length = int(pmf_length.max().item())  # int

            samples = torch.arange(max_length, device=device)  # (L,) torch.int64
            samples = samples[None, :] + minima[:, None] + medians[:, None]  # (C, L) torch.float32
            samples = samples[None, ..., None]  # (1, C, L, 1) torch.float32

            pmf = self.likelihood(samples)      # (1, C, L, 1) torch.float32
            pmf = pmf[0, ..., 0]                # (C, L) torch.float32

            lower_m = medians + minima - 0.5    # (C,) torch.float32
            upper_m = medians + maxima + 0.5    # (C,) torch.float32

            lower_tail_mass = self.cdf(lower_m[None, ..., None, None], logits=False)  # (1, C, 1, 1) torch.float32
            upper_tail_mass = 1 - self.cdf(upper_m[None, ..., None, None], logits=False)  # (1, C, 1, 1) torch.float32

            tail_mass = lower_tail_mass + upper_tail_mass   # (1, C, 1, 1) torch.float32
            tail_mass = tail_mass[0, :, 0]                  # (C, 1) torch.float32

            # Compute quantized CDF
            quantized_cdf = self._pmf_to_cdf(pmf, tail_mass, pmf_length, max_length)  # (C, L+1) torch.int32

            # Save to self
            self._offset = minima                        # (C,) torch.float32
            self._quantized_cdf = quantized_cdf          # (C, L+1) torch.int32
            self._cdf_length = pmf_length + 2            # (C,) torch.int64

            return self._quantized_cdf, self._cdf_length, self._offset

        

