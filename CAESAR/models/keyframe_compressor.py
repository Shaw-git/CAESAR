import torch.nn as nn
from .network_components import ResnetBlock, FlexiblePrior, Downsample, Upsample
from .utils import quantize, NormalDistribution
import time
import torch 
from .RangeEncoding import RangeCoder
import numpy as np
from .BCRN.bcrn_model import BluePrintConvNeXt_SR
import torch
import torch.nn.init as init

def load_yaml(file_path):
    with open(file_path, 'r') as file:
        data = yaml.safe_load(file)
    return data

def super_resolution_model(img_size = 64, in_chans=32, out_chans=1, sr_dim = "HAT", pretrain = False, sr_type = "BCRN"):
    
    if sr_type == "BCRN":
        sr_model = BluePrintConvNeXt_SR(in_chans, 1, 4, sr_dim)
        if pretrain:
            loaded_params, not_loaded_params = sr_model.load_part_model("./pretrain/BCRN_SRx4.pth")
        else:
            loaded_params, not_loaded_params = [], sr_model.parameters()
        print("Loading BCRN model")
        
        return sr_model, loaded_params, not_loaded_params
    

class Compressor(nn.Module):
    def __init__(
        self,
        dim=64,
        dim_mults=(1, 2, 3, 4),
        reverse_dim_mults=(4, 3, 2, 1),
        hyper_dims_mults=(4, 4, 4),
        channels=3,
        out_channels=3
    ):
        super().__init__()
        self.channels = channels
        self.out_channels = out_channels
        self.dims = [channels, *map(lambda m: dim * m, dim_mults)]
        self.in_out = list(zip(self.dims[:-1], self.dims[1:]))
        self.reversed_dims = [*map(lambda m: dim * m, reverse_dim_mults), out_channels]
        self.reversed_in_out = list(zip(self.reversed_dims[:-1], self.reversed_dims[1:]))
        assert self.dims[-1] == self.reversed_dims[0]
        self.hyper_dims = [self.dims[-1], *map(lambda m: dim * m, hyper_dims_mults)]
        self.hyper_in_out = list(zip(self.hyper_dims[:-1], self.hyper_dims[1:]))
        self.reversed_hyper_dims = list(
            reversed([self.dims[-1] * 2, *map(lambda m: dim * m, hyper_dims_mults)])
        )
        self.reversed_hyper_in_out = list(
            zip(self.reversed_hyper_dims[:-1], self.reversed_hyper_dims[1:])
        )
        self.prior = FlexiblePrior(self.hyper_dims[-1])
        
        self.range_coder = None
        
        self.measure_time = False   
        self.encoding_time = 0
        self.decoding_time = 0
        
    def set_timer(self):
        self.starter, self.ender = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        
    def get_extra_loss(self):
        return self.prior.get_extraloss()

    def build_network(self):
        self.enc = nn.ModuleList([])
        self.dec = nn.ModuleList([])
        self.hyper_enc = nn.ModuleList([])
        self.hyper_dec = nn.ModuleList([])
        
    def hyperprior(self, shape, latent):
        x = latent
        for i, (conv, act) in enumerate(self.hyper_enc):
            x = conv(x)
            x = act(x)
            
        hyper_latent = x
        
        q_hyper_latent = quantize(hyper_latent, "dequantize", self.prior.medians)
        
        x = q_hyper_latent
        
        for i, (deconv, act) in enumerate(self.hyper_dec):
            x = deconv(x)
            x = act(x)
        
        mean, scale = x.chunk(2, 1)
        latent_distribution = NormalDistribution(mean, scale.clamp(min=0.1))
        q_latent = quantize(latent, "dequantize", latent_distribution.mean)
        # q_latent = quantize(latent, "dequantize", mean.detach())
        
        state4bpp = {
            "latent": latent,
            "hyper_latent": hyper_latent,
            "latent_distribution": latent_distribution,
            "mean":mean,
            "scale": scale
        }
        return self.bpp(shape, state4bpp)
    
    def hyper_encode(self, latent):
        for i, (conv, act) in enumerate(self.hyper_enc):
            latent = conv(latent)
            latent = act(latent)
        hyper_latent = latent
        return hyper_latent
    
    
    def hyper_decode(self, q_hyper_latent):
        for i, (deconv, act) in enumerate(self.hyper_dec):
            q_hyper_latent = deconv(q_hyper_latent)
            q_hyper_latent = act(q_hyper_latent)
        mean, scale = q_hyper_latent.chunk(2, 1)
        return mean, scale
    
        
    def encode(self, x):
        for i, (resnet, down) in enumerate(self.enc):
            x = resnet(x)
            x = down(x)
        return x

    
    def decode(self, x):
        for i, (resnet, up) in enumerate(self.dec):
            x = resnet(x)
            x = up(x)
        return x
    
    def inference_qlatent(self, x):
        B,C,T,H,W = x.shape
        x = x.permute([0,2,1,3,4]).reshape([-1,C,H,W])
        
        latent = self.encode(x)
        hyper_latent = self.hyper_encode(latent)
        q_hyper_latent = quantize(hyper_latent, "dequantize", self.prior.medians)
        mean, scale = self.hyper_decode(q_hyper_latent)
        q_latent = quantize(latent, "dequantize", mean)
        q_latent = q_latent.view(B, T, *q_latent.shape[1:]).permute([0,2,1,3,4])
        return q_latent
        
    def bpp(self, shape, state4bpp):
        B, _, H, W = shape
        latent = state4bpp["latent"]
        hyper_latent = state4bpp["hyper_latent"]
        latent_distribution = NormalDistribution(state4bpp['mean'], state4bpp['scale'].clamp(min=0.1))
        
        if self.training:
            q_hyper_latent = quantize(hyper_latent, "noise")
            q_latent = quantize(latent, "noise")
        else:
            q_hyper_latent = quantize(hyper_latent, "dequantize", self.prior.medians)
            q_latent = quantize(latent, "dequantize", latent_distribution.mean)
            
        hyper_rate = -self.prior.likelihood(q_hyper_latent).log2()
        cond_rate = -latent_distribution.likelihood(q_latent).log2()
        # print(cond_rate.sum(dim=(0,1, 2, 3))/hyper_rate.sum(dim=(0, 1, 2, 3)))
        
        frame_bit_latent = cond_rate.sum(dim=(1, 2, 3))
        frame_bit_hyper_latent = hyper_rate.sum(dim=(1, 2, 3))
        
        # print("theory", frame_bit_latent[:10], frame_bit_hyper_latent[:10])
        
        frame_bit = (frame_bit_latent + frame_bit_hyper_latent)
        bpp = frame_bit / (H * W)
        
        return frame_bit, bpp
    

    def compress(self, x, return_latent = False, return_time = False):
        if self.range_coder is None:
            _quantized_cdf, _cdf_length, _offset = self.prior._update(30)
            self.range_coder = RangeCoder(_quantized_cdf = _quantized_cdf, _cdf_length= _cdf_length, _offset= _offset, medians = self.prior.medians.detach())
            
        B,C,T,H,W = x.shape
        original_shape = x.shape
        x = x.permute([0,2,1,3,4]).reshape([-1,C,H,W])
        
        if return_time:
            torch.cuda.synchronize()  # Wait for all GPU ops to finish
            start_time = time.time()

        latent = self.encode(x)
        hyper_latent = self.hyper_encode(latent)
        q_hyper_latent = quantize(hyper_latent, "dequantize", self.prior.medians)
        mean, scale = self.hyper_decode(q_hyper_latent)

        if return_time:
            torch.cuda.synchronize()  # Wait for all GPU ops to finish
            elapsed_time = time.time() - start_time
        
        latent_string = self.range_coder.compress(latent, mean, scale)
        hyper_latent_string = self.range_coder.compress_hyperlatent(hyper_latent)
        
        bpf_real = torch.Tensor([(len(lc)+len(hc))*8 for lc, hc in zip(latent_string, hyper_latent_string)])
        
        compressed_data = (latent_string, hyper_latent_string, original_shape, hyper_latent.shape)

        state4bpp = {"latent": latent, "hyper_latent": hyper_latent, "mean":mean, "scale": scale}
        
        bpf_theory, bpp = self.bpp(x.shape, state4bpp)
        
        result = {}
        
        result["bpf_entropy"] = bpf_theory
        result["compressed"] = compressed_data
        result["bpf_real"] = bpf_real

        
        if return_latent:
            q_latent = quantize(latent, "dequantize", mean)
            result["q_latent"] = q_latent.reshape(B,T,*q_latent.shape[-3:])
            
        if return_time:
            result["elapsed_time"] = elapsed_time
        
        return result
    
    def decompress(self, latent_string, hyper_latent_string, original_shape, hyper_shape, device = "cuda"):
        B, _, T, _, _ = original_shape
        q_hyper_latent = self.range_coder.decompress_hyperlatent(hyper_latent_string, hyper_shape)
        mean, scale = self.hyper_decode(q_hyper_latent.to(device))
        
        q_latent = self.range_coder.decompress(latent_string, mean.detach().cpu(), scale.detach().cpu())
        q_latent = q_latent.reshape(B, T, *q_latent.shape[-3:]).permute([0,2,1,3,4])
        
        return q_latent.to(device)
    
    
    def load_pretrain(self, path):
        device = next(self.parameters()).device
        checkpoint = torch.load(path, map_location=device)
        self.load_state_dict(checkpoint)
        print(f"Loaded pretrained weights from {path}")


    def forward(self, input, keep_dim = True):
        B,C,T,H,W = input.shape
        input = input.permute(0,2,1,3,4).reshape([-1,C,H,W])
        
        latent = self.encode(input)
        hyper_latent = self.hyper_encode(latent) 
        
        q_hyper_latent = quantize(hyper_latent, "dequantize", self.prior.medians)
        
        mean, scale = self.hyper_decode(q_hyper_latent)
        q_latent = quantize(latent, "dequantize", mean.detach())
        
        state4bpp = {"latent": latent, "hyper_latent":hyper_latent, "mean":mean, "scale":scale }
        frame_bit, bpp = self.bpp(input.shape, state4bpp)
        output = self.decode(q_latent)
        
        if keep_dim:
            output = output.reshape([B,T,*output.shape[1:]]).permute(0,2,1,3,4)
        
        return {
            "output": output,
            "bpp": bpp,
            "frame_bit":frame_bit,
            "q_latent": q_latent,
            "q_hyper_latent": q_hyper_latent,
            "latent":latent,
            "hyper_latent": hyper_latent,
            "mean": mean,
            "scale": scale
        }


class ResnetCompressor(Compressor):
    def __init__(
        self,
        dim=16,
        dim_mults=(1, 2, 3, 4),
        reverse_dim_mults=(4, 3, 2, 1),
        hyper_dims_mults=(4, 4, 4),
        channels=1,
        out_channels=1,
    ):
        super().__init__(
            dim,
            dim_mults,
            reverse_dim_mults,
            hyper_dims_mults,
            channels,
            out_channels
        )
        self.build_network()

    def build_network(self):

        self.enc = nn.ModuleList([])
        self.dec = nn.ModuleList([])
        self.hyper_enc = nn.ModuleList([])
        self.hyper_dec = nn.ModuleList([])

        for ind, (dim_in, dim_out) in enumerate(self.in_out):
            is_last = ind >= (len(self.in_out) - 1)
            self.enc.append(
                nn.ModuleList(
                    [
                        ResnetBlock(dim_in, dim_out, None, True if ind == 0 else False),
                        Downsample(dim_out),
                    ]
                )
            )

        for ind, (dim_in, dim_out) in enumerate(self.reversed_in_out):
            is_last = ind >= (len(self.reversed_in_out) - 1)
            self.dec.append(
                nn.ModuleList(
                    [
                        ResnetBlock(dim_in, dim_out if not is_last else dim_in),
                        Upsample(dim_out if not is_last else dim_in, dim_out),
                    ]
                )
            )

        for ind, (dim_in, dim_out) in enumerate(self.hyper_in_out):
            is_last = ind >= (len(self.hyper_in_out) - 1)
            self.hyper_enc.append(
                nn.ModuleList(
                    [
                        nn.Conv2d(dim_in, dim_out, 3, 1, 1)
                        if ind == 0
                        else nn.Conv2d(dim_in, dim_out, 5, 2, 2),
                        nn.LeakyReLU(0.2) if not is_last else nn.Identity(),
                    ]
                )
            )

        for ind, (dim_in, dim_out) in enumerate(self.reversed_hyper_in_out):
            is_last = ind >= (len(self.reversed_hyper_in_out) - 1)
            self.hyper_dec.append(
                nn.ModuleList(
                    [
                        nn.Conv2d(dim_in, dim_out, 3, 1, 1)
                        if is_last
                        else nn.ConvTranspose2d(dim_in, dim_out, 5, 2, 2, 1),
                        nn.LeakyReLU(0.2) if not is_last else nn.Identity(),
                    ]
                )
            )
            

            
def reshape_batch_2d_3d(batch_data, batch_size):
    BT,C,H,W = batch_data.shape
    T = BT//batch_size
    batch_data = batch_data.view([batch_size, T, C, H, W])
    batch_data = batch_data.permute([0,2,1,3,4])
    return batch_data

def reshape_batch_3d_2d(batch_data):
    B,C,T,H,W = batch_data.shape
    batch_data = batch_data.permute([0,2,1,3,4]).reshape([B*T,C,H,W])
    return batch_data
            
class CompressorSR(nn.Module):
    def __init__(
        self,
        dim=64,
        dim_mults=(1, 2, 3, 4),
        reverse_dim_mults=(4, 3, 2, 1),
        hyper_dims_mults=(4, 4, 4),
        channels=3,
        out_channels=3,
        sr_dim = 16
    ):
        super().__init__()  # Initialize the nn.Module parent class
        
        cpout_channels = dim * reverse_dim_mults[-1]
        
        self.compressor = ResnetCompressor(
            dim,
            dim_mults,
            reverse_dim_mults[:-1],
            hyper_dims_mults,
            channels,
            cpout_channels,
        )

        # Update channels for sr_model based on entropy_model's output

        # Initialize super-resolution model
        self.sr_model, self.loaded_params, self.not_loaded_params = super_resolution_model(
            img_size=64, in_chans=cpout_channels, out_chans=out_channels, sr_type = "BCRN", sr_dim = sr_dim
        )
    
    def forward(self, inputs):
        B = inputs.shape[0]
        
        results = self.compressor(inputs, keep_dim = False)
        outputs = results["output"]
        outputs = self.sr_model(outputs)  # Use self.sr_model
        
        # Reshape if needed
        outputs = reshape_batch_2d_3d(outputs, B)
        results["output"] = outputs
        
        return results
    
    
    def inference_qlatent(self,x):
        return self.compressor.inference_qlatent(x)
    
    def compress(self, x, return_latent = False, real = False):
        return self.compressor.compress(x, return_latent, real)
    
    def decompress(self, latent_string, hyper_latent_string, original_shape, hyper_shape, device = "cuda"):
        return self.compressor.decompress(latent_string, hyper_latent_string, original_shape, hyper_shape, device)
    
    def decode(self, x):
        x = self.compressor.decode(x)
        x = self.sr_model(x)
        return x
        
    
