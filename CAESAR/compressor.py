import torch
from collections import OrderedDict
from .models.run_gae_cuda import PCACompressor
import math
import torch.nn.functional as F
import numpy as np
import time
def normalize_latent(x):
    x_min = torch.amin(x, dim=(1, 2, 3, 4), keepdim=True)
    x_max = torch.amax(x, dim=(1, 2, 3, 4), keepdim=True)

    scale = (x_max - x_min + 1e-8) / 2
    offset = x_min + scale

    x_norm = (x - offset) / scale  # result in [-1, 1]
    return x_norm, offset, scale


class CAESAR:
    def __init__(self, 
                 model_path, 
                 use_diffusion=True, 
                 device='cuda',  n_frame = 16, interpo_rate=3, diffusion_steps = 32
                 ):
        self.pretrained_path = model_path
        self.use_diffusion = use_diffusion
        self.device = device
        self.n_frame = n_frame
        self.diffusion_steps = diffusion_steps

        self._load_models()
        
        self.interpo_rate = interpo_rate
        self.cond_idx = torch.arange(0,n_frame,interpo_rate)
        self.pred_idx = ~torch.isin(torch.arange(n_frame), self.cond_idx)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    

    def remove_module_prefix(self, state_dict):
        new_state_dict = OrderedDict()
        for k, v in state_dict.items():
            new_key = k.replace("module.", "")
            new_state_dict[new_key] = v
        return new_state_dict

    def _load_models(self):
        if not self.use_diffusion:
            self._load_caesar_v_compressor()
        else:
            self._load_caesar_d_compressor()

    def _load_caesar_v_compressor(self):
        from .models import compress_modules3d_mid_SR as compress_modules
        print("Loading CAESAE-V")
        model = compress_modules.CompressorMix(
            dim=16,
            dim_mults=[1, 2, 3, 4],
            reverse_dim_mults=[4, 3, 2],
            hyper_dims_mults=[4, 4, 4],
            channels=1,
            out_channels=1,
            d3=True,
            sr_dim=16
        )

        state_dict = self.remove_module_prefix(torch.load(self.pretrained_path, map_location=self.device))
        model.load_state_dict(state_dict)
        self.compressor_v = model.to(self.device).eval()

    def _load_caesar_d_compressor(self):
        print("Loading CAESAE-D")
        from .models import keyframe_compressor as compress_modules
        
        pretrained_models = torch.load(self.pretrained_path, map_location=self.device)
       
        model = compress_modules.ResnetCompressor(
            dim=16,
            dim_mults=[1, 2, 3, 4],
            reverse_dim_mults=[4, 3, 2, 1],
            hyper_dims_mults=[4, 4, 4],
            channels=1,
            out_channels=1
        )

        state_dict = self.remove_module_prefix(pretrained_models["vae"])
        model.load_state_dict(state_dict)
        self.keyframe_model = model.to(self.device).eval()
        
        
        from .models.video_diffusion_interpo import Unet3D, GaussianDiffusion
        model = Unet3D(
            dim=64,
            out_dim=64,
            channels=64,
            dim_mults=(1, 2, 4, 8),
            use_bert_text_cond=False
        )

        diffusion = GaussianDiffusion(
            model,
            image_size=16,
            num_frames=10,
            channels=64,
            timesteps=self.diffusion_steps,
            loss_type='l2'
        )
        
        state_dict = self.remove_module_prefix(pretrained_models["diffusion"])
        diffusion.load_state_dict(state_dict)

        self.diffusion_model = diffusion.to(self.device).eval()
        
        
    def compress(self, dataloader, eb = 1e-3):
        
        dataset_org = dataloader.dataset
        self.transform_shape = dataset_org.deblocking_hw
        
        shape = dataset_org.data_input.shape
        if self.use_diffusion:
            compressed_latent, latent_bytes = self.compress_caesar_d(dataloader)
            recons_data = self.decompress_caesar_d(compressed_latent, shape, dataset_org.filtered_blocks)
            recons_data = self.transform_shape(recons_data)

        else:
            compressed_latent, latent_bytes = self.compress_caesar_v(dataloader)

            recons_data = self.decompress_caesar_v(compressed_latent, shape, dataset_org.filtered_blocks)
            recons_data = self.transform_shape(recons_data)
  
        original_data = dataset_org.original_data()
        #print("original_data.shape after compress", original_data.shape, recons_data.shape)
        original_data, org_padding = self.padding(original_data)
        recons_data, rec_padding= self.padding(recons_data)
        
        meta_data, compressed_gae = self.postprocessing_encoding(original_data, recons_data, eb)
        return {"latent": compressed_latent, "postprocess": compressed_gae, "meta_data": meta_data, "shape": shape, "padding": rec_padding, "filtered_blocks": dataset_org.filtered_blocks}, latent_bytes + meta_data["data_bytes"]
    
    
    def decompress(self, compressed):

        shape = compressed["shape"]
        filtered_blocks = compressed["filtered_blocks"]
        if self.use_diffusion:
            recons_data = self.decompress_caesar_d(compressed["latent"], compressed["shape"], filtered_blocks)
            recons_data = self.transform_shape(recons_data)
        else:
            recons_data = self.decompress_caesar_v(compressed["latent"], compressed["shape"], filtered_blocks)
            recons_data = self.transform_shape(recons_data)
        
        recons_data, rec_padding= self.padding(recons_data) 

        recons_data = self.postprocessing_decoding(recons_data, compressed["meta_data"], compressed["postprocess"], rec_padding)
        return recons_data
            
            

    def compress_caesar_d(self, dataloader):

        total_bits = 0
        
        all_compressed_latent = []
        
        

        with torch.no_grad():
            for data in dataloader:
    
                keyframe = data["input"][:,:,self.cond_idx].cuda()
                outputs = self.keyframe_model.compress(keyframe)
                total_bits += torch.sum(outputs["bpf_real"])
                
                compressed_latent = {"compressed": outputs["compressed"],
                                     "scale": data["scale"],
                                     "offset": data["offset"],
                                     "index": data["index"]}
                
                all_compressed_latent.append(compressed_latent)
        return all_compressed_latent, total_bits/8
                
    def decompress_caesar_d(self, all_compressed, shape, filtered_blocks):
        
        torch.manual_seed(2025)
        torch.cuda.manual_seed_all(2025)
        
        
        recons_data = torch.zeros(shape)
        with torch.no_grad():
            for compressed in all_compressed:

                latent_data = self.keyframe_model.decompress(*compressed["compressed"], device = self.device)
                B,C,KT,H,W = latent_data.shape

                input_latent = torch.zeros([B, C, self.n_frame, H, W], device = self.device)
                input_latent[:,:,self.cond_idx] = latent_data
                input_latent,offset_latent, scale_latent = normalize_latent(input_latent)

                result = self.diffusion_model.sample(input_latent, self.interpo_rate, batch_size=input_latent.shape[0])
                input_latent[:,:,self.pred_idx] = result
                input_latent = input_latent*scale_latent + offset_latent

                input_latent = input_latent[:,:,:].permute(0,2,1,3,4).reshape(-1,64,16,16)

                rct_data = self.keyframe_model.decode(input_latent).detach()
                rct_data = rct_data.reshape([B, -1, 16, *rct_data.shape[-2:]])*compressed["scale"].cuda() + compressed["offset"].cuda()
                rct_data = rct_data.cpu()

                for i in range(B):
                    idx0, idx1, start_t, end_t = compressed["index"]
                    recons_data[idx0[i], idx1[i], start_t[i]:end_t[i]] = rct_data[i]
                    
        if filtered_blocks:
            V, S, T, H, W = shape
            n_frame = 16
            samples = T // n_frame
            for label, value in filtered_blocks:
                v = label // (S * samples)
                remain = label % (S * samples)
                s = remain // samples
                blk_idx = remain % samples
                start = blk_idx * n_frame
                end = (blk_idx + 1) * n_frame
                recons_data[v, s, start:end, :, :] = value

        return recons_data
                

    
    def decompress_caesar_v(self, all_compressed, shape, filtered_blocks):
        
        torch.manual_seed(2025)
        torch.cuda.manual_seed_all(2025)
        
        
        recons_data = torch.zeros(shape)
        with torch.no_grad():
            for compressed in all_compressed:
                
                    rct_data = self.compressor_v.decompress(*compressed["compressed"])
                    rct_data = rct_data*compressed["scale"].cuda() + compressed["offset"].cuda()
                    rct_data = rct_data.cpu()
                    
                    for i in range(rct_data.shape[0]):
                        idx0, idx1, start_t, end_t = compressed["index"]
                        recons_data[idx0[i], idx1[i], start_t[i]:end_t[i]] = rct_data[i]

        if filtered_blocks:
            V, S, T, H, W = shape
            n_frame = 8
            samples = T // n_frame
            for label, value in filtered_blocks:
                v = label // (S * samples)
                remain = label % (S * samples)
                s = remain // samples
                blk_idx = remain % samples
                start = blk_idx * n_frame
                end = (blk_idx + 1) * n_frame
                recons_data[v, s, start:end, :, :] = value
                
        return recons_data
    

    def compress_caesar_v(self, dataloader):

        total_bits = 0
        
        all_compressed_latent = []
        
        

        with torch.no_grad():
            for data in dataloader:
                outputs = self.compressor_v.compress(data["input"].cuda())
                total_bits += torch.sum(outputs["bpf_real"])
                
                compressed_latent = {"compressed": outputs["compressed"],
                                    "scale": data["scale"],
                                    "offset": data["offset"],
                                    "index": data["index"]}
                
                all_compressed_latent.append(compressed_latent)
                
        return all_compressed_latent, total_bits/8
                
                
    def padding(self, data, block_size=(8, 8)):
        *leading_dims, H, W = data.shape
        h_block, w_block = block_size

        H_target = math.ceil(H / h_block) * h_block
        W_target = math.ceil(W / w_block) * w_block
        dh = H_target - H
        dw = W_target - W
        top, down = dh // 2, dh - dh // 2
        left, right = dw // 2, dw - dw // 2

        data_reshaped = data.view(-1, H, W)
        data_padded = F.pad(data_reshaped, (left, right, top, down), mode='reflect')
        padded_data = data_padded.view(*leading_dims, *data_padded.shape[-2:])
        padding = (top, down, left, right)
        return padded_data, padding
    
    def unpadding(self, padded_data, padding):
        top, down, left, right = padding
        *leading_dims, H, W = padded_data.shape
        unpadded_data = padded_data[..., top:H-down, left:W-right]
        return unpadded_data
    
    def postprocessing_encoding(self, original_data, recons_data, nrmse):

        x_min, x_max, offset = original_data.min(), original_data.max(), original_data.mean()
        scale = (x_max-x_min)
        
        original_data = (original_data-offset)/scale
        recons_data = (recons_data-offset)/scale
        #self.device
        self.compressor = PCACompressor(nrmse, 2, codec_algorithm = "Zstd", device = self.device)
         
        meta_data, compressed_data, _ = self.compressor.compress(original_data, recons_data)    

        meta_data["scale"] = scale
        meta_data["offset"] = offset

        return meta_data, compressed_data
    
    def postprocessing_decoding(self, recons_data, meta_data, compressed_data, padding):
        
        recons_data = (recons_data - meta_data["offset"])/meta_data["scale"]
        
        # self.compressor = PCACompressor(0, 2, codec_algorithm = "Zstd", device = self.device)
      
        if meta_data["data_bytes"]>0:
            recons_data_gae =  self.compressor.decompress(recons_data, meta_data, compressed_data, to_np=False)
        else:
            recons_data_gae = recons_data
        
        recons_data_gae = self.unpadding(recons_data_gae, padding)
        return recons_data_gae* meta_data["scale"] + meta_data["offset"]
            
        
                    
        
        
