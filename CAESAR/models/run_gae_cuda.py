import torch
import numpy as np
import cupy as cp

import numpy as np
# from sklearn.decomposition import PCA
# import Huffman as huffman
from numpy import linalg as LA
from tqdm import tqdm
import argparse


from tqdm import tqdm

import time
import zstandard as zstd
import torch
from tqdm import tqdm

import nvidia.nvcomp as nvcomp
import cupy as cp

import os
import json


def save_json(json_pth, data, mode = "update"):
    # Load the existing JSON data
    if os.path.exists(json_pth):
        with open(json_pth, 'r') as json_file:
            existing_data = json.load(json_file)
            
        if mode == "update":
            existing_data.update(data)
            data = existing_data
        elif mode == "cat":
            for key in data:
                if key in existing_data:
                    existing_data[key] += data[key]
                else:
                    existing_data[key] = data[key]
            data = existing_data
        
    # Save the updated data back to the JSON file
    with open(json_pth, 'w') as json_file:
        json.dump(data, json_file, indent=4)
        

class PCA:
    def __init__(self, n_components=None, device='cuda'):
        self.n_components = n_components
        self.device = device
        self.components_ = None
        self.mean_ = None

    def fit(self, X):
        X = X.to(self.device)
        self.mean_ = torch.mean(X, dim=0)
        X_centered = X - self.mean_
        U, S, Vt = torch.linalg.svd(X_centered, full_matrices=False)
        if self.n_components is not None:
            Vt = Vt[:self.n_components, :]
        self.components_ = Vt
        del X_centered, U, S
        torch.cuda.empty_cache()
        
        return self

# def block2vector(block_data, patch_size=(8, 8)):
#     H, W = block_data.shape[-2:]
#     sp = block_data.shape[:-2]

#     n_h, n_w = H // patch_size[0], W // patch_size[1]

#     return block_data.view(*sp, n_h, patch_size[0], n_w, patch_size[1]) \
#                       .permute(*range(len(sp)), len(sp), len(sp)+2, len(sp)+1, len(sp)+3) \
#                       .reshape(-1, patch_size[0] * patch_size[1])

# def vector2block(vectors, shape, patch_size):
#     H, W = shape[-2:]
#     sp = shape[:-2]
#     n_h, n_w = H // patch_size[0], W // patch_size[1]
    
#     # Reshape back to block format
#     block_data = vectors.view(*sp, n_h, n_w, patch_size[0], patch_size[1]) \
#                          .permute(*range(len(sp)), len(sp), len(sp)+2, len(sp)+1, len(sp)+3) \
#                          .contiguous() \
#                          .view(*shape)
#     return block_data


def block2vector(block_data, patch_size=(8, 8)):
    # Get shape info
    
    *leading_dims, T, H, W = block_data.shape
    n_h, n_w = H // patch_size[0], W // patch_size[1]

    # Reshape and permute to extract patches
    vector_data = block_data.view(*leading_dims, T, n_h, patch_size[0], n_w, patch_size[1]) \
                             .permute(*range(len(leading_dims)), -4, -2, -5, -3, -1) \
                             .reshape(-1, patch_size[0] * patch_size[1])
    return vector_data

def vector2block(vectors, original_shape, patch_size=(8, 8)):
    *leading_dims,T, H, W = original_shape
    n_h, n_w = H // patch_size[0], W // patch_size[1]

    # Reshape vectors back to patch grid
    block_data = vectors.view(*leading_dims, n_h, n_w, T, patch_size[0], patch_size[1]) \
                         .permute(*range(len(leading_dims)), -3, -5, -2, -4, -1) \
                         .contiguous() \
                         .view(*original_shape)
    return block_data




def index_mask_prefix(arr_2d: torch.Tensor):
    
    num_cols     = arr_2d.size(1)
    reversed_arr = torch.flip(arr_2d, dims=[1])
    last_one_from_right = reversed_arr.int().argmax(dim=1)
    mask_length        = num_cols - last_one_from_right - 1
    mask = torch.arange(num_cols, device=arr_2d.device).unsqueeze(0) <= mask_length.unsqueeze(1)
    result = arr_2d[mask]
    
    return result, mask_length.to(torch.uint8)

def index_mask_reverse(prefix_mask, mask_length, num_cols):
    
    device = prefix_mask.device
    mask = torch.arange(num_cols, device=device).unsqueeze(0) <= mask_length.unsqueeze(1)
    arr_2d = torch.zeros((len(mask_length), num_cols), dtype=torch.bool, device=device)
    arr_2d[mask] = prefix_mask
    return arr_2d


def bits_to_bytes(bit_array):
    """Convert 0/1 numpy array to byte sequence"""
    if bit_array.dtype != np.uint8:
        bit_array = bit_array.astype(np.uint8)
    packed = np.packbits(bit_array)

    return packed.tobytes()


def bytes_to_bits(byte_seq, num_bits=None):
    """Convert byte sequence to 0/1 numpy array"""
    unpacked = np.unpackbits(np.frombuffer(byte_seq, dtype=np.uint8))
    if num_bits is not None:
        unpacked = unpacked[:num_bits]  # remove padding bits if we know original length
    return unpacked.astype(np.bool_)

class PCACompressor:
    def __init__(self, nrmse=-1, quan_factor=-1, device='cuda', codec_algorithm='Zstd', patch_size = (8, 8)):
        
        self.quan_bin = nrmse * quan_factor
        self.device = device
        self.codec_algorithm = codec_algorithm
        self.patch_size  = patch_size
        self.vector_size = patch_size[0] * patch_size[1]
        self.error_bound = nrmse * np.sqrt(self.vector_size)
        
    def compress(self, original_data, recons_data):
        input_shape = original_data.shape

        if isinstance(original_data, np.ndarray):
            original_data = torch.from_numpy(np.ascontiguousarray(original_data))
            recons_data   = torch.from_numpy(np.ascontiguousarray(recons_data))

        original_data = original_data.to(self.device, non_blocking=True)
        recons_data   = recons_data.to(self.device, non_blocking=True)

        if len(input_shape) == 2:
            assert(original_data.shape[1] == self.vector_size)
        else:
            original_data = block2vector(original_data, self.patch_size)
            recons_data   = block2vector(recons_data,   self.patch_size)

        residual_pca = original_data - recons_data
        norms = torch.linalg.norm(residual_pca, dim=1)
        process_mask = norms > self.error_bound

        if torch.sum(process_mask) <= 0:
            return {"data_bytes":0}, None, 0

        residual_pca = residual_pca[process_mask]
        pca = PCA(device=self.device)
        pca.fit(residual_pca)
        pca_basis = pca.components_
        all_coeff = residual_pca @ pca_basis.T

        # Delete immediately after no longer needed
        del original_data, recons_data, residual_pca
        torch.cuda.empty_cache()

        all_coeff_power = all_coeff.pow(2)
        sort_index = torch.argsort(all_coeff_power, dim=1, descending=True)

        all_coeff_sorted = torch.gather(all_coeff, 1, sort_index)
        quan_coeff_sorted = torch.round(all_coeff_sorted / self.quan_bin) * self.quan_bin
        res_coeff_sorted = all_coeff_sorted - quan_coeff_sorted

        del all_coeff_sorted  # No longer needed after this line
        torch.cuda.empty_cache()

        all_coeff_power_desc = torch.gather(all_coeff_power, 1, sort_index) - res_coeff_sorted.pow(2)
        step_errors = torch.ones_like(all_coeff_power_desc)
        remain_errors = torch.sum(all_coeff_power, dim=1)

        for i in range(step_errors.shape[1]):
            remain_errors = remain_errors - all_coeff_power_desc[:, i]
            step_errors[:, i] = remain_errors

        del all_coeff_power_desc, remain_errors  # done using both
        torch.cuda.empty_cache()

        mask = step_errors > self.error_bound ** 2
        del step_errors  # no longer needed after this
        torch.cuda.empty_cache()

        first_false_idx = torch.argmin(mask.int(), dim=1)
        mask[torch.arange(mask.shape[0]), first_false_idx] = True

        selected_coeff_q = quan_coeff_sorted * mask
        del quan_coeff_sorted  # done using it
        torch.cuda.empty_cache()

        selected_coeff_unsort_q = torch.zeros_like(selected_coeff_q)
        idx = torch.arange(selected_coeff_q.shape[0], device=self.device)[:, None]
        selected_coeff_unsort_q[idx, sort_index] = selected_coeff_q

        del selected_coeff_q, sort_index  # no longer needed after this
        torch.cuda.empty_cache()

        mask = selected_coeff_unsort_q != 0

        coeff_int_flatten = torch.round(all_coeff.reshape([-1])[mask.reshape(-1)] / self.quan_bin)
        unique_vals, coeff_int_flatten = torch.unique(coeff_int_flatten, return_inverse=True)
            
        del all_coeff  # final use of all_coeff here
        torch.cuda.empty_cache()

        prefix_mask_flatten, mask_length = index_mask_prefix(mask)
        del mask  # done using mask after this
        torch.cuda.empty_cache()

        meta_data = {
            "pca_basis": pca_basis,
            "unique_vals": unique_vals,
            "quan_bin": self.quan_bin,
            "n_vec": process_mask.shape[0],
            "prefix_length": prefix_mask_flatten.shape[0],
        }

        main_data = {
            "process_mask": process_mask,          # bool cuda
            "prefix_mask": prefix_mask_flatten,    # bool cuda
            "mask_length": mask_length,            # uint8 cuda
            "coeff_int": coeff_int_flatten         # int16 cuda
        }

        # Compress losslessly
        compressed_data, data_bytes = self.compress_lossless(meta_data, main_data)
        meta_data["data_bytes"] = data_bytes
        
        return meta_data, compressed_data, data_bytes


    def compress_lossless(self, meta_data, main_data):
        
        process_mask_uint8 = np.packbits(main_data["process_mask"].cpu().numpy().astype(np.uint8))
        prefix_mask_uint8 =  np.packbits(main_data["prefix_mask"].cpu().numpy().astype(np.uint8))
        
        data_bytes = meta_data["pca_basis"].cpu().numpy().nbytes + len(meta_data["unique_vals"])* 4
        n_vals = len(meta_data["unique_vals"])
        
        if n_vals<256:
            main_data["coeff_int"] = main_data["coeff_int"].to(torch.uint8)
            meta_data["coeff_dtype"] = '|u1'
        elif n_vals<32768:
            main_data["coeff_int"] = main_data["coeff_int"].to(torch.int16)
            meta_data["coeff_dtype"] = '<i2'
        else:
            main_data["coeff_int"] = main_data["coeff_int"].to(torch.int32)
            meta_data["coeff_dtype"] = '<i4'
        
        if self.device == "cpu":
            cctx = zstd.ZstdCompressor(level=21)
            
            compressed_data = {
                "process_mask": cctx.compress(process_mask_uint8.tobytes()),
                "prefix_mask":  cctx.compress(prefix_mask_uint8.tobytes()),
                "mask_length":  cctx.compress(main_data["mask_length"].cpu().numpy().tobytes()),
                "coeff_int":    cctx.compress(main_data["coeff_int"].cpu().numpy().tobytes()),
            }
            
            size_each = []
            for key in compressed_data.keys():
                size_each.append(len(compressed_data[key]))
                
        
        else:
            codec = nvcomp.Codec(algorithm=self.codec_algorithm)
            
            compressed_data = {
                "process_mask": codec.encode(cp.asarray(process_mask_uint8)),
                "prefix_mask":  codec.encode(cp.asarray(prefix_mask_uint8)),
                "mask_length":  codec.encode(cp.asarray(main_data["mask_length"])),
                "coeff_int":    codec.encode(cp.asarray(main_data["coeff_int"])),
            }
        
            size_each = []
            for key in compressed_data.keys():
                size_each.append(compressed_data[key].size)
        
        # coeff_int = main_data["mask_length"]- main_data["mask_length"].min()
        # encoded = encoding_unsign_integer(coeff_int)
        # size_each[-2]= (encoded/8)
        
        
        # encoded = encoding_unsign_integer(main_data["coeff_int"])
        # size_each[-1]= (encoded/8)
        
#         print(np.asarray(size_each)/size_each[-1])
        
        data_bytes += sum(size_each)
            
        
        return compressed_data, data_bytes

    def decompress_lossless(self, meta_data, compressed_data):

        codec = nvcomp.Codec(algorithm=self.codec_algorithm)
        process_mask = np.asarray(codec.decode(compressed_data["process_mask"], '|u1').cpu())
        
        process_mask = bytes_to_bits(process_mask, meta_data["n_vec"])
        process_mask = torch.from_numpy(np.ascontiguousarray(process_mask)).to(self.device, non_blocking=True)
        
        prefix_mask = np.asarray(codec.decode(compressed_data["prefix_mask"],'|u1').cpu())
        prefix_mask = bytes_to_bits(prefix_mask, meta_data["prefix_length"])
        prefix_mask = torch.from_numpy(np.ascontiguousarray(prefix_mask)).to(self.device, non_blocking=True)
        
        
        mask_length = torch.from_numpy(np.asarray(codec.decode(compressed_data["mask_length"], '|u1').cpu())).to(self.device, non_blocking=True)
        
        coeff_int = torch.from_numpy(np.asarray(codec.decode(compressed_data["coeff_int"], meta_data["coeff_dtype"]).cpu())).to(self.device, non_blocking=True)
        
#         minimal_bit = meta_data["minimal_bit"]
        
#         coeff_int = coeff_int.reshape([-1, minimal_bit])
#         shifts = torch.arange(minimal_bit-1, -1, -1, device=self.device, dtype=torch.int16)
#         coeff_int = torch.sum(coeff_int.to(torch.int16) << shifts, dim=-1)
        
        
        main_data = {
            "process_mask": process_mask,
            "prefix_mask": prefix_mask,
            "mask_length": mask_length,
            "coeff_int": coeff_int
        }
        
        return main_data
    
    def decompress_lossless_cpu(self, meta_data, compressed_data):
        
        dctx = zstd.ZstdDecompressor()
        
        process_mask = np.frombuffer(dctx.decompress(compressed_data["process_mask"]), dtype=np.uint8)
        process_mask = bytes_to_bits(process_mask, meta_data["n_vec"])
        
        prefix_mask = np.frombuffer(dctx.decompress(compressed_data["prefix_mask"]), dtype=np.uint8)
        prefix_mask = bytes_to_bits(prefix_mask, meta_data["prefix_length"])
        
        mask_length = np.frombuffer(dctx.decompress(compressed_data["mask_length"]), dtype=np.uint8)
        dtype_map= {"|u1":np.uint8,"<i2":np.int16,"<i4":int32}
        coeff_int   = np.frombuffer(dctx.decompress(compressed_data["coeff_int"]), dtype=dtype_map[meta_data["coeff_dtype"]])
                                    
        main_data = {
            "process_mask": torch.from_numpy(process_mask),
            "prefix_mask": torch.from_numpy(prefix_mask),
            "mask_length": torch.from_numpy(mask_length),
            "coeff_int": torch.from_numpy(coeff_int)
        }
        
        return main_data
    

    def decompress(self, recons_data, meta_data, compressed_data, to_np = True):
        
        
        input_shape = recons_data.shape
        
        if isinstance(recons_data, np.ndarray):
            recons_data   = torch.from_numpy(np.ascontiguousarray(recons_data))
        
        recons_data = recons_data.clone().to(self.device, non_blocking=True)
        
        if self.device == "cpu":
            main_data = self.decompress_lossless_cpu(meta_data, compressed_data)
        else:                      
            main_data = self.decompress_lossless(meta_data, compressed_data)

        index_mask = index_mask_reverse(main_data["prefix_mask"], main_data["mask_length"], meta_data["pca_basis"].shape[0])
        coeff = torch.zeros(index_mask.shape, dtype=torch.float32, device=self.device)
        coeff_int =meta_data["unique_vals"][main_data["coeff_int"].to(torch.int32)]
        
        coeff[index_mask] = coeff_int * self.quan_bin

        pca_basis = meta_data["pca_basis"]
        process_mask = main_data["process_mask"]
        
        if len(input_shape) == 2:
            assert(recons_data.shape[1] == self.vector_size)
        else:
            recons_data   = block2vector(recons_data,   self.patch_size)
        
        recons_data[process_mask] = recons_data[process_mask] + coeff @ pca_basis
        
        if len(input_shape) > 2:
            recons_data = vector2block(recons_data, input_shape, self.patch_size)
        
        recons_data_gae = recons_data.cpu()
        
        # del recons_data
        
        if to_np:
            return recons_data_gae.numpy()
        
        return recons_data_gae



        
def run_gae(args):
    
    data = np.load(args.path)
    
    original_data = data["original_data"]
    recons_data = data["recons_data"]
    data_size_bit = original_data.nbytes * 8
    
    assert(args.cr!=-1 or args.latent_bit!=-1 or "latent_bit" in data)
    
    
    if args.cr !=-1:
        args.latent_bit = data_size_bit/args.cr
    elif args.latent_bit==-1 and "latent_bit" in data:
        args.latent_bit = data["latent_bit"]
    
    
    
    print("Original/Recons Shape:", original_data.shape, recons_data.shape, original_data.dtype, recons_data.dtype)
    print("Original Data Size in Bits", data_size_bit ,  "Bits Per Pixel:", original_data.itemsize * 8)
    
    x_min, x_max, x_mean = original_data.min(), original_data.max(), original_data.mean()
    y_min, y_max, y_mean = recons_data.min(), recons_data.max(), recons_data.mean()
    
    original_data = (original_data-x_mean)/(x_max-x_min)
    recons_data = (recons_data-x_mean)/(x_max-x_min)
    
    init_nrmse = relative_rmse_error_ornl(original_data, recons_data)
    
    print("Init NRMSE:", init_nrmse, "Latent Bits:", args.latent_bit, "Original CR: %.2f"%(data_size_bit/args.latent_bit), original_data.dtype)
    
    vector_size = recons_data.shape[-1]
    
    all_nrmse = np.asarray(args.nrmse.split(","), dtype = np.float32)
    
    data_size_gbyte = recons_data.nbytes/(1024 ** 3)
    
    for nrmse in all_nrmse:
        
        save_path = args.path.replace(".npz","")+"_nrmse_%.6f.npz"%(nrmse) if args.save_path=="" else args.save_path
        all_algs = ["LZ4", "Snappy", "GDeflate", "Deflate", "Bitcomp", "ANS", "Zstd"]
        
        all_algs = ["Zstd"]
        
        for alg in all_algs: #,  "Cascaded"
            start_time = time.time()

            quan_factor = 2
    
            device = "cuda" if torch.cuda.is_available() else "cpu"
            # device = "cpu"
            compressor = PCACompressor(nrmse, quan_factor, codec_algorithm = alg, device = device)
            meta_data, compressed_data, data_bytes = compressor.compress(original_data, recons_data)
            
            
            encoding_end = time.time()

            if data_bytes>0:
                recons_data_gae =  compressor.decompress(recons_data, meta_data, compressed_data)
            else:
                recons_data_gae = recons_data
                
            decoding_end = time.time()
            

            nrmse_final = relative_rmse_error_ornl(original_data, recons_data_gae)
            CR = recons_data_gae.nbytes/(data_bytes+args.latent_bit/8)
            
            encoding_speed = data_size_gbyte/ (encoding_end - start_time)
            decoding_speed = data_size_gbyte/ (decoding_end - encoding_end)
            speed_overall = data_size_gbyte/ (decoding_end - start_time)

            print(alg, "Targe NRMSE: %.6f"%nrmse, "Final NRMSE: %.6f"%nrmse_final, "CR: %.1f"%(CR), "Processing Speed:  %.3f GB/s (%.3f || %.3f)"%(speed_overall, encoding_speed, decoding_speed))
            del meta_data, compressed_data
            torch.cuda.empty_cache()
            
        # print("------------------------------------")
        
        json_info = {"data_path":args.path, "cr":[float(CR)], "eb": [float(nrmse)], "nrmse":[float(nrmse_final)]}
        
        if  args.save_result:
            if args.save_original:
                np.savez(save_path, original_data = original_data*(x_max-x_min)+x_mean,
                         recons_data = recons_data_gae*(x_max-x_min)+x_mean, 
                         nrmse = final_nrsme, 
                         latent_bit = int(new_latent_bit))
            else:
                np.savez(save_path, recons_data = recons_data_gae*(x_max-x_min)+x_mean, nrmse = final_nrsme, latent_bit = int(new_latent_bit))

#         if args.json_path == "":
#             args.json_path = args.path.replace(".npz","")+".json"

#         save_json(args.json_path, json_info, "cat")
    
        
if __name__=="__main__":
    
    import argparse
    from metrics import *
    from Huffman2 import encoding_unsign_integer

    parser = argparse.ArgumentParser()

    parser.add_argument('--nrmse', type=str, default = "0.001,0.004,0.002,0.001,0.0005,0.0002,0.0001")
    parser.add_argument('--path', type=str, default = "/blue/ranka/shared-xiaoli/example.npz")
    parser.add_argument('--save_result', type=int, default = 0)
    parser.add_argument('--save_path', type=str, default = "")
    parser.add_argument('--latent_bit', type=int, default = -1)
    parser.add_argument('--cr', type=int, default = -1)
    parser.add_argument('--save_original', type=int, default = 0)
    parser.add_argument('--compute_cr', type=int, default = 1)
    parser.add_argument('--json_path', type=str, default = "")
    args = parser.parse_args()

    run_gae(args)
