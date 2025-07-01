from torch import nn
from .blocks import blueprint_conv_layer, Blocks, ESA, pixelshuffle_block, activation, CCALayer
import torch
from bsconv.pytorch import BSConvU



class BluePrintShortcutBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3):
        super().__init__()
        self.conv = blueprint_conv_layer(in_channels, out_channels, kernel_size)
        self.convNextBlock = Blocks(out_channels, kernel_size)
        self.esa = ESA(out_channels, BSConvU)
        self.cca = CCALayer(out_channels)

    def forward(self, x):
        x = self.conv(x)
        x = self.convNextBlock(x)
        x = self.esa(x)
        x = self.cca(x)
        return x


class BluePrintConvNeXt_SR(nn.Module):
    def __init__(self, in_channels, out_channels, upscale_factor=2, base_channels = 64):
        super().__init__()
        self.conv1 = blueprint_conv_layer(in_channels, base_channels, 3)
        self.convNext1 = BluePrintShortcutBlock(base_channels, base_channels, 3)
        self.convNext2 = BluePrintShortcutBlock(base_channels, base_channels, 3)
        self.convNext3 = BluePrintShortcutBlock(base_channels, base_channels, 3)
        self.convNext4 = BluePrintShortcutBlock(base_channels, base_channels, 3)
        self.convNext5 = BluePrintShortcutBlock(base_channels, base_channels, 3)
        self.convNext6 = BluePrintShortcutBlock(base_channels, base_channels, 3)

        self.conv2 = blueprint_conv_layer(base_channels*6, base_channels, 3)
        self.upsample_block = pixelshuffle_block(base_channels, out_channels, upscale_factor)
        self.activation = activation(act_type='gelu')

    def forward(self, x):
        out_fea = self.conv1(x)
        out_C1 = self.convNext1(out_fea)
        out_C2 = self.convNext2(out_C1)
        out_C3 = self.convNext3(out_C2)
        out_C4 = self.convNext4(out_C3)
        out_C5 = self.convNext5(out_C4)
        out_C6 = self.convNext6(out_C5)

        out_C = self.activation(self.conv2(torch.cat([out_C1, out_C2, out_C3, out_C4, out_C5, out_C6], dim=1)))
        out_lr = out_C + out_fea
        output = self.upsample_block(out_lr)
        return output
    
    
    def load_part_model(self, pretrain_path):
        """
        Loads matching parameters from a pretrained model.
        
        Args:
            pretrain_path (str): Path to the pretrained model file.
        
        Returns:
            loaded_params (list): List of parameters in the current model loaded from the pretrained model.
            not_loaded_params (list): List of parameters in the current model that were not loaded.
            predefined_params (list): List of parameters in the pretrained model not used in the current model.
        """
        # Load the pretrained state dictionary
        pretrained_state = torch.load(pretrain_path, map_location='cpu')
        
        # Initialize lists to hold parameters
        loaded_params = []
        not_loaded_params = []
        predefined_params = []

        # Iterate over the model's named parameters
        for name, param in self.named_parameters():
            if name in pretrained_state and pretrained_state[name].shape == param.shape:
                # Load the parameter from the pretrained state
                param.data.copy_(pretrained_state[name])
                loaded_params.append(param)
            else:
                not_loaded_params.append(param)
                if name not in pretrained_state:
                    print(f"Parameter '{name}' not found in pretrained model.")
                else:
                    print(f"Shape mismatch for parameter '{name}': "
                          f"model {param.shape} vs pretrained {pretrained_state[name].shape}")

        # Collect predefined parameters in the pretrained model that are not in the current model
        for name in pretrained_state:
            if name not in self.state_dict():
                predefined_params.append(pretrained_state[name])
                print(f"Predefined parameter in pretrained model not used: {name}")

        return loaded_params, not_loaded_params