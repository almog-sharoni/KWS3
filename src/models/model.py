import torch.nn as nn
import sys
# ## import Mamba from mamba/mamba_ssm/modules/mamba_simple.py
# sys.path.append('mambaPy/mamba.py')
# # from mamba_simple import MambaQuantized as Mamba
# # import mambapy
# from mambapy.mamba import MambaBlock as Mamba
# from mambapy.mamba import MambaConfig
from mambapy.mamba import MambaBlock as Mamba
from mambapy.mamba import MambaConfig

import torch

class KeywordSpottingModel_with_cls(nn.Module):
    def __init__(self, input_dim, d_model, d_state, d_conv, expand, label_names, num_mamba_layers=1, dropout_rate=0.2):
        super(KeywordSpottingModel_with_cls, self).__init__()
        
        # Initial projection layer
        self.proj = nn.Linear(input_dim, d_model)  
        
        # CLS token: learnable parameter with shape [1, 1, d_model]
        self.cls_token = nn.Parameter(torch.zeros(1, 1, d_model))
        
        # Stack multiple Mamba layers with RMSNorm layer
        self.mamba_layers = nn.ModuleList()
        self.layer_norms = nn.ModuleList()

        mamba_config = MambaConfig(d_model=d_model, n_layers=1, d_state=d_state, expand_factor=expand, d_conv=d_conv)

        for _ in range(num_mamba_layers):
            self.mamba_layers.append(Mamba(mamba_config))
            self.layer_norms.append(nn.LayerNorm(normalized_shape=d_model, eps=1e-5))

        # Output layer
        self.fc = nn.Linear(d_model, len(label_names))  
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, x):
        # Reshape to [batch_size, num_frames, num_mfcc]
        x = x.permute(0, 2, 1)
        
        # Project input to d_model dimension
        x = self.proj(x)  
        
        # Create a CLS token and expand it across the batch dimension
        batch_size = x.size(0)
        cls_tokens = self.cls_token.expand(batch_size, -1, -1)  # Shape: [batch_size, 1, d_model]
        
        # Append the CLS token to the input sequence
        x = torch.cat((x, cls_tokens), dim=1)  # Shape: [batch_size, num_frames + 1, d_model]
        x = x.permute(0, 2, 1)  # Transpose to [batch_size, d_model, num_frames + 1] for Mamba
        
        # Pass through Mamba layers and layer normalization
        for mamba_layer, layer_norm in zip(self.mamba_layers, self.layer_norms):
            x = mamba_layer(x)
            x = layer_norm(x)  # Apply RMSNorm after Mamba layer

        x = self.dropout(x)  # Apply dropout after Mamba layers
        
        # Extract the CLS token output (last token)
        cls_output = x[:, :, -1]  # Shape: [batch_size, d_model]
        
        # Pass through the output layer
        x = self.fc(cls_output)
        
        return x