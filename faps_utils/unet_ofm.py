import torch
from faps_utils.unet_nD import UNetModelWrapper

"""
Simplified version using only the UNet backbone.
Time-conditioned UNet model for flow matching.
"""


class UNet_cond(torch.nn.Module):
    def __init__(self, dims, hidden_channels, conds_channels=1,
                 t_scaling=1, num_res_blocks=1, num_heads=8,
                 attention_res="16", in_channels=None, channel_mult=None):
        super(UNet_cond, self).__init__()
        
        self.t_scaling = t_scaling
        self.dims = tuple(dims)
        
        # Get input channels from dims[0] if not specified
        if in_channels is None:
            in_channels = dims[0]
        
        self.in_channels = in_channels
        
        # UNet expects input with hidden_channels
        # dims format: [channels, *spatial_dims]
        self.unet_dims = tuple([hidden_channels, *dims[1:]])

        # Projection layer: in_channels -> hidden_channels
        if in_channels != hidden_channels:
            self.input_proj = torch.nn.Conv3d(
                in_channels, 
                hidden_channels, 
                kernel_size=1, 
                stride=1, 
                padding=0
            )
        else:
            self.input_proj = None
            
        # Output projection layer: hidden_channels -> in_channels
        if hidden_channels != in_channels:
            self.output_proj = torch.nn.Conv3d(
                hidden_channels, 
                in_channels, 
                kernel_size=1, 
                stride=1, 
                padding=0
            )
        else:
            self.output_proj = None

        # Initialize UNet backbone only
        self.unet_backbone = UNetModelWrapper(
            dim=self.unet_dims, 
            num_channels=hidden_channels, 
            num_res_blocks=num_res_blocks,
            num_heads=num_heads, 
            num_conds=conds_channels, 
            set_cond=False,
            attention_resolutions=attention_res,
            channel_mult=channel_mult
        )
        
    def forward(self, t, u, conds):
        """
        Forward pass using only the UNet backbone.
        
        Args:
            u: (batch_size, channels, *spatial_dims)
            t: either scalar or (batch_size,)
            conds: (batch_size, num_conds)
            
        Returns:
            Processed output with same shape as input
        """
        t = t / self.t_scaling
        
        if t.dim() == 0 or t.numel() == 1:
            t = torch.ones(u.shape[0], device=t.device) * t

        assert t.dim() == 1
        assert t.shape[0] == u.shape[0]
        assert conds.dim() == 2

        # Project input to hidden_channels if needed
        if self.input_proj is not None:
            u = self.input_proj(u)
        
        # Use UNet backbone
        out = self.unet_backbone(t, u, conds)
        
        # Project output back to in_channels if needed
        if self.output_proj is not None:
            out = self.output_proj(out)
        
        return out