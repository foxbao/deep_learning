import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from attention import SelfAttention,CrossAttention
from ViT import ViT

class TimeEmbedding(nn.Module):
    def __init__(self, embedding_dim,device='cuda'):
        super(TimeEmbedding, self).__init__()
        self.embedding_dim = embedding_dim
        self.device=device

    def forward(self, t):
        half_dim = self.embedding_dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, dtype=torch.float32,device=self.device) * -emb)
        emb = t[:, None] * emb[None, :]
        emb = torch.cat((torch.sin(emb), torch.cos(emb)), dim=1)
        return emb

class Unet(nn.Module):
    # cfeat - context features
    def __init__(self, in_channels, layout_channels,n_feat=256, n_cfeat=10, height=28, time_dim=128,device='cuda'):
        super(Unet, self).__init__()

        # number of input channels, number of intermediate feature maps and number of classes
        self.in_channels = in_channels
        self.n_feat = n_feat
        self.n_cfeat = n_cfeat
        # assume h == w. must be divisible by 4, so 28,24,20,16...
        self.h = height
        self.device=device

        # Initialize the initial convolutional layer
        self.init_conv = ResidualConvBlock(in_channels, n_feat, is_res=True)
        self.init_conv_layout = ResidualConvBlock(
            layout_channels, n_feat, is_res=True)
        # Initialize the down-sampling path of the U-Net with two levels

        # (Batch_Size, 3, Height , Width)->(Batch_Size, (1+4)*n_feat, Height , Width)
        # self.layout_embed = LayoutEmbed(in_channels, n_feat, self.h)
        self.final = UNET_OutputLayer((1+4)*n_feat, in_channels)
        self.encoders = nn.ModuleList([
            SwitchSequential(LayoutEmbed(in_channels, layout_channels,n_feat, self.h)),
            # (Batch_Size, (1+4)*n_feat, Height , Width)->(Batch_Size, n_feat, Height, Width)
            SwitchSequential(UNET_ResidualBlock((1+4)*n_feat, n_feat, time_dim),
                             nn.Conv2d(1 * n_feat, 1 * n_feat, kernel_size=3, stride=2, padding=1)),

            SwitchSequential(UNET_ResidualBlock(1*n_feat, 2*n_feat, time_dim),
                             nn.Conv2d(2 * n_feat, 2 * n_feat, kernel_size=3, stride=2, padding=1)),

            SwitchSequential(UNET_ResidualBlock(2*n_feat, 4*n_feat, time_dim),
                             nn.Conv2d(4 * n_feat, 4 * n_feat, kernel_size=3, stride=2, padding=1)),
        ])
        self.bottleneck = SwitchSequential(
            # (Batch_Size, 1280, Height / 64, Width / 64) -> (Batch_Size, 1280, Height / 64, Width / 64)
            # UNET_ResidualBlock(4 * n_feat, 4 * n_feat, n_time=time_dim),
            # UNET_ResidualBlock(4 * n_feat, 4 * n_feat, n_time=time_dim),
            nn.Sequential(nn.AvgPool2d((4)), nn.GELU()),
            nn.Sequential(
                # nn.ConvTranspose2d(2 * n_feat, 2 * n_feat, self.h//4, self.h//4), # up-sample
                nn.ConvTranspose2d(4 * n_feat, 4 * n_feat, 4, 4),  # up-sample
                nn.GroupNorm(8, 4 * n_feat),  # normalize
                nn.ReLU(),
            )
        )

        self.decoders = nn.ModuleList([
            # (Batch_Size, n_feat, Height/8, Width/8)->(Batch_Size, 2*n_feat, Height/4, Width/4)
            SwitchSequential(UNET_ResidualBlock(8 * n_feat, 2*n_feat, time_dim),
                             nn.ConvTranspose2d(2*n_feat, 2*n_feat, 2, 2)),
            # (Batch_Size, 2*n_feat, Height/4, Width/4)->(Batch_Size, 2*n_feat, Height/4, Width/4)
            SwitchSequential(UNET_ResidualBlock(
                4 * n_feat, 1*n_feat, time_dim), nn.ConvTranspose2d(1*n_feat, 1*n_feat, 2, 2)),
            # (Batch_Size, 2*n_feat, Height/4, Width/4)->(Batch_Size, 2*n_feat, Height/2, Width/2)
            SwitchSequential(UNET_ResidualBlock(2 * n_feat, 1*n_feat, n_time=time_dim),
                             nn.ConvTranspose2d(1*n_feat, 1*n_feat, 2, 2)),

            SwitchSequential(UNET_ResidualBlock(
                6 * n_feat, 5*n_feat, n_time=time_dim)),

        ])

    def forward(self, x, t, c=None, layout_concate=None):
        """
        x : (batch, n_feat, h, w) : input image
        t : (batch, n_cfeat)      : time step
        c : (batch, n_classes)    : context label
        """
        if c is None:
            c = torch.zeros(x.shape[0], 77, 768).to(self.device)
            pass

        if layout_concate is None:
            layout_concate = torch.zeros(x.shape).to(self.device)

        # x = self.layout_embed(x=x, layout=layout_concate)
        residue = x
        skip_connections = []
        for layers in self.encoders:
            x = layers(x=x, context=c, time=t, layout=layout_concate)
            skip_connections.append(x)

        x = self.bottleneck(x=x, context=c, time=t)
        for layers in self.decoders:
            x = torch.cat((x, skip_connections.pop()), dim=1)
            x = layers(x=x, context=c, time=t)

        out = self.final(x)
        return out


class UNET_OutputLayer(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.groupnorm = nn.GroupNorm(32, in_channels)
        self.conv = nn.Conv2d(in_channels, out_channels,
                              kernel_size=3, padding=1)

    def forward(self, x):
        # x: (Batch_Size, 320, Height / 8, Width / 8)

        # (Batch_Size, 320, Height / 8, Width / 8) -> (Batch_Size, 320, Height / 8, Width / 8)
        x = self.groupnorm(x)

        # (Batch_Size, 320, Height / 8, Width / 8) -> (Batch_Size, 320, Height / 8, Width / 8)
        x = F.silu(x)

        # (Batch_Size, 320, Height / 8, Width / 8) -> (Batch_Size, 4, Height / 8, Width / 8)
        x = self.conv(x)

        # (Batch_Size, 4, Height / 8, Width / 8)
        return x
    
class ResidualConvBlock(nn.Module):
    def __init__(
        self, in_channels: int, out_channels: int, is_res: bool = False
    ) -> None:
        super().__init__()

        # Check if input and output channels are the same for the residual connection
        self.same_channels = in_channels == out_channels

        # Flag for whether or not to use residual connection
        self.is_res = is_res

        # First convolutional layer
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, 1, 1),   # 3x3 kernel with stride 1 and padding 1
            nn.BatchNorm2d(out_channels),   # Batch normalization
            nn.GELU(),   # GELU activation function
        )

        # Second convolutional layer
        self.conv2 = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, 3, 1, 1),   # 3x3 kernel with stride 1 and padding 1
            nn.BatchNorm2d(out_channels),   # Batch normalization
            nn.GELU(),   # GELU activation function
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:

        # If using residual connection
        if self.is_res:
            # Apply first convolutional layer
            x1 = self.conv1(x)

            # Apply second convolutional layer
            x2 = self.conv2(x1)

            # If input and output channels are the same, add residual connection directly
            if self.same_channels:
                out = x + x2
            else:
                # If not, apply a 1x1 convolutional layer to match dimensions before adding residual connection
                shortcut = nn.Conv2d(x.shape[1], x2.shape[1], kernel_size=1, stride=1, padding=0).to(x.device)
                out = shortcut(x) + x2
            #print(f"resconv forward: x {x.shape}, x1 {x1.shape}, x2 {x2.shape}, out {out.shape}")

            # Normalize output tensor
            return out / 1.414

        # If not using residual connection, return output of second convolutional layer
        else:
            x1 = self.conv1(x)
            x2 = self.conv2(x1)
            return x2

    # Method to get the number of output channels for this block
    def get_out_channels(self):
        return self.conv2[0].out_channels

    # Method to set the number of output channels for this block
    def set_out_channels(self, out_channels):
        self.conv1[0].out_channels = out_channels
        self.conv2[0].in_channels = out_channels
        self.conv2[0].out_channels = out_channels

class UNET_AttentionBlock(nn.Module):
    def __init__(self, n_head: int, n_embd: int, d_context=768):
        super().__init__()
        channels = n_head * n_embd
        
        self.groupnorm = nn.GroupNorm(32, channels, eps=1e-6)
        self.conv_input = nn.Conv2d(channels, channels, kernel_size=1, padding=0)

        self.layernorm_1 = nn.LayerNorm(channels)
        self.attention_1 = SelfAttention(n_head, channels, in_proj_bias=False)
        self.layernorm_2 = nn.LayerNorm(channels)
        self.attention_2 = CrossAttention(n_head, channels, d_context, in_proj_bias=False)
        self.layernorm_3 = nn.LayerNorm(channels)
        self.linear_geglu_1  = nn.Linear(channels, 4 * channels * 2)
        self.linear_geglu_2 = nn.Linear(4 * channels, channels)

        self.conv_output = nn.Conv2d(channels, channels, kernel_size=1, padding=0)
    
    def forward(self, x: torch.Tensor, context):
        # x: (Batch_Size, Features, Height, Width)
        # context: (Batch_Size, Seq_Len, Dim)

        residue_long = x

        # (Batch_Size, Features, Height, Width) -> (Batch_Size, Features, Height, Width)
        x = self.groupnorm(x)
        
        # (Batch_Size, Features, Height, Width) -> (Batch_Size, Features, Height, Width)
        x = self.conv_input(x)
        
        n, c, h, w = x.shape
        
        # (Batch_Size, Features, Height, Width) -> (Batch_Size, Features, Height * Width)
        x = x.view((n, c, h * w))
        
        # (Batch_Size, Features, Height * Width) -> (Batch_Size, Height * Width, Features)
        x = x.transpose(-1, -2)
        
        # Normalization + Self-Attention with skip connection

        # (Batch_Size, Height * Width, Features)
        residue_short = x
        
        # (Batch_Size, Height * Width, Features) -> (Batch_Size, Height * Width, Features)
        x = self.layernorm_1(x)
        
        # (Batch_Size, Height * Width, Features) -> (Batch_Size, Height * Width, Features)
        x = self.attention_1(x)
        
        # (Batch_Size, Height * Width, Features) + (Batch_Size, Height * Width, Features) -> (Batch_Size, Height * Width, Features)
        x += residue_short
        
        # (Batch_Size, Height * Width, Features)
        residue_short = x

        # Normalization + Cross-Attention with skip connection
        
        # (Batch_Size, Height * Width, Features) -> (Batch_Size, Height * Width, Features)
        x = self.layernorm_2(x)
        
        # (Batch_Size, Height * Width, Features) -> (Batch_Size, Height * Width, Features)
        x = self.attention_2(x, context)
        
        # (Batch_Size, Height * Width, Features) + (Batch_Size, Height * Width, Features) -> (Batch_Size, Height * Width, Features)
        x += residue_short
        
        # (Batch_Size, Height * Width, Features)
        residue_short = x

        # Normalization + FFN with GeGLU and skip connection
        
        # (Batch_Size, Height * Width, Features) -> (Batch_Size, Height * Width, Features)
        x = self.layernorm_3(x)
        
        # GeGLU as implemented in the original code: https://github.com/CompVis/stable-diffusion/blob/21f890f9da3cfbeaba8e2ac3c425ee9e998d5229/ldm/modules/attention.py#L37C10-L37C10
        # (Batch_Size, Height * Width, Features) -> two tensors of shape (Batch_Size, Height * Width, Features * 4)
        x, gate = self.linear_geglu_1(x).chunk(2, dim=-1) 
        
        # Element-wise product: (Batch_Size, Height * Width, Features * 4) * (Batch_Size, Height * Width, Features * 4) -> (Batch_Size, Height * Width, Features * 4)
        x = x * F.gelu(gate)
        
        # (Batch_Size, Height * Width, Features * 4) -> (Batch_Size, Height * Width, Features)
        x = self.linear_geglu_2(x)
        
        # (Batch_Size, Height * Width, Features) + (Batch_Size, Height * Width, Features) -> (Batch_Size, Height * Width, Features)
        x += residue_short
        
        # (Batch_Size, Height * Width, Features) -> (Batch_Size, Features, Height * Width)
        x = x.transpose(-1, -2)
        
        # (Batch_Size, Features, Height * Width) -> (Batch_Size, Features, Height, Width)
        x = x.view((n, c, h, w))

        # Final skip connection between initial input and output of the block
        # (Batch_Size, Features, Height, Width) + (Batch_Size, Features, Height, Width) -> (Batch_Size, Features, Height, Width)
        return self.conv_output(x) + residue_long
        

class UnetUp(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(UnetUp, self).__init__()
        
        # Create a list of layers for the upsampling block
        # The block consists of a ConvTranspose2d layer for upsampling, followed by two ResidualConvBlock layers
        layers = [
            nn.ConvTranspose2d(in_channels, out_channels, 2, 2),
            ResidualConvBlock(out_channels, out_channels),
            ResidualConvBlock(out_channels, out_channels),
        ]
        
        # Use the layers to create a sequential model
        self.model = nn.Sequential(*layers)

    def forward(self, x, skip):
        # Concatenate the input tensor x with the skip connection tensor along the channel dimension
        x = torch.cat((x, skip), 1)
        
        # Pass the concatenated tensor through the sequential model and return the output
        x = self.model(x)
        return x

    
class UnetDown(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(UnetDown, self).__init__()
        
        # Create a list of layers for the downsampling block
        # Each block consists of two ResidualConvBlock layers, followed by a MaxPool2d layer for downsampling
        layers = [ResidualConvBlock(in_channels, out_channels), ResidualConvBlock(out_channels, out_channels), nn.MaxPool2d(2)]
        
        # Use the layers to create a sequential model
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        # Pass the input through the sequential model and return the output
        return self.model(x)
    
class UNET_ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, n_time=128):
        super().__init__()
        self.groupnorm_feature = nn.GroupNorm(32, in_channels)
        self.conv_feature = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.linear_time = nn.Linear(n_time, out_channels)
        # self.timeembed1 = EmbedFC(1, out_channels)
        self.groupnorm_merged = nn.GroupNorm(32, out_channels)
        self.conv_merged = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.n_time=n_time
        if in_channels == out_channels:
            self.residual_layer = nn.Identity()
        else:
            self.residual_layer = nn.Conv2d(in_channels, out_channels, kernel_size=1, padding=0)
    
    def forward(self, feature, time):
        # feature: (Batch_Size, In_Channels, Height, Width)
        # time: (1, 1280)

        residue = feature
        
        # (Batch_Size, In_Channels, Height, Width) -> (Batch_Size, In_Channels, Height, Width)
        feature = self.groupnorm_feature(feature)
        
        # (Batch_Size, In_Channels, Height, Width) -> (Batch_Size, In_Channels, Height, Width)
        feature = F.silu(feature)
        
        # (Batch_Size, In_Channels, Height, Width) -> (Batch_Size, Out_Channels, Height, Width)
        feature = self.conv_feature(feature)
        
        # (1, 1280) -> (1, 1280)
        time = F.silu(time)
        time = time.view(-1, self.n_time)
        # (1, 1280) -> (1, Out_Channels)
        time = self.linear_time(time)
        
        # Add width and height dimension to time. 
        # (Batch_Size, Out_Channels, Height, Width) + (1, Out_Channels, 1, 1) -> (Batch_Size, Out_Channels, Height, Width)
        merged = feature + time.unsqueeze(-1).unsqueeze(-1)
        
        # (Batch_Size, Out_Channels, Height, Width) -> (Batch_Size, Out_Channels, Height, Width)
        merged = self.groupnorm_merged(merged)
        
        # (Batch_Size, Out_Channels, Height, Width) -> (Batch_Size, Out_Channels, Height, Width)
        merged = F.silu(merged)
        
        # (Batch_Size, Out_Channels, Height, Width) -> (Batch_Size, Out_Channels, Height, Width)
        merged = self.conv_merged(merged)
        
        # (Batch_Size, Out_Channels, Height, Width) + (Batch_Size, Out_Channels, Height, Width) -> (Batch_Size, Out_Channels, Height, Width)
        return merged + self.residual_layer(residue)

class SwitchSequential(nn.Sequential):
    def forward(self, x, context=None, time=None,layout=None):
        for layer in self:
            if isinstance(layer,LayoutEmbed):
                x = layer(x, layout)
            # elif isinstance(layer, UnetResTime):
            #     x = layer(x, time)
            elif isinstance(layer, UNET_ResidualBlock):
                x = layer(x, time)
            elif isinstance(layer, UNET_AttentionBlock):
                x = layer(x, context)
            else:
                x = layer(x)
        return x
    


class LayoutEmbed(nn.Module):
    def __init__(self, in_channels,layout_in_channels, n_feat, h):
        super(LayoutEmbed, self).__init__()
        self.h = h
        self.init_conv = ResidualConvBlock(in_channels, n_feat, is_res=True)
        self.init_conv_layout = ResidualConvBlock(
            layout_in_channels, n_feat, is_res=True)
        self.vitembedConcate = ViT(
            image_size=(n_feat, self.h, self.h), out_channels=4 * n_feat, patch_size=4, concate_mode=True
        )

    def forward(self, x, layout):
        x = self.init_conv(x)
        layout = self.init_conv_layout(layout)
        vembedConcate = self.vitembedConcate(layout)
        x = torch.concat((x, vembedConcate), dim=1)
        return x
    
class Diffusion(nn.Module):
    def __init__(self,in_channels, layout_channels,n_feat=256, n_cfeat=10, height=28, time_dim=320):
        super(Diffusion, self).__init__()
        self.time_embedding = TimeEmbedding(time_dim)
        self.unet=Unet(in_channels, layout_channels,n_feat, n_cfeat, height, time_dim)
        
    def forward(self, latent, layout, context,time):
        time_emb = self.time_embedding(time/1.0)
        pred_noise=self.unet(x=latent, t=time_emb, c=context, layout_concate=layout)
        return pred_noise
        