# coding: utf-8
"""
Enhanced SPADE decoder(G) with advanced features for generating high-quality animated images from warped features.
"""
import torch
from torch import nn
import torch.nn.functional as F
from .util import SPADEResnetBlock

class EnhancedSPADEDecoder(nn.Module):
    def __init__(self, upscale=1, max_features=256, block_expansion=64, out_channels=64, num_down_blocks=2,
                 num_middle_blocks=6, use_attention=True, use_self_attention=True, dropout_rate=0.2):
        super().__init__()
        self.upscale = upscale
        self.use_attention = use_attention
        self.use_self_attention = use_self_attention

        input_channels = min(max_features, block_expansion * (2 ** (num_down_blocks + 1)))
        norm_G = 'spadespectralinstance'
        label_num_channels = input_channels  # 256

        self.fc = nn.Conv2d(input_channels, 2 * input_channels, 3, padding=1)

        # Middle SPADE ResNet blocks with residual connections
        self.middle_blocks = nn.ModuleList([
            SPADEResnetBlock(2 * input_channels, 2 * input_channels, norm_G, label_num_channels)
            for _ in range(num_middle_blocks)
        ])

        # Attention mechanism
        if self.use_attention:
            self.attention = nn.MultiheadAttention(2 * input_channels, num_heads=8)

        # Self-attention mechanism
        if self.use_self_attention:
            self.self_attention = SelfAttention(2 * input_channels)

        # Upsampling blocks
        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.up_0 = SPADEResnetBlock(2 * input_channels, input_channels, norm_G, label_num_channels)
        self.up_1 = SPADEResnetBlock(input_channels, out_channels, norm_G, label_num_channels)

        # Output convolution
        if self.upscale is None or self.upscale <= 1:
            self.conv_img = nn.Conv2d(out_channels, 3, 3, padding=1)
        else:
            self.conv_img = nn.Sequential(
                nn.Conv2d(out_channels, 3 * (self.upscale ** 2), kernel_size=3, padding=1),
                nn.PixelShuffle(upscale_factor=self.upscale)
            )

        # Dropout for regularization
        self.dropout = nn.Dropout(dropout_rate)

        # Instance Normalization for style flexibility
        self.inst_norm = nn.InstanceNorm2d(out_channels)

    def forward(self, feature):
        seg = feature  # Bx256x64x64
        x = self.fc(feature)  # Bx512x64x64

        # Apply middle blocks with residual connections
        for block in self.middle_blocks:
            residual = x
            x = block(x, seg)
            x = x + residual
            x = self.dropout(x)

        # Apply attention if enabled
        if self.use_attention:
            b, c, h, w = x.size()
            x_flat = x.view(b, c, -1).permute(2, 0, 1)
            x_att, _ = self.attention(x_flat, x_flat, x_flat)
            x = x + x_att.permute(1, 2, 0).view(b, c, h, w)

        # Apply self-attention if enabled
        if self.use_self_attention:
            x = self.self_attention(x)

        # First upsampling
        x = self.up(x)  # Bx512x64x64 -> Bx512x128x128
        x = self.up_0(x, seg)  # Bx512x128x128 -> Bx256x128x128

        # Second upsampling
        x = self.up(x)  # Bx256x128x128 -> Bx256x256x256
        x = self.up_1(x, seg)  # Bx256x256x256 -> Bx64x256x256

        # Instance Normalization for style flexibility
        x = self.inst_norm(x)

        # Final convolution and activation
        x = self.conv_img(F.leaky_relu(x, 2e-1))  # Bx64x256x256 -> Bx3xHxW
        x = torch.sigmoid(x)  # Bx3xHxW

        return x

class SelfAttention(nn.Module):
    def __init__(self, in_dim):
        super(SelfAttention, self).__init__()
        self.query_conv = nn.Conv2d(in_dim, in_dim // 8, kernel_size=1)
        self.key_conv = nn.Conv2d(in_dim, in_dim // 8, kernel_size=1)
        self.value_conv = nn.Conv2d(in_dim, in_dim, kernel_size=1)
        self.gamma = nn.Parameter(torch.zeros(1))

    def forward(self, x):
        batch_size, C, width, height = x.size()
        proj_query = self.query_conv(x).view(batch_size, -1, width * height).permute(0, 2, 1)
        proj_key = self.key_conv(x).view(batch_size, -1, width * height)
        energy = torch.bmm(proj_query, proj_key)
        attention = F.softmax(energy, dim=-1)
        proj_value = self.value_conv(x).view(batch_size, -1, width * height)
        out = torch.bmm(proj_value, attention.permute(0, 2, 1))
        out = out.view(batch_size, C, width, height)
        out = self.gamma * out + x
        return out
