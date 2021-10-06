import torch
import torch.nn as nn
import torch.nn.functional as F 
from einops.layers.torch import Reduce

class ChannelAttention(nn.Module):
    def __init__(self, map_channel, reduction_ratio = 16,  pool_types = ['avg', 'max']):
        super().__init__()
        self.map_channel = map_channel
        self.pool_types = pool_types
        self.shared_mlp = nn.Sequential(
                nn.Flatten(),
                nn.Linear(map_channel, map_channel // reduction_ratio),
                nn.ReLU(),
                nn.Linear(map_channel // reduction_ratio, map_channel)
            )

    def forward(self, x):
        B, _, H, W = x.shape 

        for pool_type in self.pool_types:       
            if pool_type == 'avg':
                avg_pooled_map = F.avg_pool2d(input = x,kernel_size= (H, W), stride = (H, W))
    
            elif pool_type == 'max':
                max_pooled_map = F.max_pool2d(input = x, kernel_size = (H, W), stride = (H, W))

        #print(avg_pooled_map.shape)
        scale = self.shared_mlp(avg_pooled_map) + self.shared_mlp(max_pooled_map)
        scale = torch.sigmoid(scale)
        scale = scale.unsqueeze(2).unsqueeze(3).expand_as(x)

        return x * scale, scale


class SpatialAttention(nn.Module):
    def __init__(self, kernel_size = 7):
        super().__init__()

        self.conv = nn.Conv2d(2, 1, kernel_size = kernel_size, stride = 1, padding = (kernel_size - 1) // 2)
        self.max_pooling = Reduce('b c h w -> b 1 h w', 'max')
        self.avg_pooling = Reduce('b c h w -> b 1 h w', 'mean')

    def forward(self, x):
        max_pooled_map = self.max_pooling(x)
        avg_pooled_map = self.avg_pooling(x)
        aggregation = torch.cat([max_pooled_map, avg_pooled_map], dim = 1)
        out = self.conv(aggregation)
        out = torch.sigmoid(out)

        return x * out, out


class SelfAttention(nn.Module):
    def __init__(self, in_channels, reduction_ratio = 8):
        super().__init__()

        self.query_projector = nn.Conv2d(in_channels, in_channels // reduction_ratio, kernel_size = 1)
        self.key_projector = nn.Conv2d(in_channels, in_channels // reduction_ratio, kernel_size = 1)
        self.value_projector = nn.Conv2d(in_channels, in_channels // reduction_ratio, kernel_size=1)
        self.output_projector = nn.Conv2d(in_channels // reduction_ratio, in_channels, kernel_size = 1)
        self.reduction_ratio = reduction_ratio
        self.gamma = nn.Parameter(torch.randn(1))

        self.softmax = nn.Softmax(dim = -1)

    def forward(self, x):
        B, C, H, W = x.shape
        C_hat = C // self.reduction_ratio
        query = self.query_projector(x)
        key = self.key_projector(x)
        value = self.value_projector(x)


        query = query.view(B,C_hat, -1).transpose(2, 1)
        key = key.view(B, C_hat, -1)
        attention_map = torch.matmul(query, key)
        attention_map = self.softmax(attention_map)

        value = value.view(B, C_hat, -1)
        out = torch.matmul(value, attention_map)
        out = out.view(B, C_hat, H, W)
        out = self.output_projector(out)

        return out + self.gamma * out, attention_map


class AttentionBlock(nn.Module):
    def __init__(self, in_channels):
        super().__init__()

        self.spatial_attention = SpatialAttention()
        self.channel_attention = ChannelAttention(in_channels)

    def forward(self, x):
        out1, map1 = self.channel_attention(x)
        out2, map2 = self.spatial_attention(out1)

        return out2, (map1, map2)
