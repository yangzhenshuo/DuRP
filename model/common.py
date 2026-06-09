import torch
import torch.nn as nn
import functools
from einops import rearrange
from .layer_utils.funcs import get_norm_layer

########################################################### Scene Polarization reconstruction #############################################################################
class PTNetwork(nn.Module):
    def __init__(self, in_chans, embed_dim, n_downsampling, n_blocks, norm_type='instance', use_dropout=False):
        super(PTNetwork, self).__init__()
        self.Ex_feature = nn.Sequential(
            nn.Conv2d(in_chans*6, embed_dim, kernel_size=1, stride=1),
            nn.InstanceNorm2d(embed_dim, affine=True),
            nn.RReLU(inplace=True),
            nn.Conv2d(embed_dim, embed_dim, kernel_size=3, stride=1, padding=1),
            nn.InstanceNorm2d(embed_dim, affine=True),
            nn.RReLU(inplace=True),
            nn.Conv2d(embed_dim, embed_dim, kernel_size=1, stride=1),
            nn.InstanceNorm2d(embed_dim, affine=True),
            nn.RReLU(inplace=True)
        )
        self.backbone = AttentionAutoencoderBackbone(embed_dim, output_nc=embed_dim, n_downsampling=n_downsampling, 
                                                    n_blocks=n_blocks, norm_type=norm_type, use_dropout=use_dropout,
                                                    use_channel_shuffle=True)
        self.Re_outblock = nn.Sequential(
            nn.Conv2d(in_channels=embed_dim, out_channels=in_chans, kernel_size=1, stride=1),
            nn.ReLU(inplace=True)
        )
    
        
    def forward(self, x):
        # extract feature
        feature = self.Ex_feature(x)
        # apply backbone
        backbone_out = self.backbone(feature)
        # reconstruct
        out = self.Re_outblock(backbone_out)
        return out

class APNetwork(nn.Module):
    def __init__(self, in_chans, embed_dim, n_downsampling, n_blocks, norm_type='instance', use_dropout=False):
        super(APNetwork, self).__init__()
        self.Ex_feature_1 = nn.Sequential(
            nn.Conv2d(in_chans, embed_dim//2, kernel_size=1, stride=1),
            nn.InstanceNorm2d(embed_dim//2, affine=True),
            nn.RReLU(inplace=True),
            nn.Conv2d(embed_dim//2, embed_dim//2, kernel_size=3, stride=1, padding=1),
            nn.InstanceNorm2d(embed_dim//2, affine=True),
            nn.RReLU(inplace=True),
            nn.Conv2d(embed_dim//2, embed_dim//2, kernel_size=1, stride=1),
            nn.InstanceNorm2d(embed_dim//2, affine=True),
            nn.RReLU(inplace=True)
        )
        self.Ex_feature_2 = nn.Sequential(
            nn.Conv2d(in_chans, embed_dim//2, kernel_size=1, stride=1),
            nn.InstanceNorm2d(embed_dim//2, affine=True),
            nn.RReLU(inplace=True),
            nn.Conv2d(embed_dim//2, embed_dim//2, kernel_size=3, stride=1, padding=1),
            nn.InstanceNorm2d(embed_dim//2, affine=True),
            nn.RReLU(inplace=True),
            nn.Conv2d(embed_dim//2, embed_dim//2, kernel_size=1, stride=1),
            nn.InstanceNorm2d(embed_dim//2, affine=True),
            nn.RReLU(inplace=True)
        )
        self.backbone = AttentionAutoencoderBackbone(embed_dim, output_nc=embed_dim, n_downsampling=n_downsampling, 
                                                    n_blocks=n_blocks, norm_type=norm_type, use_dropout=use_dropout,
                                                    use_channel_shuffle=True)
        self.output = nn.Sequential(
            nn.Conv2d(in_channels=embed_dim, out_channels=in_chans, kernel_size=1, stride=1),
            nn.Tanh()
        )
        
    def forward(self, x):
        # Preprocessing
        i0, i45, i90, i135, i_un, i_pool = torch.split(x, 3, dim=1)
        s0 = i_un + i_pool
        inv_s0 = 1.0 / (s0 + 1e-6)
        x1 = ((i0 - i90) * inv_s0 + 1.0) * 0.5
        x2 = ((i45 - i135) * inv_s0 + 1.0) * 0.5
        # Extract features
        feature1 = torch.cat([self.Ex_feature_1(x1), self.Ex_feature_2(x2)], dim=1)
        feature2 = torch.cat([self.Ex_feature_1(x2), self.Ex_feature_2(x1)], dim=1)
        # Apply backbone
        backbone1 = self.backbone(feature1)
        backbone2 = self.backbone(feature2)
        # Apply output
        s1_Ap = self.output(backbone1) + x1
        s2_Ap = self.output(backbone2) + x2

        m_A = torch.complex(s1_Ap * 2 - 1, s2_Ap * 2 - 1)

        return m_A
    
class SPRNetwork(nn.Module):
    def __init__(self, in_chans, embed_dim, n_downsampling, n_blocks, norm_type='instance', use_dropout=False):
        super(SPRNetwork, self).__init__()
        self.Ex_feature_Dp = nn.Sequential(
            nn.Conv2d(in_chans*2, embed_dim//2, kernel_size=1, stride=1),
            nn.InstanceNorm2d(embed_dim//2, affine=True),
            nn.RReLU(inplace=True),
            nn.Conv2d(embed_dim//2, embed_dim//2, kernel_size=3, stride=1, padding=1),
            nn.InstanceNorm2d(embed_dim//2, affine=True),
            nn.RReLU(inplace=True),
            nn.Conv2d(embed_dim//2, embed_dim//2, kernel_size=1, stride=1),
            nn.InstanceNorm2d(embed_dim//2, affine=True),
            nn.RReLU(inplace=True)
        )
        self.Ex_feature_Ip = nn.Sequential(
            nn.Conv2d(in_chans*2, embed_dim//2, kernel_size=1, stride=1),
            nn.InstanceNorm2d(embed_dim//2, affine=True),
            nn.RReLU(inplace=True),
            nn.Conv2d(embed_dim//2, embed_dim//2, kernel_size=3, stride=1, padding=1),
            nn.InstanceNorm2d(embed_dim//2, affine=True),
            nn.RReLU(inplace=True),
            nn.Conv2d(embed_dim//2, embed_dim//2, kernel_size=1, stride=1),
            nn.InstanceNorm2d(embed_dim//2, affine=True),
            nn.RReLU(inplace=True)
        )
        self.backbone = AttentionAutoencoderBackbone(embed_dim, output_nc=embed_dim, n_downsampling=n_downsampling, 
                                                    n_blocks=n_blocks, norm_type=norm_type, use_dropout=use_dropout,
                                                    use_channel_shuffle=True)        
        self.out_s1 = nn.Sequential(
            nn.Conv2d(embed_dim, in_chans, kernel_size=1, stride=1),
            nn.Tanh()
        )
        self.out_s2 = nn.Sequential(
            nn.Conv2d(embed_dim, in_chans, kernel_size=1, stride=1),
            nn.Tanh()
        )
    def forward(self, x1, x2):
        x11, x12 = torch.split(x1, 3, dim=1)
        feature = torch.cat([self.Ex_feature_Dp(x1), self.Ex_feature_Ip(x2)], dim=1)
        backbone_out = self.backbone(feature)
        s1_Dp = self.out_s1(backbone_out) + x11
        s2_Dp = self.out_s2(backbone_out) + x12
        m_D = torch.complex(s1_Dp * 2 - 1, s2_Dp * 2 - 1)
        return m_D
########################################################### Scene Radiance reconstruction #############################################################################
class ALNetwork(nn.Module):
    def __init__(self, in_chans, embed_dim, n_downsampling, n_blocks, norm_type='instance', use_dropout=False):
        super(ALNetwork, self).__init__()
        self.Ex_feature = nn.Sequential(
            nn.Conv2d(in_chans*6, embed_dim, kernel_size=1, stride=1),
            nn.InstanceNorm2d(embed_dim, affine=True),
            nn.RReLU(inplace=True),
            nn.Conv2d(embed_dim, embed_dim, kernel_size=3, stride=1, padding=1),
            nn.InstanceNorm2d(embed_dim, affine=True),
            nn.RReLU(inplace=True),
            nn.Conv2d(embed_dim, embed_dim, kernel_size=1, stride=1),
            nn.InstanceNorm2d(embed_dim, affine=True),
            nn.RReLU(inplace=True)
        )
        self.backbone = AttentionAutoencoderBackbone(embed_dim, output_nc=embed_dim, n_downsampling=n_downsampling, 
                                                    n_blocks=n_blocks, norm_type=norm_type, use_dropout=use_dropout,
                                                    use_channel_shuffle=True)
        self.out_block = nn.Sequential(
            nn.Conv2d(embed_dim, in_chans, kernel_size=1, stride=1),
            nn.Sigmoid()
        )
    def forward(self, x):
        # Extract features
        feature = self.Ex_feature(x) # N, embed_dim, H, W
        backbone_out = self.backbone(feature) # N, embed_dim, H, W
        # Apply attention
        out = self.out_block(backbone_out) # N, in_chans, 1, 1
        return out


class SRRNetwork(nn.Module):
    def __init__(self, in_chans, embed_dim, n_downsampling, n_blocks, norm_type='instance', use_dropout=False):
        super(SRRNetwork, self).__init__()
        self.Ex_feature = nn.Sequential(
            nn.Conv2d(in_chans*5, embed_dim, kernel_size=1, stride=1),
            nn.InstanceNorm2d(embed_dim, affine=True),
            nn.RReLU(inplace=True),
            nn.Conv2d(embed_dim, embed_dim, kernel_size=3, stride=1, padding=1),
            nn.InstanceNorm2d(embed_dim, affine=True),
            nn.RReLU(inplace=True),
            nn.Conv2d(embed_dim, embed_dim, kernel_size=1, stride=1),
            nn.InstanceNorm2d(embed_dim, affine=True),
            nn.RReLU(inplace=True)
        )
        self.backbone = AttentionAutoencoderBackbone(embed_dim, output_nc=embed_dim, n_downsampling=n_downsampling, 
                                                    n_blocks=n_blocks, norm_type=norm_type, use_dropout=use_dropout,
                                                    use_channel_shuffle=True)
        self.out_block = nn.Sequential(
            nn.Conv2d(embed_dim, in_chans, kernel_size=1, stride=1),
            nn.Tanh()
        )
    def forward(self, x1, x2):
        feature = self.Ex_feature(torch.cat([x1, x2], dim=1))
        backbone_out = self.backbone(feature)
        out = self.out_block(backbone_out) + x1
        return out
#############################################################################################################################################################################
class Self_Attn_FM(nn.Module):
    """ Self attention Layer for Feature Map dimension"""

    def __init__(self, in_dim, latent_dim=8, subsample=True):
        super(Self_Attn_FM, self).__init__()
        self.channel_latent = in_dim // latent_dim
        self.query_conv = nn.Conv2d(in_channels=in_dim, out_channels=self.channel_latent, kernel_size=1, stride=1)
        self.key_conv = nn.Conv2d(in_channels=in_dim, out_channels=self.channel_latent, kernel_size=1, stride=1)
        self.value_conv = nn.Conv2d(in_channels=in_dim, out_channels=self.channel_latent, kernel_size=1, stride=1)
        self.out_conv = nn.Conv2d(in_channels=self.channel_latent, out_channels=in_dim, kernel_size=1, stride=1)
        self.gamma = nn.Parameter(torch.zeros(1))
        self.softmax = nn.Softmax(dim=-1)

        if subsample:
            self.key_conv = nn.Sequential(
                self.key_conv,
                nn.MaxPool2d(2)
            )
            self.value_conv = nn.Sequential(
                self.value_conv,
                nn.MaxPool2d(2)
            )

    def forward(self, x):
        """
            inputs :
                x : input feature maps(B x C x H x W)
            returns :
                out : self attention value + input feature
        """
        batchsize, C, height, width = x.size()
        c = self.channel_latent
        # proj_query: reshape to B x N x c, N = H x W
        proj_query = self.query_conv(x).view(batchsize, c, -1).permute(0, 2, 1)
        # proj_key: reshape to B x c x N_, N_ = H_ x W_
        proj_key = self.key_conv(x).view(batchsize, c, -1)
        # energy: B x N x N_, N = H x W, N_ = H_ x W_
        energy = torch.bmm(proj_query, proj_key)
        # attention: B x N_ x N, N = H x W, N_ = H_ x W_
        attention = self.softmax(energy).permute(0, 2, 1)
        # proj_value: B x c x N_, N_ = H_ x W_
        proj_value = self.value_conv(x).view(batchsize, c, -1)
        # attention_out: B x c x N, N = H x W
        attention_out = torch.bmm(proj_value, attention)
        # out: B x C x H x W
        out = self.out_conv(attention_out.view(batchsize, c, height, width))

        out = self.gamma * out + x
        return out


class Chuncked_Self_Attn_FM(nn.Module):
    """
        in_channel -> in_channel
    """

    def __init__(self, in_channel, latent_dim=8, subsample=True, grid=(8, 8)):
        super(Chuncked_Self_Attn_FM, self).__init__()

        self.self_attn_fm = Self_Attn_FM(in_channel, latent_dim=latent_dim, subsample=subsample)
        self.grid = grid

    def forward(self, x):
        N, C, H, W = x.shape
        chunk_size_H, chunk_size_W = H // self.grid[0], W // self.grid[1]
        x_ = x.reshape(N, C, self.grid[0], chunk_size_H, self.grid[1], chunk_size_W).permute(0, 2, 4, 1, 3, 5).reshape(
            N * self.grid[0] * self.grid[1], C, chunk_size_H, chunk_size_W)
        output = self.self_attn_fm(x_).reshape(N, self.grid[0], self.grid[1], C, chunk_size_H,
                                               chunk_size_W).permute(0, 3, 1, 4, 2, 5).reshape(N, C, H, W)
        return output

class AttentionAutoencoderBackbone(nn.Module):
    """
        Attention autoencoder backbone
        input_nc -> output_nc
    """

    def __init__(self, input_nc, output_nc=64, n_downsampling=2, n_blocks=3, norm_type='instance', use_dropout=False,
                 use_channel_shuffle=True):
        super(AttentionAutoencoderBackbone, self).__init__()

        norm_layer = get_norm_layer(norm_type)
        if type(norm_layer) == functools.partial:  # no need to use bias as BatchNorm2d has affine parameters
            use_bias = norm_layer.func != nn.BatchNorm2d
        else:
            use_bias = norm_layer != nn.BatchNorm2d

        self.n_downsampling = n_downsampling
        self.n_blocks = n_blocks

        self.projection = nn.Sequential(
            nn.Conv2d(input_nc, output_nc, kernel_size=7, stride=1, padding=3, bias=use_bias),
            norm_layer(output_nc),
            nn.ReLU(inplace=True)
        )
        self.in_conv = nn.Sequential(
            nn.Conv2d(output_nc, output_nc, kernel_size=3, stride=1, padding=1, bias=use_bias),
            norm_layer(output_nc),
            nn.ReLU(inplace=True)
        )
        self.out_conv = nn.Sequential(
            nn.Conv2d(2 * output_nc, output_nc, kernel_size=3, stride=1, padding=1, bias=use_bias),
            norm_layer(output_nc),
            nn.ReLU(inplace=True)
        )
        self.downsampling_blocks = nn.ModuleList()
        self.upsampling_blocks = nn.ModuleList()

        dim = output_nc
        for i in range(n_downsampling):
            self.downsampling_blocks.append(
                SkipAutoencoderDownsamplingBlock(dim, 2 * dim, norm_layer, use_dropout, use_bias, use_channel_shuffle)
            )
            dim *= 2

        dense_blocks_seq = n_blocks * [DenseBlock(dim)]
        self.dense_blocks = nn.Sequential(*dense_blocks_seq)

        for i in range(n_downsampling):
            self.upsampling_blocks.append(
                AttentionAutoencoderUpsamplingBlock(dim, dim // 2, dim // 2, norm_layer, use_dropout, use_bias,
                                                    use_channel_shuffle)
            )
            dim //= 2

    def forward(self, x):
        x_ = self.projection(x)
        out = self.in_conv(x_)

        skip_links = list()
        for i in range(self.n_downsampling):
            skip_links.append(out)
            out = self.downsampling_blocks[i](out)

        out = self.dense_blocks(out)

        for i in range(self.n_downsampling):
            out = self.upsampling_blocks[i](out, skip_links[-i - 1])

        out = self.out_conv(torch.cat((x_, out), dim=1))
        return out

class ChannelShuffle(nn.Module):
    def __init__(self, groups=8):
        super(ChannelShuffle, self).__init__()
        self.groups = groups

    def forward(self, x):
        N, C, H, W = x.shape
        return x.reshape(N, self.groups, C // self.groups, H, W).transpose(1, 2).reshape(N, C, H, W)
        
class SkipAutoencoderDownsamplingBlock(nn.Module):
    """
        Autoencoder downsampling block with skip links
        in_channel -> out_channel
    """

    def __init__(self, in_channel, out_channel, norm_layer, use_dropout, use_bias, use_channel_shuffle):
        super(SkipAutoencoderDownsamplingBlock, self).__init__()

        self.projection = nn.Conv2d(in_channel, out_channel, kernel_size=1, stride=1)
        if use_channel_shuffle:
            self.bottleneck = nn.Sequential(
                nn.Conv2d(out_channel, out_channel, kernel_size=1, stride=1, bias=use_bias),
                norm_layer(out_channel),
                nn.ReLU(inplace=True),
                ChannelShuffle(groups=8),
                nn.Conv2d(out_channel, out_channel, kernel_size=3, stride=1, padding=1, bias=use_bias),
                norm_layer(out_channel),
                nn.ReLU(inplace=True),
                nn.Conv2d(out_channel, out_channel, kernel_size=1, stride=1, bias=use_bias),
            )
        else:
            self.bottleneck = nn.Sequential(
                nn.Conv2d(out_channel, out_channel, kernel_size=1, stride=1, bias=use_bias),
                norm_layer(out_channel),
                nn.ReLU(inplace=True),
                nn.Conv2d(out_channel, out_channel, kernel_size=3, stride=1, padding=1, bias=use_bias),
                norm_layer(out_channel),
                nn.ReLU(inplace=True),
                nn.Conv2d(out_channel, out_channel, kernel_size=1, stride=1, bias=use_bias),
            )
        out_sequence = [
            norm_layer(out_channel),
            nn.ReLU(inplace=True)
        ]

        if use_dropout:
            out_sequence += [nn.Dropout(0.5)]
        out_sequence += [nn.MaxPool2d(2)]

        self.out_block = nn.Sequential(*out_sequence)

    def forward(self, x):
        x_ = self.projection(x)
        out = self.out_block(x_ + self.bottleneck(x_))
        return out

class DenseCell(nn.Module):
    def __init__(self, in_channel, growth_rate, kernel_size=3):
        super(DenseCell, self).__init__()
        self.conv_block = nn.Sequential(
            nn.Conv2d(in_channels=in_channel, out_channels=growth_rate, kernel_size=kernel_size, stride=1,
                      padding=(kernel_size - 1) // 2, bias=False),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return torch.cat((x, self.conv_block(x)), dim=1)

class DenseBlock(nn.Module):
    """
        DenseBlock using bottleneck structure
        in_channel -> in_channel
    """

    def __init__(self, in_channel, growth_rate=32, n_blocks=3):
        super(DenseBlock, self).__init__()

        sequence = nn.ModuleList()

        dim = in_channel
        for i in range(n_blocks):
            sequence.append(DenseCell(dim, growth_rate))
            dim += growth_rate

        self.dense_cells = nn.Sequential(*sequence)
        self.fusion = nn.Conv2d(in_channels=dim, out_channels=in_channel, kernel_size=1, stride=1, bias=False)

    def forward(self, x):
        return self.fusion(self.dense_cells(x)) + x

class AttentionBlock(nn.Module):
    """
        attention block
        x:in_channel_x  g:in_channel_g  -->  in_channel_x
    """

    def __init__(self, in_channel_x, in_channel_g, channel_t, norm_layer, use_bias):
        # in_channel_x: input signal channels
        # in_channel_g: gating signal channels
        super(AttentionBlock, self).__init__()
        self.x_block = nn.Sequential(
            nn.Conv2d(in_channel_x, channel_t, kernel_size=1, stride=1, padding=0, bias=use_bias),
            norm_layer(channel_t)
        )

        self.g_block = nn.Sequential(
            nn.Conv2d(in_channel_g, channel_t, kernel_size=1, stride=1, padding=0, bias=use_bias),
            norm_layer(channel_t)
        )

        self.t_block = nn.Sequential(
            nn.Conv2d(channel_t, 1, kernel_size=1, stride=1, padding=0, bias=use_bias),
            norm_layer(1),
            nn.Sigmoid()
        )

        self.relu = nn.ReLU(inplace=True)

    def forward(self, x, g):
        # x: (N, in_channel_x, H, W)
        # g: (N, in_channel_g, H, W)
        x_out = self.x_block(x)  # (N, channel_t, H, W)
        g_out = self.g_block(g)  # (N, channel_t, H, W)
        t_in = self.relu(x_out + g_out)  # (N, 1, H, W)
        attention_map = self.t_block(t_in)  # (N, 1, H, W)
        return x * attention_map  # (N, in_channel_x, H, W)
      
class AttentionAutoencoderUpsamplingBlock(nn.Module):
    """
        Attention autoencoder upsampling block
        x1:in_channel1  x2:in_channel2  -->  out_channel
    """

    def __init__(self, in_channel1, in_channel2, out_channel, norm_layer, use_dropout, use_bias, use_channel_shuffle):
        super(AttentionAutoencoderUpsamplingBlock, self).__init__()
        # in_channel1: channels from the signal to be upsampled (gating signal)
        # in_channel2: channels from skip link (input signal)
        self.upsample = nn.ConvTranspose2d(in_channel1, in_channel1 // 2, kernel_size=4, stride=2, padding=1,
                                           bias=use_bias)
        self.attention = AttentionBlock(in_channel2, in_channel1 // 2, in_channel2, norm_layer, use_bias)
        self.projection = nn.Conv2d(in_channel1 // 2 + in_channel2, out_channel, kernel_size=1, stride=1)
        if use_channel_shuffle:
            self.bottleneck = nn.Sequential(
                nn.Conv2d(out_channel, out_channel, kernel_size=1, stride=1, bias=use_bias),
                norm_layer(out_channel),
                nn.ReLU(inplace=True),
                ChannelShuffle(groups=8),
                nn.Conv2d(out_channel, out_channel, kernel_size=3, stride=1, padding=1, bias=use_bias),
                norm_layer(out_channel),
                nn.ReLU(inplace=True),
                nn.Conv2d(out_channel, out_channel, kernel_size=1, stride=1, bias=use_bias),
            )
        else:
            self.bottleneck = nn.Sequential(
                nn.Conv2d(out_channel, out_channel, kernel_size=1, stride=1, bias=use_bias),
                norm_layer(out_channel),
                nn.ReLU(inplace=True),
                nn.Conv2d(out_channel, out_channel, kernel_size=3, stride=1, padding=1, bias=use_bias),
                norm_layer(out_channel),
                nn.ReLU(inplace=True),
                nn.Conv2d(out_channel, out_channel, kernel_size=1, stride=1, bias=use_bias),
            )
        out_sequence = [
            norm_layer(out_channel),
            nn.ReLU(inplace=True)
        ]

        if use_dropout:
            out_sequence += [nn.Dropout(0.5)]

        self.out_block = nn.Sequential(*out_sequence)

    def forward(self, x1, x2):
        # x1: the signal to be upsampled (gating signal)
        # x2: skip link (input signal)
        upsampled_x1 = self.upsample(x1)
        attentioned_x2 = self.attention(x2, upsampled_x1)
        x_ = self.projection(torch.cat((attentioned_x2, upsampled_x1), dim=1))
        out = self.out_block(x_ + self.bottleneck(x_))
        return out
    
class Attention(nn.Module):
    def __init__(self, dim, bias):
        super(Attention, self).__init__()
        self.qkv = nn.Conv2d(dim, dim*3, kernel_size=1, bias=bias)
        self.qkv_dwconv = nn.Conv2d(dim*3, dim*3, kernel_size=3, stride=1, padding=1, groups=dim*3, bias=bias)
        self.project_out = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)
        
    def forward(self, x):
        b,c,h,w = x.shape

        qkv = self.qkv_dwconv(self.qkv(x))
        q,k,v = qkv.chunk(3, dim=1)   
        q = rearrange(q, 'b c h w -> b c (h w)') 
        k = rearrange(k, 'b c h w -> b c (h w)')
        v = rearrange(v, 'b c h w -> b c (h w)')

        q = torch.nn.functional.normalize(q, dim=-1)
        k = torch.nn.functional.normalize(k, dim=-1)

        attn = (q @ k.transpose(-2, -1))
        attn = attn.softmax(dim=-1)

        out = (attn @ v)
        
        out = rearrange(out, 'b c (h w) -> b c h w', h=h, w=w)

        out = self.project_out(out)
        return out

