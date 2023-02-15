import torch.nn as nn
import torch.nn.functional as F
import torch
from base.base_net import BaseNet
from einops import rearrange
from einops.layers.torch import Rearrange

def conv_3x3_bn(inp, oup, image_size, downsample=False):
    stride = 1 if downsample == False else 2
    return nn.Sequential(
        nn.Conv2d(inp, oup, 3, stride, 1, bias=False),
        nn.BatchNorm2d(oup),
        nn.GELU()
    )
def conv_1x1_bn(inp, oup, image_size, downsample=False):
    stride = 1 if downsample == False else 2
    return nn.Sequential(
        nn.Conv1d(inp, oup, 3, stride, 1, bias=False),
        nn.BatchNorm1d(oup),
        #nn.GELU()
    )



class PreNorm(nn.Module):
    def __init__(self, dim, fn, norm):
        #self.conv = PreNorm(inp, self.conv, nn.BatchNorm2d)
        #PreNorm(inp, self.attn, nn.LayerNorm),
        super().__init__()
        self.norm = norm(dim)
        self.fn = fn

    def forward(self, x, **kwargs):
        return self.fn(self.norm(x), **kwargs)


class SE(nn.Module):
    def __init__(self, inp, oup, expansion=0.25):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Sequential(
            nn.Linear(oup, int(inp * expansion), bias=False),
            nn.GELU(),
            nn.Linear(int(inp * expansion), oup, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1)
        return x * y

class MBConv(nn.Module):
    # block(inp, oup, image_size)
    def __init__(self, inp, oup, image_size, downsample=False, expansion=4):
        super().__init__()
        self.downsample = downsample
        stride = 1 if self.downsample == False else 2
        hidden_dim = int(inp * expansion)

        if self.downsample:
            self.pool = nn.MaxPool1d(3, 2, 1)
            self.proj = nn.Conv1d(inp, oup, 1, 1, 0, bias=False)

        if expansion == 1:
            self.conv = nn.Sequential(
                # dw
                nn.Conv1d(hidden_dim, hidden_dim, 3, stride,
                          1, groups=hidden_dim, bias=False),
                nn.BatchNorm1d(hidden_dim),
                nn.GELU(),
                # pw-linear
                nn.Conv1d(hidden_dim, oup, 1, 1, 0, bias=False),
                nn.BatchNorm1d(oup),
            )
        else:
            self.conv = nn.Sequential(
                # pw
                # down-sample in the first conv
                nn.Conv1d(inp, hidden_dim, 1, stride, 0, bias=False),
                nn.BatchNorm1d(hidden_dim),
                nn.GELU(),
                # dw
                nn.Conv1d(hidden_dim, hidden_dim, 3, 1, 1,
                          groups=hidden_dim, bias=False),
                nn.BatchNorm1d(hidden_dim),
                nn.GELU(),
                SE(inp, hidden_dim),
                # pw-linear
                nn.Conv1d(hidden_dim, oup, 1, 1, 0, bias=False),
                nn.BatchNorm1d(oup),
            )

        self.conv = PreNorm(inp, self.conv, nn.BatchNorm1d)

    def forward(self, x):
        if self.downsample:
            return self.proj(self.pool(x)) + self.conv(x)
        else:
            print('x + self.conv(x)==========================', x.shape, self.conv(x).shape)
            return x + self.conv(x)

class Attention(nn.Module):
    def __init__(self, inp, oup, image_size, heads=8, dim_head=32, dropout=0.):
        #self.attn = Attention(inp, oup, image_size, heads, dim_head, dropout)
        super().__init__()
        inner_dim = dim_head * heads
        project_out = not (heads == 1 and dim_head == inp)

        self.ih, self.iw = image_size

        self.heads = heads
        self.scale = dim_head ** -0.5

        # parameter table of relative position bias
        self.relative_bias_table = nn.Parameter(
            torch.zeros((2 * self.ih - 1) * (2 * self.iw - 1), heads))

        coords = torch.meshgrid((torch.arange(self.ih), torch.arange(self.iw)))
        coords = torch.flatten(torch.stack(coords), 1)
        relative_coords = coords[:, :, None] - coords[:, None, :]

        relative_coords[0] += self.ih - 1
        relative_coords[1] += self.iw - 1
        relative_coords[0] *= 2 * self.iw - 1
        relative_coords = rearrange(relative_coords, 'c h w -> h w c')
        relative_index = relative_coords.sum(-1).flatten().unsqueeze(1)
        self.register_buffer("relative_index", relative_index)

        self.attend = nn.Softmax(dim=-1)
        self.to_qkv = nn.Linear(inp, inner_dim * 3, bias=False)

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, oup),
            nn.Dropout(dropout)
        ) if project_out else nn.Identity()

    def forward(self, x):
        qkv = self.to_qkv(x).chunk(3, dim=-1)

        q, k, v = map(lambda t: rearrange(
            t, 'b n (h d) -> b h n d', h=self.heads), qkv)
        dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale

        # Use "gather" for more efficiency on GPUs
        relative_bias = self.relative_bias_table.gather(
            0, self.relative_index.repeat(1, self.heads))
        relative_bias = rearrange(
            relative_bias, '(h w) c -> 1 c h w', h=self.ih*self.iw, w=self.ih*self.iw)
        dots = dots + relative_bias

        attn = self.attend(dots)
        out = torch.matmul(attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        out = self.to_out(out)


        return out
class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout=0.):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        return self.net(x)

class Transformer(nn.Module):
    def __init__(self, inp, oup, image_size, heads=8, dim_head=32, downsample=False, dropout=0.):
        super().__init__()
        hidden_dim = int(inp * 4)

        self.ih, self.iw = image_size
        self.downsample = downsample

        if self.downsample:
            self.pool1 = nn.MaxPool1d(3, 2, 1)
            self.pool2 = nn.MaxPool1d(3, 2, 1)
            self.proj = nn.Conv1d(inp, oup, 1, 1, 0, bias=False)

        self.attn = Attention(inp, oup, image_size, heads, dim_head, dropout)
        self.ff = FeedForward(oup, hidden_dim, dropout)


        self.attn = nn.Sequential(
            Rearrange('b c ih  -> b (ih ) c'),
            PreNorm(inp, self.attn, nn.LayerNorm),
            Rearrange('b (ih ) c -> b c ih ', ih=self.ih)
        )

        self.ff = nn.Sequential(
            Rearrange('b c ih  -> b (ih ) c'),
            PreNorm(oup, self.ff, nn.LayerNorm),
            Rearrange('b (ih ) c -> b c ih', ih=self.ih)
        )

    def forward(self, x):
        if self.downsample:
            x = self.proj(self.pool1(x)) + self.attn(self.pool2(x))
        else:
            x = x + self.attn(x)
        x = x + self.ff(x)
        return x



############################################################################
class h_sigmoid(nn.Module):
    def __init__(self, inplace=True):
        super(h_sigmoid, self).__init__()
        self.relu = nn.ReLU6(inplace=inplace)

    def forward(self, x):
        return self.relu(x + 3) / 6

class h_swish(nn.Module):
    def __init__(self, inplace=True):
        super(h_swish, self).__init__()
        self.sigmoid = h_sigmoid(inplace=inplace)

    def forward(self, x):
        return x * self.sigmoid(x)
# Residual Channel Attention Block (RCAB)nn.GELU()
class RCAB(nn.Module):
    def __init__(self, n_feat, kernel_size, reduction, bias=True, bn=False, act=h_swish(), res_scale=1):
        super(RCAB, self).__init__()
        modules_body = []
        for i in range(2):
            modules_body.append(
                nn.Conv1d(n_feat, n_feat, kernel_size=kernel_size, stride=1, padding=kernel_size // 2, bias=bias))
            modules_body.append(nn.BatchNorm1d(n_feat))
            modules_body.append(act)
            # if bn:
            #     modules_body.append(nn.BatchNorm2d(n_feat))
            # if i == 0:
            #     modules_body.append(act)
        # modules_body.append(CALayer(n_feat, reduction))
        self.body = nn.Sequential(*modules_body)
        self.res_scale = res_scale

    def forward(self, x):
        res = self.body(x).mul(self.res_scale)
        res = res + x
        return res


# Residual Group (RG)
#class ResidualGroup(nn.Module):
class block_hz(nn.Module):
    def __init__(self, inp, oup, image_size=1, downsample=False, kernel_size=3,
                 reduction=16, res_scale=1, n_resblocks=2, act=nn.ReLU(False)):
        super(block_hz, self).__init__()
        n_feat = inp
        modules_body = [
            RCAB(n_feat, kernel_size, reduction, bias=True, bn=False, act=act, res_scale=res_scale)
            for _ in range(n_resblocks)]
        modules_body.append(nn.Conv1d(n_feat, n_feat, kernel_size=kernel_size, stride=1, padding=kernel_size // 2))
        self.body = nn.Sequential(*modules_body)

    def forward(self, x):
        res = self.body(x)
        res = res + x
        return res

############################################################################
class CTA(BaseNet):

    def __init__(self, image_size, in_channels, num_blocks=[2, 2, 3, 5, 2], channels=[64, 64, 64, 64, 64], num_classes=1000,
                 block_types=['C', 'C', 'T', 'T'], block = {'C': block_hz, 'T': Transformer}):
        super().__init__()
        # num_blocks = [2, 2, 3, 5, 2]  # L
        # channels = [64, 64, 64, 64, 64]  # D
        self.rep_dim = num_classes

        ih, iw = image_size
        # block = {'C': MBConv, 'T': Transformer}

        self.s0 = self._make_layer(
            conv_1x1_bn, in_channels, channels[0], num_blocks[0], (ih // 1, iw // 1))
        self.s1 = self._make_layer(
            block[block_types[0]], channels[0], channels[1], num_blocks[1], (ih // 4, iw // 4))
        self.s2 = self._make_layer(
            block[block_types[1]], channels[1], channels[2], num_blocks[2], (ih // 8, iw // 8))
        self.s3 = self._make_layer(
            block[block_types[2]], channels[2], channels[3], num_blocks[3], (ih // 1, iw // 1))
        self.s4 = self._make_layer(
            block[block_types[3]], channels[3], channels[4], num_blocks[4], (ih // 1, iw // 1))

        self.pool = nn.AvgPool1d(ih, 1)
        self.fc = nn.Linear(channels[-1] * ih * iw, num_classes, bias=False)

    def forward(self, x):
        x = torch.unsqueeze(x, 2)

        # print('x=======================', x.shape)
        #x======================= torch.Size([128, 175, 1])
        w = x.shape[2]
        x = self.s0(x)
        x = self.s1(x)
        x = self.s2(x)
        x = self.s3(x)


        x = self.s4(x)
        #############################################%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        x = x.view(x.shape[0], -1)
        x = self.fc(x)
        return x
    def _make_layer(self, block, inp, oup, depth, image_size):
        layers = nn.ModuleList([])
        for i in range(depth):
            if i == 0:
                layers.append(block(inp, oup, image_size))#, downsample=True))
            else:
                layers.append(block(oup, oup, image_size))
        return nn.Sequential(*layers)




class CTA_Decoder(BaseNet):

    def __init__(self, image_size, in_channels, num_blocks=[2, 2, 3, 5, 2], channels=[64, 64, 64, 64, 64], num_classes=1000,
                 block_types=['C', 'C', 'T', 'T'], block = {'C': block_hz, 'T': Transformer}):
        super().__init__()

        self.rep_dim = num_classes

        ih, iw = image_size
        self.ih = ih
        self.iw = iw
        # block = {'C': MBConv, 'T': Transformer}

        self.d0 = self._make_layer(
            block[block_types[3]], channels[4], channels[3], num_blocks[4], (ih // 1, iw // 1))
        self.d1 = self._make_layer(
            block[block_types[2]], channels[3], channels[2], num_blocks[3], (ih // 1, iw // 1))
        self.d2 = self._make_layer(
            block[block_types[1]], channels[2], channels[1], num_blocks[2], (ih // 1, iw // 1))
        self.d3 = self._make_layer(
            block[block_types[0]], channels[1], channels[0], num_blocks[1], (ih // 1, iw // 1))
        self.d4 = self._make_layer(
            conv_1x1_bn, channels[0], in_channels, num_blocks[0], (ih // 1, iw // 1))

        self.pool = nn.AvgPool1d(ih, 1)
        self.dfc = nn.Linear(num_classes, channels[-1] * ih * iw, bias=False)
        self.dsigmoid = nn.Sigmoid()
        #m = nn.Sigmoid()





    def forward(self, x):
        x = self.dfc(x)
        x = x.view(x.shape[0], -1, self.ih)

        x = self.d0(x)
        x = self.d1(x)
        x = self.d2(x)
        x = self.d3(x)
        x = self.d4(x)

        x = self.dsigmoid(x)
        x = torch.squeeze(x, 2)
        return x
    def _make_layer(self, block, inp, oup, depth, image_size):
        layers = nn.ModuleList([])
        for i in range(depth):
            if i == 0:
                layers.append(block(inp, oup, image_size))#, downsample=True))
            else:
                layers.append(block(oup, oup, image_size))
        return nn.Sequential(*layers)




class CTA_Autoencoder(BaseNet):

    def __init__(self, image_size, in_channels, num_blocks=[2, 2, 3, 5, 2], channels=[64, 64, 64, 64, 64], num_classes=1000,
                 block_types=['C', 'C', 'T', 'T']):
        super().__init__()

        self.rep_dim = num_classes
        ih, iw = image_size
        self.ih = ih
        self.iw = iw
        block = {'C': block_hz, 'T': Transformer}

        self.encoder = CTA(image_size, in_channels, num_blocks=num_blocks, channels=channels, num_classes=num_classes, block_types=block_types, block = block)
        self.decoder = CTA_Decoder(image_size, in_channels, num_blocks=num_blocks, channels=channels, num_classes=num_classes, block_types=block_types, block = block)

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x
    def _make_layer(self, block, inp, oup, depth, image_size):
        layers = nn.ModuleList([])
        for i in range(depth):
            if i == 0:
                layers.append(block(inp, oup, image_size))#, downsample=True))
            else:
                layers.append(block(oup, oup, image_size))
        return nn.Sequential(*layers)


class Linear_BN_leakyReLU(nn.Module):
    """
    A nn.Module that consists of a Linear layer followed by BatchNorm1d and a leaky ReLu activation
    """

    def __init__(self, in_features, out_features, bias=False, eps=1e-04):
        super(Linear_BN_leakyReLU, self).__init__()

        self.linear = nn.Linear(in_features, out_features, bias=bias)
        self.bn = nn.BatchNorm1d(out_features, eps=eps, affine=bias)

    def forward(self, x):
        return F.leaky_relu(self.bn(self.linear(x)))
