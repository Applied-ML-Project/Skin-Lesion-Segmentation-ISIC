import torch
import torch.nn as nn
import math


class ConvBlock(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.conv(x)


class ResBlock(nn.Module):
    """Residual block for CNN encoder."""
    def __init__(self, in_ch, out_ch, stride=1):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, stride=stride, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
        )
        self.skip = nn.Sequential()
        if stride != 1 or in_ch != out_ch:
            self.skip = nn.Sequential(
                nn.Conv2d(in_ch, out_ch, 1, stride=stride, bias=False),
                nn.BatchNorm2d(out_ch),
            )
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.relu(self.conv(x) + self.skip(x))


class TransformerBlock(nn.Module):
    def __init__(self, embed_dim, num_heads, mlp_ratio=4.0, dropout=0.1):
        super().__init__()
        self.norm1 = nn.LayerNorm(embed_dim)
        self.attn = nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout, batch_first=True)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, int(embed_dim * mlp_ratio)),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(int(embed_dim * mlp_ratio), embed_dim),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        h = self.norm1(x)
        x = x + self.attn(h, h, h)[0]
        x = x + self.mlp(self.norm2(x))
        return x


class TransUNet(nn.Module):
    def __init__(
        self,
        img_size=256,
        in_channels=3,
        out_channels=1,
        embed_dim=512,
        num_heads=8,
        num_layers=6,
        encoder_channels=[64, 128, 256],
        dropout=0.1,
    ):
        super().__init__()
        self.img_size = img_size

        # CNN encoder — downsample by 2 at each stage
        self.enc1 = ResBlock(in_channels, encoder_channels[0], stride=1)
        self.pool1 = nn.MaxPool2d(2)
        self.enc2 = ResBlock(encoder_channels[0], encoder_channels[1], stride=1)
        self.pool2 = nn.MaxPool2d(2)
        self.enc3 = ResBlock(encoder_channels[1], encoder_channels[2], stride=1)
        self.pool3 = nn.MaxPool2d(2)

        # after 3 pools: 256 -> 32
        self.enc_bridge = ResBlock(encoder_channels[2], encoder_channels[2], stride=1)
        self.pool_bridge = nn.MaxPool2d(2)
        # now 16x16

        feat_size = img_size // 16
        num_patches = feat_size * feat_size

        self.patch_embed = nn.Conv2d(encoder_channels[2], embed_dim, 1)
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches, embed_dim))
        self.pos_drop = nn.Dropout(dropout)

        self.transformer = nn.Sequential(
            *[TransformerBlock(embed_dim, num_heads, dropout=dropout) for _ in range(num_layers)]
        )

        self.proj_back = nn.Conv2d(embed_dim, encoder_channels[2], 1)

        # decoder — upsample keeps channels matching the skip connection
        self.up3 = nn.ConvTranspose2d(encoder_channels[2], encoder_channels[2], 2, stride=2)
        self.dec3 = ConvBlock(encoder_channels[2] * 2, encoder_channels[2])

        self.up2 = nn.ConvTranspose2d(encoder_channels[2], encoder_channels[2], 2, stride=2)
        self.dec2 = ConvBlock(encoder_channels[2] + encoder_channels[2], encoder_channels[1])

        self.up1 = nn.ConvTranspose2d(encoder_channels[1], encoder_channels[1], 2, stride=2)
        self.dec1 = ConvBlock(encoder_channels[1] + encoder_channels[1], encoder_channels[0])

        self.up0 = nn.ConvTranspose2d(encoder_channels[0], encoder_channels[0], 2, stride=2)
        self.dec0 = ConvBlock(encoder_channels[0], encoder_channels[0])

        self.final = nn.Conv2d(encoder_channels[0], out_channels, 1)

        self._init_pos_embed(num_patches, embed_dim)

    def _init_pos_embed(self, num_patches, embed_dim):
        pos = torch.arange(num_patches).unsqueeze(1).float()
        dim = torch.arange(embed_dim).unsqueeze(0).float()
        angles = pos / (10000 ** (2 * (dim // 2) / embed_dim))
        pe = torch.zeros(1, num_patches, embed_dim)
        pe[0, :, 0::2] = torch.sin(angles[:, 0::2])
        pe[0, :, 1::2] = torch.cos(angles[:, 1::2])
        self.pos_embed.data.copy_(pe)

    def forward(self, x):
        # encoder
        e1 = self.enc1(x)          # B, 64, 256, 256
        e2 = self.enc2(self.pool1(e1))  # B, 128, 128, 128
        e3 = self.enc3(self.pool2(e2))  # B, 256, 64, 64
        e_bridge = self.enc_bridge(self.pool3(e3))  # B, 256, 32, 32
        e4 = self.pool_bridge(e_bridge)  # B, 256, 16, 16

        # patch embed + transformer
        B, C, H, W = e4.shape
        t = self.patch_embed(e4)        # B, embed_dim, 16, 16
        t = t.flatten(2).transpose(1, 2)  # B, 256, embed_dim
        t = self.pos_drop(t + self.pos_embed)
        t = self.transformer(t)
        t = t.transpose(1, 2).view(B, -1, H, W)
        t = self.proj_back(t)           # B, 256, 16, 16

        # decoder with skip connections
        d3 = self.up3(t)                        # B, 256, 32, 32
        d3 = self.dec3(torch.cat([d3, e_bridge], dim=1))

        d2 = self.up2(d3)                       # B, 128, 64, 64
        d2 = self.dec2(torch.cat([d2, e3], dim=1))

        d1 = self.up1(d2)                       # B, 64, 128, 128
        d1 = self.dec1(torch.cat([d1, e2], dim=1))

        d0 = self.up0(d1)                       # B, 64, 256, 256
        d0 = self.dec0(d0)

        return torch.sigmoid(self.final(d0))
