import torch
import torch.nn as nn
from einops import rearrange
from einops.layers.torch import Rearrange
import math

# ... (FeedForward, Attention, and Transformer classes remain unchanged)

def sinusoidalpositionalencoding(d_model, length):
    """
    :param d_model: dimension of the model
    :param length: length of positions
    :return: length*d_model position matrix
    """
    if d_model % 2 != 0:
        raise ValueError("Cannot use sin/cos positional encoding with "
                         "odd dim (got dim={:d})".format(d_model))
    pe = torch.zeros(1, length, d_model)
    position = torch.arange(0, length).unsqueeze(1)
    div_term = torch.exp((torch.arange(0, d_model, 2, dtype=torch.float) *
                         -(math.log(10000.0) / d_model)))
    pe[:, :, 0::2] = torch.sin(position.float() * div_term)
    pe[:, :, 1::2] = torch.cos(position.float() * div_term)

    return pe

class FeatureMap(torch.nn.Module):

    """Class to perform final 1D convolution before calculating cross entropy or using softmax."""

    def __init__(self, input_n, output_n):
        super(FeatureMap, self).__init__()

        self.feature_out = torch.nn.Conv2d(input_n, output_n, kernel_size=(1, 1), stride=(1, 1))

    def forward(self, x):
        """Pass x through final layer."""
        return self.feature_out(x)

class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout = 0.):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        return self.net(x)

class Attention(nn.Module):
    def __init__(self, dim, heads = 8, dim_head = 64, dropout = 0.):
        super().__init__()
        inner_dim = dim_head *  heads
        project_out = not (heads == 1 and dim_head == dim)

        self.heads = heads
        self.scale = dim_head ** -0.5

        self.norm = nn.LayerNorm(dim)

        self.attend = nn.Softmax(dim = -1)
        self.dropout = nn.Dropout(dropout)

        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias = False)

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        ) if project_out else nn.Identity()

    def forward(self, x):
        x = self.norm(x)

        qkv = self.to_qkv(x).chunk(3, dim = -1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = self.heads), qkv)

        dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale

        attn = self.attend(dots)
        attn = self.dropout(attn)

        out = torch.matmul(attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        return self.to_out(out)

class Transformer(nn.Module):
    def __init__(self, dim, depth, heads, dim_head, mlp_dim, dropout = 0.):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([ 
                Attention(dim, heads = heads, dim_head = dim_head, dropout = dropout),
                FeedForward(dim, mlp_dim, dropout = dropout)
            ]))

    def forward(self, x):
        for attn, ff in self.layers:
            x = attn(x) + x
            x = ff(x) + x
        return self.norm(x)

class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels, mid_channels=None):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        #print(f"DoubleConv input shape: {x.shape}, dtype: {x.dtype}")
        return self.double_conv(x)

class SegmentationViT(nn.Module):
    def __init__(self, options, **kwargs):
        super().__init__()

        self.image_size = options['patch_size']
        self.patch_size = options['vit']['patch_size']
        dim = options['vit']['dim']
        depth = options['vit']['depth']
        heads = options['vit']['heads']
        mlp_dim = options['vit']['mlp_dim']
        channels = len(options['train_variables'])
        dim_head = options['vit']['dim_head']
        dropout = options['vit']['dropout']
        
        self.num_patches = (self.image_size // self.patch_size) ** 2
        
        # Vision Transformer Encoder
        self.to_patch_embedding = nn.Sequential(
            Rearrange('b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1=self.patch_size, p2=self.patch_size),
            nn.Linear(self.patch_size * self.patch_size * channels, dim),
        )
        
        self.pos_embedding = sinusoidalpositionalencoding(dim, self.num_patches + 1) #nn.Parameter(torch.randn(1, self.num_patches + 1, dim))
        self.cls_token = nn.Parameter(torch.randn(1, 1, dim))
        
        self.transformer = Transformer(dim, depth, heads, dim_head, mlp_dim, dropout)
        
        # CNN Decoder with DoubleConv blocks
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(dim, 256, kernel_size=4, stride=2, padding=1),
            DoubleConv(256, 256),
            nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1),
            DoubleConv(128, 128),
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),
            DoubleConv(64, 64),
            nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1),
            DoubleConv(32, 32),
        )
        
        self.out_SIC = FeatureMap(32, output_n=options['n_classes']['SIC'])  #torch.nn.Linear(32, 1) 
        self.out_SOD = FeatureMap(32, output_n=options['n_classes']['SOD']) 
        self.out_FLOE = FeatureMap(32, output_n=options['n_classes']['FLOE'])

    def forward(self, img):
        x = self.to_patch_embedding(img)
        b, n, _ = x.shape
        
        cls_tokens = self.cls_token.expand(b, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)
        x += self.pos_embedding[:, :(n + 1)]
        
        x = self.transformer(x)
        
        # Remove cls token and reshape
        x = x[:, 1:].transpose(1, 2).view(b, -1, self.image_size // self.patch_size, self.image_size // self.patch_size)
        
        x = self.decoder(x)
        
        # Apply task-specific heads
        return {'SIC': self.out_SIC(x), #(x.permute(0, 2, 3, 1)), 
                'SOD': self.out_SOD(x),
                'FLOE': self.out_FLOE(x)}
