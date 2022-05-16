import torch
from torch import nn, einsum
import torch.nn.functional as F
from einops import rearrange, repeat
from einops.layers.torch import Rearrange

# helper methods

def group_dict_by_key(cond, d):
    return_val = [dict(), dict()]
    for key in d.keys():
        match = bool(cond(key))
        ind = int(not match)
        return_val[ind][key] = d[key]
    return (*return_val,)

def group_by_key_prefix_and_remove_prefix(prefix, d):
    kwargs_with_prefix, kwargs = group_dict_by_key(lambda x: x.startswith(prefix), d)
    kwargs_without_prefix = dict(map(lambda x: (x[0][len(prefix):], x[1]), tuple(kwargs_with_prefix.items())))
    return kwargs_without_prefix, kwargs

# classes

class LayerNorm(nn.Module):
    def __init__(self, dim, eps = 1e-5):
        super().__init__()
        self.eps = eps
        self.g = nn.Parameter(torch.ones(1, dim, 1, 1))
        self.b = nn.Parameter(torch.zeros(1, dim, 1, 1))

    def forward(self, x):
        std = torch.var(x, dim = 1, unbiased = False, keepdim = True).sqrt()
        mean = torch.mean(x, dim = 1, keepdim = True)
        return (x - mean) / (std + self.eps) * self.g + self.b
    
class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = LayerNorm(dim)
        self.fn = fn
        
    def forward(self, x,  **kwargs):
        x = self.norm(x)
        return self.fn(x, **kwargs)

class FeedForward(nn.Module):
    def __init__(self, dim, mult = 4, dropout = 0.):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(dim, dim * mult, 1),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Conv2d(dim * mult, dim, 1),
            nn.Dropout(dropout)
        )
        
    def forward(self, x):
        return self.net(x)
    
class Attention(nn.Module):
    def __init__(self, dim, heads, dim_head, dropout = 0. ):
        super().__init__()
        inner_dim = dim_head *  heads
        self.heads = heads
        self.scale = dim_head ** -0.5
        self.attend = nn.Softmax(dim = -1)
        self.to_qkv = nn.Conv2d(dim, inner_dim * 3, 1)
        self.to_out = nn.Sequential(
            nn.Conv2d(inner_dim, dim, 1),
            nn.Dropout(dropout)
        )
        
    def forward(self, x):
        shape = x.shape
        b,c,h,w = x.size()
        b, n, _, y, h = *shape, self.heads
        qkv = self.to_qkv(x).chunk(3, dim = 1)
        q, k, v = map(lambda t: rearrange(t, 'b (h d) x y -> b h (x y) d', h = h), qkv)
        dots = einsum('b h i d, b h j d -> b h i j', q, k) * self.scale
        attn = self.attend(dots)
        out = einsum('b h i j, b h j d -> b h i d', attn, v)
        out = rearrange(out, 'b h (x y) d -> b (h d) x y', h = h ,y = y)
        return self.to_out(out)

class Transformer(nn.Module):
    def __init__(self, dim, depth, heads, dim_head = 64, mlp_mult = 4, dropout = 0.):
        super().__init__()
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                PreNorm(dim, Attention(dim, heads = heads, dim_head = dim_head, dropout = dropout)),
                PreNorm(dim, FeedForward(dim, mlp_mult, dropout = dropout))
            ]))
            
    def forward(self, x):
        for attn, ff in self.layers:
            x = attn(x) + x
            x = ff(x) + x
        return x

class Our-S(nn.Module):
    def __init__(
        self,
        *,
        num_classes,
        s1_emb_dim1 = 96, s1_emb_dim2 = 128, s1_emb_kernel = 7, s1_padding = 3,
        s1_heads = 1, s1_depth = 3, s1_mlp_mult = 4, s1_max_kernel = 2, s1_stride = 2,
        
        s2_emb_dim1 = 192, s2_emb_dim2 = 256, s2_emb_kernel = 3, s2_padding = 1,
        s2_heads = 3, s2_depth = 6, s2_mlp_mult = 4, s2_max_kernel = 2, s2_stride = 1,
        
        s3_emb_dim1 = 256, s3_emb_dim2 = 384, s3_emb_kernel = 3, s3_padding = 1,
        s3_heads = 6, s3_depth = 6, s3_mlp_mult = 4, s3_max_kernel = 2, s3_stride = 1,
        
    ):
        super().__init__()
        kwargs = dict(locals())

        dim = 1      #input image channel
        layers = []

        for prefix in ('s1', 's2', 's3'):
            config, kwargs = group_by_key_prefix_and_remove_prefix(f'{prefix}_', kwargs)

            layers.append(nn.Sequential(
                nn.Conv2d(dim, config['emb_dim1'], kernel_size = config['emb_kernel'], padding = config['padding'], stride = config['stride']),
                nn.GELU(),
                nn.BatchNorm2d(config['emb_dim1']),
                nn.Conv2d(config['emb_dim1'], config['emb_dim2'], kernel_size = config['emb_kernel'], padding = config['padding'], stride = config['stride']),
                nn.GELU(),
                nn.MaxPool2d(kernel_size = config['max_kernel'], stride = config['max_kernel']),
                LayerNorm(config['emb_dim2']),
                Transformer(dim = config['emb_dim2'], depth = config['depth'], heads = config['heads'], mlp_mult = config['mlp_mult'], dropout = dropout)
            ))

            dim = config['emb_dim2']

        self.layers = nn.Sequential(
            *layers,
            nn.AdaptiveAvgPool2d(1),
            Rearrange('... () () -> ...'),
            nn.Linear(dim, num_classes)
        )

    def forward(self, x):
        return self.layers(x)
    
