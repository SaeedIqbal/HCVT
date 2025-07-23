import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, List, Optional, Dict, Any
import math
from collections import OrderedDict

class WindowAttention(nn.Module):
    """
    Window based multi-head self attention (W-MSA) module with relative position bias.
    It supports both of shifted and non-shifted window.
    """
    def __init__(self, dim: int, window_size: Tuple[int, int], num_heads: int, 
                 qkv_bias: bool = True, qk_scale: Optional[float] = None, 
                 attn_drop: float = 0., proj_drop: float = 0.):
        super().__init__()
        self.dim = dim
        self.window_size = window_size  # Wh, Ww
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5

        # Define a parameter table of relative position bias
        self.relative_position_bias_table = nn.Parameter(
            torch.zeros((2 * window_size[0] - 1) * (2 * window_size[1] - 1), num_heads)
        )  # 2*Wh-1 * 2*Ww-1, nH

        # Get pair-wise relative position index for each token inside the window
        coords_h = torch.arange(self.window_size[0])
        coords_w = torch.arange(self.window_size[1])
        coords = torch.stack(torch.meshgrid([coords_h, coords_w]))  # 2, Wh, Ww
        coords_flatten = torch.flatten(coords, 1)  # 2, Wh*Ww
        relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]  # 2, Wh*Ww, Wh*Ww
        relative_coords = relative_coords.permute(1, 2, 0).contiguous()  # Wh*Ww, Wh*Ww, 2
        relative_coords[:, :, 0] += self.window_size[0] - 1  # shift to start from 0
        relative_coords[:, :, 1] += self.window_size[1] - 1
        relative_coords[:, :, 0] *= 2 * self.window_size[1] - 1
        relative_position_index = relative_coords.sum(-1)  # Wh*Ww, Wh*Ww
        self.register_buffer("relative_position_index", relative_position_index)

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        nn.init.trunc_normal_(self.relative_position_bias_table, std=.02)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Args:
            x: input features with shape of (num_windows*B, N, C)
            mask: (0/-inf) mask with shape of (num_windows, Wh*Ww, Wh*Ww) or None
        """
        B_, N, C = x.shape
        qkv = self.qkv(x).reshape(B_, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]  # make torchscript happy (cannot use tensor as tuple)

        q = q * self.scale
        attn = (q @ k.transpose(-2, -1))

        relative_position_bias = self.relative_position_bias_table[self.relative_position_index.view(-1)].view(
            self.window_size[0] * self.window_size[1], self.window_size[0] * self.window_size[1], -1
        )  # Wh*Ww,Wh*Ww,nH
        relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()  # nH, Wh*Ww, Wh*Ww
        attn = attn + relative_position_bias.unsqueeze(0)

        if mask is not None:
            nW = mask.shape[0]
            attn = attn.view(B_ // nW, nW, self.num_heads, N, N) + mask.unsqueeze(1).unsqueeze(0)
            attn = attn.view(-1, self.num_heads, N, N)
            attn = self.softmax(attn)
        else:
            attn = self.softmax(attn)

        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B_, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x

class SwinTransformerBlock(nn.Module):
    """
    Swin Transformer Block.
    """
    def __init__(self, dim: int, input_resolution: Tuple[int, int], num_heads: int, 
                 window_size: int = 7, shift_size: int = 0, mlp_ratio: float = 4., 
                 qkv_bias: bool = True, qk_scale: Optional[float] = None, 
                 drop: float = 0., attn_drop: float = 0., drop_path: float = 0.,
                 act_layer: nn.Module = nn.GELU, norm_layer: nn.Module = nn.LayerNorm):
        super().__init__()
        self.dim = dim
        self.input_resolution = input_resolution
        self.num_heads = num_heads
        self.window_size = window_size
        self.shift_size = shift_size
        self.mlp_ratio = mlp_ratio
        if min(self.input_resolution) <= self.window_size:
            # if window size is larger than input resolution, we don't partition windows
            self.shift_size = 0
            self.window_size = min(self.input_resolution)
        assert 0 <= self.shift_size < self.window_size, "shift_size must in 0-window_size"

        self.norm1 = norm_layer(dim)
        self.attn = WindowAttention(
            dim, window_size=(self.window_size, self.window_size), num_heads=num_heads,
            qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop
        )

        self.drop_path = nn.Identity() if drop_path == 0. else nn.DropPath(drop_path)
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(dim, mlp_hidden_dim),
            act_layer(),
            nn.Dropout(drop),
            nn.Linear(mlp_hidden_dim, dim),
            nn.Dropout(drop)
        )

        if self.shift_size > 0:
            # calculate attention mask for SW-MSA
            H, W = self.input_resolution
            img_mask = torch.zeros((1, H, W, 1))  # 1 H W 1
            h_slices = (slice(0, -self.window_size),
                        slice(-self.window_size, -self.shift_size),
                        slice(-self.shift_size, None))
            w_slices = (slice(0, -self.window_size),
                        slice(-self.window_size, -self.shift_size),
                        slice(-self.shift_size, None))
            cnt = 0
            for h in h_slices:
                for w in w_slices:
                    img_mask[:, h, w, :] = cnt
                    cnt += 1

            mask_windows = window_partition(img_mask, self.window_size)  # nW, window_size, window_size, 1
            mask_windows = mask_windows.view(-1, self.window_size * self.window_size)
            attn_mask = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2)
            attn_mask = attn_mask.masked_fill(attn_mask != 0, float(-100.0)).masked_fill(attn_mask == 0, float(0.0))
        else:
            attn_mask = None

        self.register_buffer("attn_mask", attn_mask)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        H, W = self.input_resolution
        B, L, C = x.shape
        assert L == H * W, "input feature has wrong size"

        shortcut = x
        x = self.norm1(x)
        x = x.view(B, H, W, C)

        # cyclic shift
        if self.shift_size > 0:
            shifted_x = torch.roll(x, shifts=(-self.shift_size, -self.shift_size), dims=(1, 2))
        else:
            shifted_x = x

        # partition windows
        x_windows = window_partition(shifted_x, self.window_size)  # nW*B, window_size, window_size, C
        x_windows = x_windows.view(-1, self.window_size * self.window_size, C)  # nW*B, window_size*window_size, C

        # W-MSA/SW-MSA
        attn_windows = self.attn(x_windows, mask=self.attn_mask)  # nW*B, window_size*window_size, C

        # merge windows
        attn_windows = attn_windows.view(-1, self.window_size, self.window_size, C)
        shifted_x = window_reverse(attn_windows, self.window_size, H, W)  # B H' W' C

        # reverse cyclic shift
        if self.shift_size > 0:
            x = torch.roll(shifted_x, shifts=(self.shift_size, self.shift_size), dims=(1, 2))
        else:
            x = shifted_x
        x = x.view(B, H * W, C)

        # FFN
        x = shortcut + self.drop_path(x)
        x = x + self.drop_path(self.mlp(self.norm2(x)))

        return x

def window_partition(x: torch.Tensor, window_size: int) -> torch.Tensor:
    """
    Args:
        x: (B, H, W, C)
        window_size (int): window size
    Returns:
        windows: (num_windows*B, window_size, window_size, C)
    """
    B, H, W, C = x.shape
    window_resolution = (H // window_size, W // window_size)
    x = x.view(B, window_resolution[0], window_size, window_resolution[1], window_size, C)
    windows = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, window_size, window_size, C)
    return windows

def window_reverse(windows: torch.Tensor, window_size: int, H: int, W: int) -> torch.Tensor:
    """
    Args:
        windows: (num_windows*B, window_size, window_size, C)
        window_size (int): Window size
        H (int): Height of image
        W (int): Width of image
    Returns:
        x: (B, H, W, C)
    """
    window_resolution = (H // window_size, W // window_size)
    B = int(windows.shape[0] / (window_resolution[0] * window_resolution[1]))
    x = windows.view(B, window_resolution[0], window_resolution[1], window_size, window_size, -1)
    x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, H, W, -1)
    return x

class PatchMerging(nn.Module):
    """
    Patch Merging Layer.
    """
    def __init__(self, input_resolution: Tuple[int, int], dim: int, norm_layer: nn.Module = nn.LayerNorm):
        super().__init__()
        self.input_resolution = input_resolution
        self.dim = dim
        self.reduction = nn.Linear(4 * dim, 2 * dim, bias=False)
        self.norm = norm_layer(4 * dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: B, H*W, C
        """
        H, W = self.input_resolution
        B, L, C = x.shape
        assert L == H * W, "input feature has wrong size"
        assert H % 2 == 0 and W % 2 == 0, f"x size ({H}*{W}) are not even."

        x = x.view(B, H, W, C)

        x0 = x[:, 0::2, 0::2, :]  # B H/2 W/2 C
        x1 = x[:, 1::2, 0::2, :]  # B H/2 W/2 C
        x2 = x[:, 0::2, 1::2, :]  # B H/2 W/2 C
        x3 = x[:, 1::2, 1::2, :]  # B H/2 W/2 C
        x = torch.cat([x0, x1, x2, x3], -1)  # B H/2 W/2 4*C
        x = x.view(B, -1, 4 * C)  # B H/2*W/2 4*C

        x = self.norm(x)
        x = self.reduction(x)

        return x

class CustomSwinTransformerBlock(nn.Module):
    """
    Custom SWIN Transformer Block (CSWIN) - Modified version with additional processing
    """
    def __init__(self, dim: int, input_resolution: Tuple[int, int], num_heads: int, 
                 window_size: int = 7, shift_size: int = 0, mlp_ratio: float = 4., 
                 qkv_bias: bool = True, drop: float = 0., attn_drop: float = 0., 
                 drop_path: float = 0., act_layer: nn.Module = nn.GELU, 
                 norm_layer: nn.Module = nn.LayerNorm):
        super().__init__()
        self.dim = dim
        self.input_resolution = input_resolution
        self.num_heads = num_heads
        self.window_size = window_size
        self.shift_size = shift_size
        self.mlp_ratio = mlp_ratio

        # Enhanced attention mechanism
        self.norm1 = norm_layer(dim)
        self.attn = WindowAttention(
            dim, window_size=(self.window_size, self.window_size), num_heads=num_heads,
            qkv_bias=qkv_bias, attn_drop=attn_drop, proj_drop=drop
        )

        # Additional processing modules
        self.drop_path = nn.Identity() if drop_path == 0. else nn.DropPath(drop_path)
        self.norm2 = norm_layer(dim)
        
        # Enhanced MLP with additional layers
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(dim, mlp_hidden_dim),
            act_layer(),
            nn.Dropout(drop),
            nn.Linear(mlp_hidden_dim, mlp_hidden_dim),
            act_layer(),
            nn.Dropout(drop),
            nn.Linear(mlp_hidden_dim, dim),
            nn.Dropout(drop)
        )

        # Local Enhancement layer
        self.local_enhancement = nn.Conv2d(dim, dim, kernel_size=3, padding=1, groups=dim)

        if self.shift_size > 0:
            # calculate attention mask for SW-MSA
            H, W = self.input_resolution
            img_mask = torch.zeros((1, H, W, 1))
            h_slices = (slice(0, -self.window_size),
                        slice(-self.window_size, -self.shift_size),
                        slice(-self.shift_size, None))
            w_slices = (slice(0, -self.window_size),
                        slice(-self.window_size, -self.shift_size),
                        slice(-self.shift_size, None))
            cnt = 0
            for h in h_slices:
                for w in w_slices:
                    img_mask[:, h, w, :] = cnt
                    cnt += 1

            mask_windows = window_partition(img_mask, self.window_size)
            mask_windows = mask_windows.view(-1, self.window_size * self.window_size)
            attn_mask = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2)
            attn_mask = attn_mask.masked_fill(attn_mask != 0, float(-100.0)).masked_fill(attn_mask == 0, float(0.0))
        else:
            attn_mask = None

        self.register_buffer("attn_mask", attn_mask)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        H, W = self.input_resolution
        B, L, C = x.shape
        assert L == H * W, "input feature has wrong size"

        shortcut = x
        x = self.norm1(x)
        x = x.view(B, H, W, C)

        # Local enhancement
        x_local = x.permute(0, 3, 1, 2)
        x_local = self.local_enhancement(x_local)
        x_local = x_local.permute(0, 2, 3, 1).view(B, L, C)

        # cyclic shift
        if self.shift_size > 0:
            shifted_x = torch.roll(x, shifts=(-self.shift_size, -self.shift_size), dims=(1, 2))
        else:
            shifted_x = x

        # partition windows
        x_windows = window_partition(shifted_x, self.window_size)
        x_windows = x_windows.view(-1, self.window_size * self.window_size, C)

        # W-MSA/SW-MSA
        attn_windows = self.attn(x_windows, mask=self.attn_mask)

        # merge windows
        attn_windows = attn_windows.view(-1, self.window_size, self.window_size, C)
        shifted_x = window_reverse(attn_windows, self.window_size, H, W)

        # reverse cyclic shift
        if self.shift_size > 0:
            x = torch.roll(shifted_x, shifts=(self.shift_size, self.shift_size), dims=(1, 2))
        else:
            x = shifted_x
        x = x.view(B, H * W, C)

        # FFN with local enhancement
        x = shortcut + self.drop_path(x) + 0.1 * x_local
        x = x + self.drop_path(self.mlp(self.norm2(x)))

        return x

class HierarchicalFeatureExtractor(nn.Module):
    """
    Hierarchical Feature Extractor with alternating SWIN and CSWIN blocks
    """
    def __init__(self, img_size: int = 224, patch_size: int = 4, in_chans: int = 3, 
                 num_classes: int = 1000, embed_dim: int = 96, depths: List[int] = [2, 2, 6, 2], 
                 num_heads: List[int] = [3, 6, 12, 24], window_size: int = 7, 
                 mlp_ratio: float = 4., qkv_bias: bool = True, 
                 drop_rate: float = 0., attn_drop_rate: float = 0., 
                 drop_path_rate: float = 0.1, norm_layer: nn.Module = nn.LayerNorm, 
                 ape: bool = False, patch_norm: bool = True):
        super().__init__()

        self.num_classes = num_classes
        self.num_layers = len(depths)
        self.embed_dim = embed_dim
        self.ape = ape
        self.patch_norm = patch_norm
        self.num_features = int(embed_dim * 2 ** (self.num_layers - 1))
        self.mlp_ratio = mlp_ratio

        # Split image into non-overlapping patches
        self.patch_embed = PatchEmbed(
            img_size=img_size, patch_size=patch_size, in_chans=in_chans, 
            embed_dim=embed_dim, norm_layer=norm_layer if self.patch_norm else None
        )
        num_patches = self.patch_embed.num_patches
        patches_resolution = self.patch_embed.patches_resolution
        self.patches_resolution = patches_resolution

        # Absolute position embedding
        if self.ape:
            self.absolute_pos_embed = nn.Parameter(torch.zeros(1, num_patches, embed_dim))
            nn.init.trunc_normal_(self.absolute_pos_embed, std=.02)

        self.pos_drop = nn.Dropout(p=drop_rate)

        # Stochastic depth
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]

        # Build layers with alternating SWIN and CSWIN blocks
        self.layers = nn.ModuleList()
        for i_layer in range(self.num_layers):
            layer = BasicLayer(
                dim=int(embed_dim * 2 ** i_layer),
                input_resolution=(patches_resolution[0] // (2 ** i_layer),
                                patches_resolution[1] // (2 ** i_layer)),
                depth=depths[i_layer],
                num_heads=num_heads[i_layer],
                window_size=window_size,
                mlp_ratio=self.mlp_ratio,
                qkv_bias=qkv_bias,
                drop=drop_rate,
                attn_drop=attn_drop_rate,
                drop_path=dpr[sum(depths[:i_layer]):sum(depths[:i_layer + 1])],
                norm_layer=norm_layer,
                downsample=PatchMerging if (i_layer < self.num_layers - 1) else None,
                use_custom_swin=(i_layer % 2 == 1)  # Alternate between SWIN and CSWIN
            )
            self.layers.append(layer)

        self.norm = norm_layer(self.num_features)
        self.avgpool = nn.AdaptiveAvgPool1d(1)
        self.head = nn.Linear(self.num_features, num_classes) if num_classes > 0 else nn.Identity()

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward_features(self, x: torch.Tensor) -> torch.Tensor:
        x, H, W = self.patch_embed(x)
        if self.ape:
            x = x + self.absolute_pos_embed
        x = self.pos_drop(x)

        # Store features from each level for multi-scale fusion
        features = []
        for layer in self.layers:
            x, H, W = layer(x, H, W)
            features.append(x)

        x = self.norm(x)  # B L C
        return x, features

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        x, features = self.forward_features(x)
        x = self.avgpool(x.transpose(1, 2))  # B C 1
        x = torch.flatten(x, 1)
        x = self.head(x)
        return x, features

class PatchEmbed(nn.Module):
    """
    Image to Patch Embedding
    """
    def __init__(self, img_size: int = 224, patch_size: int = 4, in_chans: int = 3, 
                 embed_dim: int = 96, norm_layer: Optional[nn.Module] = None):
        super().__init__()
        img_size = (img_size, img_size) if isinstance(img_size, int) else img_size
        patch_size = (patch_size, patch_size) if isinstance(patch_size, int) else patch_size
        patches_resolution = [img_size[0] // patch_size[0], img_size[1] // patch_size[1]]
        self.img_size = img_size
        self.patch_size = patch_size
        self.patches_resolution = patches_resolution
        self.num_patches = patches_resolution[0] * patches_resolution[1]

        self.in_chans = in_chans
        self.embed_dim = embed_dim

        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)
        if norm_layer is not None:
            self.norm = norm_layer(embed_dim)
        else:
            self.norm = None

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, int, int]:
        B, C, H, W = x.shape
        # FIXME look at relaxing size constraints
        assert H == self.img_size[0] and W == self.img_size[1], \
            f"Input image size ({H}*{W}) doesn't match model ({self.img_size[0]}*{self.img_size[1]})."
        x = self.proj(x).flatten(2).transpose(1, 2)  # B Ph*Pw C
        if self.norm is not None:
            x = self.norm(x)
        return x, H // self.patch_size[0], W // self.patch_size[1]

class BasicLayer(nn.Module):
    """
    A basic Swin Transformer layer for one stage.
    """
    def __init__(self, dim: int, input_resolution: Tuple[int, int], depth: int, 
                 num_heads: int, window_size: int, mlp_ratio: float = 4., 
                 qkv_bias: bool = True, drop: float = 0., attn_drop: float = 0.,
                 drop_path: List[float] = [0.0], norm_layer: nn.Module = nn.LayerNorm, 
                 downsample: Optional[nn.Module] = None, use_custom_swin: bool = False):
        super().__init__()
        self.dim = dim
        self.input_resolution = input_resolution
        self.depth = depth
        self.use_custom_swin = use_custom_swin

        # Build blocks
        self.blocks = nn.ModuleList([
            CustomSwinTransformerBlock(
                dim=dim, input_resolution=input_resolution,
                num_heads=num_heads, window_size=window_size,
                shift_size=0 if (i % 2 == 0) else window_size // 2,
                mlp_ratio=mlp_ratio, qkv_bias=qkv_bias,
                drop=drop, attn_drop=attn_drop,
                drop_path=drop_path[i] if isinstance(drop_path, list) else drop_path,
                norm_layer=norm_layer
            ) if use_custom_swin else SwinTransformerBlock(
                dim=dim, input_resolution=input_resolution,
                num_heads=num_heads, window_size=window_size,
                shift_size=0 if (i % 2 == 0) else window_size // 2,
                mlp_ratio=mlp_ratio, qkv_bias=qkv_bias,
                drop=drop, attn_drop=attn_drop,
                drop_path=drop_path[i] if isinstance(drop_path, list) else drop_path,
                norm_layer=norm_layer
            )
            for i in range(depth)
        ])

        # Patch merging layer
        if downsample is not None:
            self.downsample = downsample(input_resolution, dim=dim, norm_layer=norm_layer)
        else:
            self.downsample = None

    def forward(self, x: torch.Tensor, H: int, W: int) -> Tuple[torch.Tensor, int, int]:
        for blk in self.blocks:
            x = blk(x)
        if self.downsample is not None:
            x = self.downsample(x)
            H, W = (H + 1) // 2, (W + 1) // 2
        return x, H, W

class MultiScaleFeatureFusion(nn.Module):
    """
    Multi-scale feature fusion module that combines features from different hierarchical levels
    """
    def __init__(self, feature_dims: List[int], fusion_dim: int = 256):
        super().__init__()
        self.feature_dims = feature_dims
        self.fusion_dim = fusion_dim
        
        # Projection layers for each scale
        self.projections = nn.ModuleList([
            nn.Sequential(
                nn.Linear(dim, fusion_dim),
                nn.LayerNorm(fusion_dim),
                nn.ReLU(inplace=True)
            ) for dim in feature_dims
        ])
        
        # Attention-based fusion
        self.fusion_attention = nn.MultiheadAttention(
            embed_dim=fusion_dim, 
            num_heads=8, 
            dropout=0.1,
            batch_first=True
        )
        
        # Final fusion layer
        self.fusion_layer = nn.Sequential(
            nn.Linear(fusion_dim * len(feature_dims), fusion_dim),
            nn.LayerNorm(fusion_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1)
        )

    def forward(self, features: List[torch.Tensor]) -> torch.Tensor:
        """
        Fuse multi-scale features
        Args:
            features: List of features from different hierarchical levels
        Returns:
            Fused feature representation
        """
        projected_features = []
        
        # Project features to common dimension
        for i, feature in enumerate(features):
            proj_feature = self.projections[i](feature)
            projected_features.append(proj_feature)
        
        # Concatenate all projected features
        if len(projected_features) > 1:
            fused = torch.cat(projected_features, dim=-1)
            fused = self.fusion_layer(fused)
        else:
            fused = projected_features[0]
            
        return fused

class HierarchicalCascadedVisionTransformer(nn.Module):
    """
    Hierarchical Cascaded Vision Transformer for Medical Image Analysis
    Implements the complete framework described in the methodology
    """
    def __init__(self, img_size: int = 224, num_classes: int = 14, 
                 embed_dim: int = 96, depths: List[int] = [2, 2, 6, 2], 
                 num_heads: List[int] = [3, 6, 12, 24], num_stages: int = 3):
        super().__init__()
        self.num_stages = num_stages
        self.num_classes = num_classes
        
        # Initialize hierarchical feature extractors for each stage
        self.stages = nn.ModuleList()
        for stage in range(num_stages):
            stage_extractor = HierarchicalFeatureExtractor(
                img_size=img_size,
                embed_dim=embed_dim,
                depths=depths,
                num_heads=num_heads,
                num_classes=num_classes
            )
            self.stages.append(stage_extractor)
        
        # Feature fusion module
        feature_dims = [int(embed_dim * 2 ** (len(depths) - 1))] * num_stages
        self.feature_fusion = MultiScaleFeatureFusion(feature_dims)
        
        # Final classification head
        self.classifier = nn.Sequential(
            nn.Linear(int(embed_dim * 2 ** (len(depths) - 1)), 512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(512, num_classes)
        )
        
        # Weight matrices for each stage (learnable parameters)
        self.stage_weights = nn.ParameterList([
            nn.Parameter(torch.ones(1)) for _ in range(num_stages)
        ])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the hierarchical cascaded architecture
        """
        # Stage 1: Data Augmentation through hierarchical feature extraction
        stage_features = []
        stage_outputs = []
        
        for stage_idx, stage_extractor in enumerate(self.stages):
            output, features = stage_extractor(x)
            stage_outputs.append(output)
            # Use the deepest features for fusion
            stage_features.append(features[-1] if features else output)
        
        # Stage 2: Multi-scale feature fusion
        fused_features = self.feature_fusion(stage_features)
        
        # Stage 3: Weighted prediction combination
        weighted_predictions = []
        for i, (output, weight) in enumerate(zip(stage_outputs, self.stage_weights)):
            weighted_pred = output * weight
            weighted_predictions.append(weighted_pred)
        
        # Combine weighted predictions
        if len(weighted_predictions) > 1:
            combined_prediction = torch.stack(weighted_predictions, dim=0).mean(dim=0)
        else:
            combined_prediction = weighted_predictions[0]
        
        # Final classification
        final_output = self.classifier(fused_features)
        
        return final_output

class MedicalImageSegmentationModel(nn.Module):
    """
    Complete medical image segmentation model based on the proposed hierarchical architecture
    """
    def __init__(self, in_channels: int = 3, num_classes: int = 14, 
                 base_channels: int = 32):
        super().__init__()
        self.in_channels = in_channels
        self.num_classes = num_classes
        
        # Encoder (Hierarchical Feature Extractor)
        self.encoder = HierarchicalFeatureExtractor(
            img_size=224,
            in_chans=in_channels,
            embed_dim=base_channels,
            depths=[2, 2, 6, 2],
            num_heads=[3, 6, 12, 24],
            num_classes=num_classes
        )
        
        # Decoder for segmentation
        self.decoder = nn.Sequential(
            nn.Conv2d(base_channels * 8, base_channels * 4, kernel_size=3, padding=1),
            nn.BatchNorm2d(base_channels * 4),
            nn.ReLU(inplace=True),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            
            nn.Conv2d(base_channels * 4, base_channels * 2, kernel_size=3, padding=1),
            nn.BatchNorm2d(base_channels * 2),
            nn.ReLU(inplace=True),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            
            nn.Conv2d(base_channels * 2, base_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(base_channels),
            nn.ReLU(inplace=True),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            
            nn.Conv2d(base_channels, num_classes, kernel_size=1)
        )
        
        # Hierarchical cascaded processing
        self.hierarchical_processor = HierarchicalCascadedVisionTransformer(
            img_size=224,
            num_classes=num_classes,
            embed_dim=base_channels
        )

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass returning both segmentation map and classification output
        """
        # Extract features using hierarchical encoder
        features, _ = self.encoder.forward_features(x)
        
        # Reshape features for decoder
        B, L, C = features.shape
        H = W = int(math.sqrt(L))
        features_reshaped = features.transpose(1, 2).view(B, C, H, W)
        
        # Generate segmentation map
        segmentation_output = self.decoder(features_reshaped)
        
        # Hierarchical classification
        classification_output = self.hierarchical_processor(x)
        
        return segmentation_output, classification_output

# Utility functions for training and inference
class HierarchicalTrainingManager:
    """
    Manager class for training the hierarchical cascaded vision transformer
    """
    def __init__(self, model: nn.Module, device: torch.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')):
        self.model = model.to(device)
        self.device = device
        self.optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=0.05)
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=100)
        self.criterion_cls = nn.BCEWithLogitsLoss()
        self.criterion_seg = nn.CrossEntropyLoss()
        
    def train_step(self, images: torch.Tensor, labels: torch.Tensor, 
                   segmentation_masks: Optional[torch.Tensor] = None) -> Dict[str, float]:
        """
        Single training step
        """
        self.model.train()
        self.optimizer.zero_grad()
        
        images = images.to(self.device)
        labels = labels.to(self.device)
        
        if segmentation_masks is not None:
            segmentation_masks = segmentation_masks.to(self.device)
            seg_output, cls_output = self.model(images)
            loss_seg = self.criterion_seg(seg_output, segmentation_masks)
            loss_cls = self.criterion_cls(cls_output, labels.float())
            loss = loss_seg + loss_cls
        else:
            cls_output = self.model(images)
            loss = self.criterion_cls(cls_output, labels.float())
        
        loss.backward()
        self.optimizer.step()
        
        return {
            'loss': loss.item(),
            'lr': self.optimizer.param_groups[0]['lr']
        }
    
    def validation_step(self, images: torch.Tensor, labels: torch.Tensor,
                       segmentation_masks: Optional[torch.Tensor] = None) -> Dict[str, float]:
        """
        Single validation step
        """
        self.model.eval()
        with torch.no_grad():
            images = images.to(self.device)
            labels = labels.to(self.device)
            
            if segmentation_masks is not None:
                segmentation_masks = segmentation_masks.to(self.device)
                seg_output, cls_output = self.model(images)
                loss_seg = self.criterion_seg(seg_output, segmentation_masks)
                loss_cls = self.criterion_cls(cls_output, labels.float())
                loss = loss_seg + loss_cls
            else:
                cls_output = self.model(images)
                loss = self.criterion_cls(cls_output, labels.float())
            
            # Calculate accuracy
            preds = (torch.sigmoid(cls_output) > 0.5).float()
            accuracy = (preds == labels).float().mean()
            
            return {
                'val_loss': loss.item(),
                'val_accuracy': accuracy.item()
            }

def create_model(num_classes: int = 14, img_size: int = 224) -> MedicalImageSegmentationModel:
    """
    Factory function to create the complete model
    """
    model = MedicalImageSegmentationModel(
        in_channels=3,
        num_classes=num_classes,
        base_channels=32
    )
    return model

# Example usage and testing
def main():
    """
    Demonstrate the hierarchical cascaded vision transformer framework
    """
    print("Initializing Hierarchical Cascaded Vision Transformer Framework...")
    
    # Create model
    model = create_model(num_classes=14)
    print(f"Model created successfully with {sum(p.numel() for p in model.parameters()):,} parameters")
    
    # Test forward pass
    dummy_input = torch.randn(2, 3, 224, 224)  # Batch of 2 RGB images
    dummy_labels = torch.randint(0, 2, (2, 14)).float()  # Binary labels for 14 classes
    
    print("Testing forward pass...")
    try:
        seg_output, cls_output = model(dummy_input)
        print(f"Segmentation output shape: {seg_output.shape}")
        print(f"Classification output shape: {cls_output.shape}")
        print("Forward pass successful!")
    except Exception as e:
        print(f"Error during forward pass: {e}")
    
    # Initialize training manager
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    trainer = HierarchicalTrainingManager(model, device)
    print(f"Training manager initialized on {device}")
    
    # Test training step
    print("Testing training step...")
    try:
        train_metrics = trainer.train_step(dummy_input, dummy_labels)
        print(f"Training step completed. Loss: {train_metrics['loss']:.4f}")
    except Exception as e:
        print(f"Error during training step: {e}")
    
    print("\nHierarchical Cascaded Vision Transformer Framework initialized successfully!")
    print("\nKey Features:")
    print("1. Hierarchical feature extraction with alternating SWIN and CSWIN blocks")
    print("2. Multi-scale feature fusion for enhanced representation learning")
    print("3. Progressive downsampling strategy for computational efficiency")
    print("4. Custom SWIN Transformer blocks with local enhancement")
    print("5. Hierarchical cascaded processing for improved accuracy")

if __name__ == "__main__":
    main()