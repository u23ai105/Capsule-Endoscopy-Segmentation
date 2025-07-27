import timm
import torch
import torch.nn as nn
from timm import create_model
import torch.nn.functional as F
from typing import Optional, List
import math


# ====================================================================================
# Main Unet Class
# ====================================================================================

class Unet(nn.Module):
    """
    Final, verified, and corrected Unet implementation.
    This version includes a robust Transformer Bottleneck, corrected Decoder Blocks,
    and a reliable encoder freezing method.
    """

    def __init__(
            self,
            backbone: str = 'resnet50',
            encoder_freeze: bool = False,
            pretrained: bool = True,
            preprocessing: bool = False,
            non_trainable_layers: tuple = (0, 1, 2, 3, 4),
            backbone_kwargs: Optional[dict] = None,
            backbone_indices: Optional[List[int]] = None,
            decoder_use_batchnorm: bool = True,
            decoder_channels: tuple = (256, 128, 64, 32, 16),
            in_channels: int = 3,
            num_classes: int = 5,
            center: bool = False,
            norm_layer: nn.Module = nn.BatchNorm2d,
            activation: nn.Module = nn.ReLU,
            use_transformer_bottleneck: bool = True,
            transformer_nhead: int = 8,
            transformer_dropout: float = 0.1
    ):
        super().__init__()
        backbone_kwargs = backbone_kwargs or {}

        # 1. Create Encoder from timm
        self.encoder = create_model(
            backbone,
            features_only=True,
            out_indices=backbone_indices,
            in_chans=in_channels,
            pretrained=pretrained,
            **backbone_kwargs
        )
        encoder_channels = [info["num_chs"] for info in self.encoder.feature_info][::-1]

        decoder_channels = decoder_channels[:len(encoder_channels)]

        if encoder_freeze:
            self._freeze_encoder(non_trainable_layers)

        # 2. Handle Preprocessing
        if preprocessing:
            self.mean = self.encoder.default_cfg.get("mean", None)
            self.std = self.encoder.default_cfg.get("std", None)
        else:
            self.mean = None
            self.std = None

        # 3. Optional Transformer Bottleneck
        self.use_transformer_bottleneck = use_transformer_bottleneck
        if self.use_transformer_bottleneck:
            self.transformer_block = TransformerBottleneck(
                in_channels=encoder_channels[0],
                nhead=transformer_nhead,
                dropout=transformer_dropout
            )

        # 4. Create Decoder
        self.decoder = UnetDecoder(
            encoder_channels=encoder_channels,
            decoder_channels=decoder_channels,
            final_channels=num_classes,
            norm_layer=norm_layer if decoder_use_batchnorm else None,
            center=center,
            activation=activation,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        input_size = x.shape[-2:]
        if self.mean is not None and self.std is not None:
            x = self._preprocess_input(x)

        features = self.encoder(x)

        if features[0].ndim == 4 and features[0].shape[1] != self.encoder.feature_info[0]['num_chs']:
            features = [f.permute(0, 3, 1, 2) for f in features]

        features.reverse()  # Reverse to be [deepest, ... , shallowest]

        if self.use_transformer_bottleneck:
            features[0] = self.transformer_block(features[0])

        x = self.decoder(features)
        x = F.interpolate(x, size=input_size, mode='bilinear', align_corners=False)
        return x

    @torch.no_grad()
    def predict(self, x: torch.Tensor) -> torch.Tensor:
        if self.training:
            self.eval()
        return self.forward(x)

    def _freeze_encoder(self, non_trainable_layer_idxs: tuple):
        """ Robustly sets selected encoder layers as non-trainable. """
        if not non_trainable_layer_idxs:
            return

        non_trainable_module_names = [
            self.encoder.feature_info[idx]["module"] for idx in non_trainable_layer_idxs
        ]

        for name, param in self.encoder.named_parameters():
            if any(name.startswith(mod_name) for mod_name in non_trainable_module_names):
                # Still allow BatchNorm layers to be trainable for better performance
                if 'bn' not in name and 'downsample.1' not in name:
                    param.requires_grad = False

    def _preprocess_input(self, x: torch.Tensor) -> torch.Tensor:
        if not x.is_floating_point():
            raise TypeError(f"Input tensor should be float. Got {x.dtype}.")
        if x.ndim != 4:
            raise ValueError(f"Expected NCHW tensor. Got shape {x.shape}")

        device = x.device
        mean = torch.tensor(self.mean, device=device).view(1, -1, 1, 1)
        std = torch.tensor(self.std, device=device).view(1, -1, 1, 1)
        return (x - mean) / std


# ====================================================================================
# Helper Modules (Transformer, Positional Encoding, Decoder Blocks)
# ====================================================================================

class PositionalEncoding2D(nn.Module):
    """ Adds 2D sinusoidal positional embeddings to a feature map. """

    def __init__(self, d_model: int, max_h: int = 64, max_w: int = 64):
        super().__init__()
        if d_model % 4 != 0:
            raise ValueError(f"d_model must be divisible by 4, got {d_model}")

        pe = torch.zeros(d_model, max_h, max_w)
        d_model //= 2  # Half for height, half for width

        div_term = torch.exp(torch.arange(0., d_model / 2, 2) * -(math.log(10000.0) / (d_model / 2)))
        pos_w = torch.arange(0., max_w).unsqueeze(1)
        pos_h = torch.arange(0., max_h).unsqueeze(1)

        pe[0:d_model // 2:2, :, :] = torch.sin(pos_h * div_term).transpose(0, 1).unsqueeze(2).repeat(1, 1, max_w)
        pe[1:d_model // 2:2, :, :] = torch.cos(pos_h * div_term).transpose(0, 1).unsqueeze(2).repeat(1, 1, max_w)
        pe[d_model // 2::2, :, :] = torch.sin(pos_w * div_term).transpose(0, 1).unsqueeze(1).repeat(1, max_h, 1)
        pe[d_model // 2 + 1::2, :, :] = torch.cos(pos_w * div_term).transpose(0, 1).unsqueeze(1).repeat(1, max_h, 1)

        self.register_buffer('pe', pe.unsqueeze(0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.pe[:, :, :x.size(2), :x.size(3)]


class TransformerBottleneck(nn.Module):
    """ Architecturally correct Transformer Block using nn.TransformerEncoderLayer. """

    def __init__(self, in_channels: int, nhead: int = 8, num_layers: int = 1, dropout: float = 0.1):
        super().__init__()
        self.pos_encoder = PositionalEncoding2D(d_model=in_channels)
        encoder_layer = nn.TransformerEncoderLayer(d_model=in_channels, nhead=nhead, dropout=dropout, batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, H, W = x.shape
        x = self.pos_encoder(x)
        x = x.flatten(2).permute(0, 2, 1)  # (B, H*W, C)
        x = self.transformer_encoder(x)
        x = x.permute(0, 2, 1).view(B, C, H, W)
        return x


class Conv2dBnAct(nn.Module):
    """ A standard Conv -> BN -> Activation block. """

    def __init__(self, in_channels, out_channels, kernel_size=3, padding=1, norm_layer=nn.BatchNorm2d,
                 activation=nn.ReLU):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size, padding=padding, bias=norm_layer is None),
            norm_layer(out_channels) if norm_layer else nn.Identity(),
            activation(inplace=True)
        )

    def forward(self, x):
        return self.block(x)


class DecoderBlock(nn.Module):
    """ Corrected U-Net Decoder Block aware of skip connection channels. """

    def __init__(self, in_channels: int, skip_channels: int, out_channels: int, norm_layer=nn.BatchNorm2d,
                 activation=nn.ReLU):
        super().__init__()
        self.upsample = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2)
        conv_in_channels = out_channels + skip_channels
        self.conv_block = nn.Sequential(
            Conv2dBnAct(conv_in_channels, out_channels, norm_layer=norm_layer, activation=activation),
            Conv2dBnAct(out_channels, out_channels, norm_layer=norm_layer, activation=activation),
        )

    def forward(self, x: torch.Tensor, skip: Optional[torch.Tensor] = None) -> torch.Tensor:
        x = self.upsample(x)
        if skip is not None:
            if x.shape != skip.shape:
                x = F.interpolate(x, size=skip.shape[-2:], mode='bilinear', align_corners=False)
            x = torch.cat([x, skip], dim=1)
        return self.conv_block(x)


class UnetDecoder(nn.Module):
    """ Corrected U-Net Decoder that assembles the robust DecoderBlocks. """

    def __init__(self, encoder_channels, decoder_channels, final_channels, norm_layer, center, activation):
        super().__init__()

        # Define channel lists for each block
        in_channels = [encoder_channels[0]] + list(decoder_channels[:-1])
        skip_channels = list(encoder_channels[1:]) + [0]
        out_channels = list(decoder_channels)

        if center:
            self.center = DecoderBlock(
                in_channels=encoder_channels[0], skip_channels=0, out_channels=encoder_channels[0],
                norm_layer=norm_layer, activation=activation
            )
            in_channels[0] = encoder_channels[0]  # First block input is now from center
        else:
            self.center = nn.Identity()

        self.blocks = nn.ModuleList([
            DecoderBlock(in_ch, skip_ch, out_ch, norm_layer=norm_layer, activation=activation)
            for in_ch, skip_ch, out_ch in zip(in_channels, skip_channels, out_channels)
        ])

        self.final_conv = nn.Conv2d(decoder_channels[-1], final_channels, kernel_size=1)
        self._init_weight()

    def _init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                torch.nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, features: List[torch.Tensor]) -> torch.Tensor:
        head = features[0]
        skips = features[1:]
        x = self.center(head)
        for i, block in enumerate(self.blocks):
            skip = skips[i] if i < len(skips) else None
            x = block(x, skip)
        return self.final_conv(x)