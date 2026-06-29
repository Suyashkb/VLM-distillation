"""
Compact student VLM: MobileViTv2 vision encoder + 4-layer cross-attention decoder.

Architecture:
  - Vision backbone: mobilevitv2_100 from timm (~20M params, 512-dim features)
  - Question encoder: learned embedding + positional encoding
  - Fusion decoder: 4 transformer layers with cross-attention (image → text)
  - VQA head: linear over num_answers
  - Projection heads: adapt student feature dims → teacher feature dims for distillation

Total: ~60M params
"""

import math
from typing import List, Optional, Tuple

import timm
import torch
import torch.nn as nn
import torch.nn.functional as F


class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 512, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        pe = torch.zeros(max_len, d_model)
        pos = torch.arange(max_len).unsqueeze(1).float()
        div = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer("pe", pe.unsqueeze(0))  # (1, max_len, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.dropout(x + self.pe[:, : x.size(1)])


class StudentDecoder(nn.Module):
    """4-layer transformer decoder with self-attn + cross-attn (attends to image features)."""

    def __init__(
        self,
        d_model: int,
        nhead: int,
        ffn_dim: int,
        num_layers: int,
        dropout: float = 0.1,
    ):
        super().__init__()
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=ffn_dim,
            dropout=dropout,
            batch_first=True,
            norm_first=True,   # pre-norm for stability
        )
        self.layers = nn.ModuleList([
            nn.TransformerDecoderLayer(
                d_model=d_model,
                nhead=nhead,
                dim_feedforward=ffn_dim,
                dropout=dropout,
                batch_first=True,
                norm_first=True,
            )
            for _ in range(num_layers)
        ])
        self.norm = nn.LayerNorm(d_model)

    def forward(
        self,
        tgt: torch.Tensor,
        memory: torch.Tensor,
        tgt_key_padding_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        x = tgt
        intermediate_feats = []
        for layer in self.layers:
            x = layer(x, memory, tgt_key_padding_mask=tgt_key_padding_mask)
            intermediate_feats.append(x)
        return self.norm(x), intermediate_feats


class StudentVLM(nn.Module):
    def __init__(
        self,
        vision_backbone: str = "mobilevitv2_100",
        vision_embed_dim: int = 512,
        decoder_layers: int = 4,
        decoder_heads: int = 8,
        decoder_ffn_dim: int = 1024,
        dropout: float = 0.1,
        num_answers: int = 3130,
        vocab_size: int = 51289,    # Florence-2 tokenizer vocab size
        max_question_len: int = 64,
        teacher_feat_dim: int = 768, # Florence-2 hidden dim for projection heads
    ):
        super().__init__()
        self.d_model = vision_embed_dim

        # Vision encoder: pretrained MobileViTv2 (frozen after init optionally)
        backbone = timm.create_model(
            vision_backbone, pretrained=True, num_classes=0, global_pool=""
        )
        self.vision_encoder = backbone
        # Determine backbone output channels via a dummy pass
        with torch.no_grad():
            dummy = torch.zeros(1, 3, 224, 224)
            out = backbone(dummy)
            # timm returns (B, H, W, C) for some models, (B, C, H, W) for others
            if out.dim() == 4:
                if out.shape[1] < out.shape[-1]:  # (B, H, W, C)
                    backbone_out_dim = out.shape[-1]
                    self._backbone_format = "BHWC"
                else:  # (B, C, H, W)
                    backbone_out_dim = out.shape[1]
                    self._backbone_format = "BCHW"
            else:
                backbone_out_dim = out.shape[-1]
                self._backbone_format = "flat"

        self.vision_proj = nn.Linear(backbone_out_dim, vision_embed_dim)

        # Question encoder: token embedding + positional encoding
        self.token_embed = nn.Embedding(vocab_size, vision_embed_dim)
        self.pos_enc = PositionalEncoding(vision_embed_dim, max_question_len, dropout)

        # Fusion decoder
        self.decoder = StudentDecoder(
            d_model=vision_embed_dim,
            nhead=decoder_heads,
            ffn_dim=decoder_ffn_dim,
            num_layers=decoder_layers,
            dropout=dropout,
        )

        # VQA classification head
        self.vqa_head = nn.Linear(vision_embed_dim, num_answers)

        # Projection heads to align student features → teacher feature dim for distillation
        self.feat_proj = nn.ModuleList([
            nn.Linear(vision_embed_dim, teacher_feat_dim)
            for _ in range(decoder_layers)
        ])

    def encode_image(self, pixel_values: torch.Tensor) -> torch.Tensor:
        # Florence-2 processor outputs 384×384; MobileViTv2 was trained on 256×256.
        # Resize to avoid massive intermediate feature maps on large batches.
        if pixel_values.shape[-1] != 256 or pixel_values.shape[-2] != 256:
            pixel_values = F.interpolate(
                pixel_values, size=(256, 256), mode="bilinear", align_corners=False
            )
        feats = self.vision_encoder(pixel_values)
        if self._backbone_format == "BCHW":
            B, C, H, W = feats.shape
            feats = feats.permute(0, 2, 3, 1).reshape(B, H * W, C)
        elif self._backbone_format == "BHWC":
            B, H, W, C = feats.shape
            feats = feats.reshape(B, H * W, C)
        # else already (B, S, C)
        return self.vision_proj(feats)  # (B, S, d_model)

    def forward(
        self,
        pixel_values: torch.Tensor,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, List[torch.Tensor], List[torch.Tensor]]:
        # image features as memory
        img_feats = self.encode_image(pixel_values)   # (B, S_img, d_model)

        # question token embeddings as decoder input
        q_emb = self.pos_enc(self.token_embed(input_ids))  # (B, T, d_model)

        # padding mask: True where padding (TransformerDecoder convention)
        tgt_pad_mask = None
        if attention_mask is not None:
            tgt_pad_mask = ~attention_mask.bool()

        # decode: fusion of question tokens over image memory
        out, layer_feats = self.decoder(q_emb, img_feats, tgt_pad_mask)

        # pool over question tokens (mean of non-padded positions) for classification
        if attention_mask is not None:
            mask = attention_mask.unsqueeze(-1).to(out.dtype)
            pooled = (out * mask).sum(1) / mask.sum(1).clamp(min=1)
        else:
            pooled = out.mean(1)

        logits = self.vqa_head(pooled)  # (B, num_answers)

        # project intermediate decoder features for distillation alignment
        proj_feats = [self.feat_proj[i](f) for i, f in enumerate(layer_feats)]

        # attention weights not directly accessible from nn.TransformerDecoderLayer
        # without patching; return empty list — feature matching is the primary signal
        return logits, proj_feats, []

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


def build_student(cfg: dict, num_answers: int) -> StudentVLM:
    sc = cfg["student"]
    return StudentVLM(
        vision_backbone=sc["vision_backbone"],
        vision_embed_dim=sc["vision_embed_dim"],
        decoder_layers=sc["decoder_layers"],
        decoder_heads=sc["decoder_heads"],
        decoder_ffn_dim=sc["decoder_ffn_dim"],
        dropout=sc["dropout"],
        num_answers=num_answers,
        teacher_feat_dim=sc["teacher_feat_dim"],
    )
