"""
StreetCLIP multi-view model: CLIP image encoder + fusion + two heads
- Inputs: 4 directional images (N/E/S/W), each tensor [B, 3, H, W] (already normalized for CLIP)
- Outputs:
  - state_probs: [B, num_states]  (softmax probabilities)
  - gps_pred:    [B, 2]           (lat, lon in degrees by default)

Install:
  pip install torch torchvision transformers

Notes:
- Use CLIPProcessor to preprocess images (resize/crop/normalize) to `pixel_values`.
- This model uses ONLY the vision encoder from StreetCLIP; no text is needed for your task.
"""

from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import CLIPModel, CLIPProcessor


# ----------------------------
# Fusion blocks
# ----------------------------
class ViewTransformerFusion(nn.Module):
    """
    Fuses 4 view embeddings using a small Transformer encoder over view tokens.
    Input:  x [B, V, D]
    Output: fused [B, D]
    """
    def __init__(self, d_model: int, n_heads: int = 8, n_layers: int = 2, dropout: float = 0.1):
        super().__init__()
        enc_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=4 * d_model,
            dropout=dropout,
            batch_first=True,
            activation="gelu",
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=n_layers)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, d_model))
        self.pos_embed = nn.Parameter(torch.zeros(1, 1 + 4, d_model))  # cls + 4 views
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        nn.init.trunc_normal_(self.pos_embed, std=0.02)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, v, d = x.shape
        assert v == 4, f"Expected 4 views, got {v}"
        cls = self.cls_token.expand(b, -1, -1)          # [B, 1, D]
        tokens = torch.cat([cls, x], dim=1)             # [B, 5, D]
        tokens = tokens + self.pos_embed                # [B, 5, D]
        tokens = self.encoder(tokens)                   # [B, 5, D]
        fused = tokens[:, 0]                            # take CLS
        return fused


class MLP(nn.Module):
    def __init__(self, in_dim: int, hidden_dim: int, out_dim: int, dropout: float = 0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, out_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


# ----------------------------
# Main Model
# ----------------------------
class StreetCLIPMultiView(nn.Module):
    """
    CLIP vision encoder (StreetCLIP) + fusion head + multi-task heads.
    """
    def __init__(
        self,
        num_states: int,
        clip_name: str = "geolocal/StreetCLIP",
        fusion: str = "transformer",  # "transformer" or "mean"
        fusion_layers: int = 2,
        fusion_heads: int = 8,
        head_hidden: int = 512,
        dropout: float = 0.1,
        freeze_clip: bool = False,
    ):
        super().__init__()

        self.clip = CLIPModel.from_pretrained(clip_name)
        self.embed_dim = self.clip.config.projection_dim  # typically 512

        if freeze_clip:
            for p in self.clip.parameters():
                p.requires_grad = False

        if fusion == "transformer":
            self.fusion = ViewTransformerFusion(
                d_model=self.embed_dim,
                n_heads=fusion_heads,
                n_layers=fusion_layers,
                dropout=dropout,
            )
        elif fusion == "mean":
            self.fusion = None
        else:
            raise ValueError(f"Unknown fusion='{fusion}'. Use 'transformer' or 'mean'.")

        # Heads
        self.state_head = MLP(self.embed_dim, head_hidden, num_states, dropout=dropout)
        self.gps_head = MLP(self.embed_dim, head_hidden, 2, dropout=dropout)

        self._fusion_mode = fusion

    @torch.no_grad()
    def get_expected_pixel_size(self) -> int:
        # CLIP expects square inputs, usually 224 or 336 depending on model.
        return self.clip.config.vision_config.image_size

    def encode_view(self, pixel_values: torch.Tensor) -> torch.Tensor:
        """
        pixel_values: [B, 3, H, W] normalized for CLIP
        returns:      [B, D] projected image embedding
        """
        # CLIPModel has helper that returns projected embeddings:
        img_emb = self.clip.get_image_features(pixel_values=pixel_values)  # [B, D]
        # Optional: normalize (CLIP typically uses normalized features)
        img_emb = F.normalize(img_emb, dim=-1)
        return img_emb

    def forward(
        self,
        pixel_values_n: torch.Tensor,
        pixel_values_e: torch.Tensor,
        pixel_values_s: torch.Tensor,
        pixel_values_w: torch.Tensor,
        return_logits: bool = False,
    ) -> Dict[str, torch.Tensor]:
        """
        Inputs: 4 tensors [B, 3, H, W] each
        """
        fn = self.encode_view(pixel_values_n)
        fe = self.encode_view(pixel_values_e)
        fs = self.encode_view(pixel_values_s)
        fw = self.encode_view(pixel_values_w)

        views = torch.stack([fn, fe, fs, fw], dim=1)  # [B, 4, D]

        if self._fusion_mode == "mean":
            fused = views.mean(dim=1)                 # [B, D]
        else:
            fused = self.fusion(views)                # [B, D]

        state_logits = self.state_head(fused)         # [B, num_states]
        gps_pred = self.gps_head(fused)               # [B, 2]

        out = {
            "state_probs": F.softmax(state_logits, dim=-1),
            "gps_pred": gps_pred,
        }
        if return_logits:
            out["state_logits"] = state_logits
        return out


# ----------------------------
# Preprocessing utilities
# ----------------------------
@dataclass
class FourViewBatch:
    n: torch.Tensor  # [B, 3, H, W]
    e: torch.Tensor
    s: torch.Tensor
    w: torch.Tensor
    state: Optional[torch.Tensor] = None  # [B] int64
    gps: Optional[torch.Tensor] = None    # [B,2] float32 lat/lon


class StreetCLIPPreprocessor:
    """
    Wraps CLIPProcessor to produce pixel_values for 4 images per sample.
    """
    def __init__(self, clip_name: str = "geolocal/StreetCLIP"):
        self.processor = CLIPProcessor.from_pretrained(clip_name)

    def __call__(
        self,
        img_n,
        img_e,
        img_s,
        img_w,
        device: Optional[torch.device] = None,
    ) -> FourViewBatch:
        # processor returns dict with pixel_values [1, 3, H, W] for a single image
        pn = self.processor(images=img_n, return_tensors="pt")["pixel_values"]
        pe = self.processor(images=img_e, return_tensors="pt")["pixel_values"]
        ps = self.processor(images=img_s, return_tensors="pt")["pixel_values"]
        pw = self.processor(images=img_w, return_tensors="pt")["pixel_values"]

        if device is not None:
            pn, pe, ps, pw = pn.to(device), pe.to(device), ps.to(device), pw.to(device)

        # Remove the leading batch dim from processor (it’s 1); you can stack later in a DataLoader collate
        return FourViewBatch(n=pn[0], e=pe[0], s=ps[0], w=pw[0])


# ----------------------------
# Example usage (single sample)
# ----------------------------
if __name__ == "__main__":
    from PIL import Image

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Example: dummy 4 views (replace with your own N/E/S/W images)
    img = Image.new("RGB", (256, 256), color=(128, 128, 128))
    pre = StreetCLIPPreprocessor("geolocal/StreetCLIP")
    sample = pre(img, img, img, img, device=None)

    # Make a batch of size 1
    batch_n = sample.n.unsqueeze(0).to(device)
    batch_e = sample.e.unsqueeze(0).to(device)
    batch_s = sample.s.unsqueeze(0).to(device)
    batch_w = sample.w.unsqueeze(0).to(device)

    model = StreetCLIPMultiView(
        num_states=33,                 # <-- set to your number of states
        clip_name="geolocal/StreetCLIP",
        fusion="transformer",          # or "mean"
        fusion_layers=2,
        fusion_heads=8,
        head_hidden=512,
        dropout=0.1,
        freeze_clip=False,
    ).to(device)

    model.eval()
    with torch.no_grad():
        out = model(batch_n, batch_e, batch_s, batch_w, return_logits=False)

    print("state_probs:", out["state_probs"].shape)  # [1, num_states]
    print("gps_pred:", out["gps_pred"].shape)        # [1, 2]
