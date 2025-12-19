# To be done



#!/usr/bin/env python3
"""
infer.py — StreetCLIP (geolocal/StreetCLIP) multi-view inference script

What it does:
- Loads a trained StreetCLIPMultiView checkpoint (best.pt / final.pt / epoch_XXX.pt)
- Reads a *test* CSV (same columns as train, but without labels)
- Runs inference on 4 views (N/E/S/W)
- Writes predictions to a submission-style CSV:
  sample_id,image_north,image_east,image_south,image_west,
  predicted_state_idx_1..5,predicted_latitude,predicted_longitude

Usage:
  python infer.py \
    --ckpt /vast/bl3912/outputs_streetclip_freeze/checkpoints/best.pt \
    --test_csv /vast/bl3912/dataset/extracted_files/kaggle_dataset/test.csv \
    --images_dir /vast/bl3912/dataset/extracted_files/kaggle_dataset/test_images \
    --out_csv /vast/bl3912/outputs_streetclip_freeze/preds_test.csv \
    --batch_size 64 --num_workers 8
"""

import os
import json
import argparse
from typing import Any, Dict

import numpy as np
import pandas as pd
from PIL import Image
from tqdm import tqdm

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from transformers import CLIPProcessor
from model import StreetCLIPMultiView
from dataset import GeoGuessrTestDirDataset, geoguessr_test_collate_fn

# ----------------------------
# Checkpoint loading
# ----------------------------
def _get_base_model(model: nn.Module) -> nn.Module:
    return model.module if isinstance(model, nn.DataParallel) else model

def load_model_from_ckpt(
    ckpt_path: str,
    device: torch.device,
    override_cfg: Dict[str, Any] | None = None,
) -> nn.Module:
    ckpt = torch.load(ckpt_path, map_location="cpu")
    cfg = ckpt.get("config", {}) if isinstance(ckpt, dict) else {}

    if override_cfg:
        cfg = {**cfg, **override_cfg}

    # Build model with the same hyperparams used in training.
    # If your StreetCLIPMultiView signature differs, update these args accordingly.
    model = StreetCLIPMultiView(
        num_states=cfg.get("num_states", 50),
        clip_name=cfg.get("clip_name", "geolocal/StreetCLIP"),
        fusion=cfg.get("fusion", "transformer"),
        fusion_layers=cfg.get("fusion_layers", 2),
        fusion_heads=cfg.get("fusion_heads", 8),
        head_hidden=cfg.get("head_hidden", 512),
        dropout=cfg.get("dropout", 0.1),
        freeze_clip=False,  # inference: doesn't matter; keep trainable False is fine too
    )

    state_dict = ckpt["model"] if isinstance(ckpt, dict) and "model" in ckpt else ckpt
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    if missing:
        print(f"[WARN] Missing keys when loading ckpt: {missing[:10]}{'...' if len(missing) > 10 else ''}")
    if unexpected:
        print(f"[WARN] Unexpected keys when loading ckpt: {unexpected[:10]}{'...' if len(unexpected) > 10 else ''}")

    model = model.to(device)
    model.eval()
    return model


# ----------------------------
# Inference
# ----------------------------
@torch.no_grad()
def run_inference(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    amp: bool = True,
) -> pd.DataFrame:
    rows = []

    use_amp = amp and device.type == "cuda"
    autocast_ctx = torch.cuda.amp.autocast if use_amp else torch.cpu.amp.autocast  # safe fallback

    for batch in tqdm(loader):
        pv_n = batch["pixel_values_n"].to(device, non_blocking=True)
        pv_e = batch["pixel_values_e"].to(device, non_blocking=True)
        pv_s = batch["pixel_values_s"].to(device, non_blocking=True)
        pv_w = batch["pixel_values_w"].to(device, non_blocking=True)

        with (torch.cuda.amp.autocast() if use_amp else torch.no_grad()):
            out = model(pv_n, pv_e, pv_s, pv_w, return_logits=False)

        state_probs = out["state_probs"]          # [B,50] (softmax)
        gps_pred = out["gps_pred"]                # [B,2] (lat, lon)

        top5 = torch.topk(state_probs, k=5, dim=1).indices  # [B,5]

        sample_ids = batch["sample_id"].cpu().numpy().tolist()
        img_n_list = batch["image_north"]
        img_e_list = batch["image_east"]
        img_s_list = batch["image_south"]
        img_w_list = batch["image_west"]

        top5_np = top5.cpu().numpy()
        gps_np = gps_pred.cpu().numpy()

        for i in range(len(sample_ids)):
            rows.append({
                "sample_id": int(sample_ids[i]),
                "image_north": img_n_list[i],
                "image_east": img_e_list[i],
                "image_south": img_s_list[i],
                "image_west": img_w_list[i],
                "predicted_state_idx_1": int(top5_np[i, 0]),
                "predicted_state_idx_2": int(top5_np[i, 1]),
                "predicted_state_idx_3": int(top5_np[i, 2]),
                "predicted_state_idx_4": int(top5_np[i, 3]),
                "predicted_state_idx_5": int(top5_np[i, 4]),
                "predicted_latitude": float(gps_np[i, 0]),
                "predicted_longitude": float(gps_np[i, 1]),
            })

    df_out = pd.DataFrame(rows)

    # Keep deterministic ordering (often expected by graders)
    df_out = df_out.sort_values("sample_id").reset_index(drop=True)
    return df_out


# ----------------------------
# CLI
# ----------------------------
def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True, help="Path to checkpoint .pt (best.pt/final.pt/epoch_XXX.pt)")
    ap.add_argument("--images_dir", required=True, help="Directory containing test images")
    ap.add_argument("--out_csv", required=True, help="Where to save predictions CSV")

    ap.add_argument("--batch_size", type=int, default=4)
    ap.add_argument("--num_workers", type=int, default=4)
    ap.add_argument("--device", type=str, default="cuda")
    ap.add_argument("--amp", action="store_true", help="Use AMP on CUDA (faster, less memory)")

    # If you want to override processor/model name at inference time:
    ap.add_argument("--clip_name", type=str, default=None, help="Override clip_name (e.g., geolocal/StreetCLIP)")

    # DataParallel at inference (optional)
    ap.add_argument("--dataparallel", action="store_true", help="Wrap model with nn.DataParallel if multiple GPUs visible")

    return ap.parse_args()


def main():
    args = parse_args()

    device = torch.device(args.device if torch.cuda.is_available() or args.device == "cpu" else "cpu")
    print(f"Device: {device}")

    # Load checkpoint config (for clip_name, etc.)
    ckpt = torch.load(args.ckpt, map_location="cpu")
    cfg = ckpt.get("config", {}) if isinstance(ckpt, dict) else {}
    clip_name = args.clip_name or cfg.get("clip_name", "geolocal/StreetCLIP")

    # Processor
    processor = CLIPProcessor.from_pretrained(clip_name)

    # NEW:
    ds = GeoGuessrTestDirDataset(
        images_dir=args.images_dir,
        processor=processor,
        strict_files=True,
    )

    loader = DataLoader(
        ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=(device.type == "cuda"),
        collate_fn=geoguessr_test_collate_fn,
        drop_last=False,
    )

    # Model
    override = {"clip_name": clip_name} if args.clip_name else None
    model = load_model_from_ckpt(args.ckpt, device=device, override_cfg=override)

    if args.dataparallel and device.type == "cuda" and torch.cuda.device_count() > 1:
        print(f"Using DataParallel on {torch.cuda.device_count()} GPUs")
        model = nn.DataParallel(model).to(device)
        model.eval()

    # Inference
    df_out = run_inference(model, loader, device=device, amp=args.amp)

    # Save
    os.makedirs(os.path.dirname(args.out_csv) or ".", exist_ok=True)
    df_out.to_csv(args.out_csv, index=False)
    print(f"Saved: {args.out_csv}  (rows={len(df_out)})")
    print(df_out.head(3).to_string(index=False))


if __name__ == "__main__":
    main()
