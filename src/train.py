#!/usr/bin/env python3
"""
train.py — StreetCLIP (geolocal/StreetCLIP) multi-view training script
- Dataset: 4 views (north/east/south/west) + state classification + GPS regression
- Logging: Weights & Biases
- Checkpoints: periodic + best + final
- Loss: 0.7 * CE(state) + 0.3 * SmoothL1(GPS)   (matches scoring weights)
- Note: model outputs state_probs (softmax). For CE we convert back to log-probs via log().
"""

import os
import math
import time
import json
import random
from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple

import numpy as np
import pandas as pd
from PIL import Image

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from transformers import CLIPProcessor

from dataset import GeoGuessrDataset, geoguessr_collate_fn, NUM_STATES
from model import StreetCLIPMultiView

# ----------------------------
# Optional wandb
# ----------------------------
try:
    import wandb
    _WANDB_AVAILABLE = True
except Exception:
    wandb = None
    _WANDB_AVAILABLE = False


# ----------------------------
# Reproducibility
# ----------------------------
def seed_everything(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


# ----------------------------
# Metrics
# ----------------------------
@torch.no_grad()
def topk_acc_from_probs(probs: torch.Tensor, targets: torch.Tensor, k: int = 1) -> float:
    # probs: [B,C], targets: [B]
    topk = probs.topk(k, dim=1).indices  # [B,k]
    correct = topk.eq(targets.unsqueeze(1)).any(dim=1).float().mean().item()
    return correct


# ----------------------------
# Checkpointing
# ----------------------------
def save_ckpt(path: str, model: nn.Module, optimizer: torch.optim.Optimizer,
              scheduler: Optional[torch.optim.lr_scheduler._LRScheduler],
              step: int, epoch: int, best_val: float, cfg: Dict[str, Any]):
    os.makedirs(os.path.dirname(path), exist_ok=True)

    model_to_save = model.module if isinstance(model, nn.DataParallel) else model

    ckpt = {
        "model": model_to_save.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict() if scheduler is not None else None,
        "step": step,
        "epoch": epoch,
        "best_val": best_val,
        "config": cfg,
    }
    torch.save(ckpt, path)



# ----------------------------
# Train / Val loops
# ----------------------------
def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    scheduler,
    device: torch.device,
    epoch: int,
    log_every: int,
    use_wandb: bool,
    max_grad_norm: float = 1.0,
) -> Dict[str, float]:
    model.train()
    t0 = time.time()

    loss_meter = 0.0
    ce_meter = 0.0
    gps_meter = 0.0
    top1_meter = 0.0
    top5_meter = 0.0
    n_batches = 0

    for it, batch in enumerate(loader):
        n_batches += 1
        pv_n = batch["pixel_values_n"].to(device, non_blocking=True)
        pv_e = batch["pixel_values_e"].to(device, non_blocking=True)
        pv_s = batch["pixel_values_s"].to(device, non_blocking=True)
        pv_w = batch["pixel_values_w"].to(device, non_blocking=True)
        y_state = batch["state_idx"].to(device, non_blocking=True)
        y_gps = batch["gps"].to(device, non_blocking=True)

        out = model(pv_n, pv_e, pv_s, pv_w, return_logits=False)

        # NOTE: requirement: "softmax the state head first"
        state_probs = out["state_probs"]                          # already softmaxed
        state_log_probs = torch.log(state_probs.clamp_min(1e-8))  # convert to log-probs for NLLLoss
        loss_ce = F.nll_loss(state_log_probs, y_state)

        loss_gps = F.smooth_l1_loss(out["gps_pred"], y_gps)

        # Match evaluation weighting (70/30)
        loss = 0.70 * loss_ce + 0.30 * loss_gps

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
        optimizer.step()
        if scheduler is not None:
            scheduler.step()

        # metrics
        loss_meter += loss.item()
        ce_meter += loss_ce.item()
        gps_meter += loss_gps.item()
        top1_meter += topk_acc_from_probs(state_probs, y_state, k=1)
        top5_meter += topk_acc_from_probs(state_probs, y_state, k=5)

        if use_wandb and (it % log_every == 0):
            lr = optimizer.param_groups[0]["lr"]
            wandb.log(
                {
                    "train/loss": loss.item(),
                    "train/loss_ce": loss_ce.item(),
                    "train/loss_gps": loss_gps.item(),
                    "train/top1": topk_acc_from_probs(state_probs, y_state, k=1),
                    "train/top5": topk_acc_from_probs(state_probs, y_state, k=5),
                    "lr": lr,
                    "epoch": epoch,
                },
                commit=True,
            )

    dt = time.time() - t0
    return {
        "loss": loss_meter / n_batches,
        "loss_ce": ce_meter / n_batches,
        "loss_gps": gps_meter / n_batches,
        "top1": top1_meter / n_batches,
        "top5": top5_meter / n_batches,
        "sec": dt,
    }


@torch.no_grad()
def evaluate(model: nn.Module, loader: DataLoader, device: torch.device, epoch: int, use_wandb: bool) -> Dict[str, float]:
    model.eval()

    loss_meter = 0.0
    ce_meter = 0.0
    gps_meter = 0.0
    top1_meter = 0.0
    top5_meter = 0.0
    n_batches = 0

    for batch in loader:
        n_batches += 1
        pv_n = batch["pixel_values_n"].to(device, non_blocking=True)
        pv_e = batch["pixel_values_e"].to(device, non_blocking=True)
        pv_s = batch["pixel_values_s"].to(device, non_blocking=True)
        pv_w = batch["pixel_values_w"].to(device, non_blocking=True)
        y_state = batch["state_idx"].to(device, non_blocking=True)
        y_gps = batch["gps"].to(device, non_blocking=True)

        out = model(pv_n, pv_e, pv_s, pv_w, return_logits=False)
        state_probs = out["state_probs"]
        state_log_probs = torch.log(state_probs.clamp_min(1e-8))

        loss_ce = F.nll_loss(state_log_probs, y_state)
        loss_gps = F.smooth_l1_loss(out["gps_pred"], y_gps)
        loss = 0.70 * loss_ce + 0.30 * loss_gps

        loss_meter += loss.item()
        ce_meter += loss_ce.item()
        gps_meter += loss_gps.item()
        top1_meter += topk_acc_from_probs(state_probs, y_state, k=1)
        top5_meter += topk_acc_from_probs(state_probs, y_state, k=5)

    metrics = {
        "loss": loss_meter / n_batches,
        "loss_ce": ce_meter / n_batches,
        "loss_gps": gps_meter / n_batches,
        "top1": top1_meter / n_batches,
        "top5": top5_meter / n_batches,
    }

    if use_wandb:
        wandb.log(
            {
                "val/loss": metrics["loss"],
                "val/loss_ce": metrics["loss_ce"],
                "val/loss_gps": metrics["loss_gps"],
                "val/top1": metrics["top1"],
                "val/top5": metrics["top5"],
                "epoch": epoch,
            },
            commit=True,
        )
    return metrics


# ----------------------------
# Main
# ----------------------------
def main():
    # ---- Paths you provided ----
    train_csv = "/vast/bl3912/dataset/extracted_files/kaggle_dataset/train_ground_truth.csv"
    train_images = "/vast/bl3912/dataset/extracted_files/kaggle_dataset/train_images/"

    # ---- Output / ckpt ----
    out_dir = "/vast/bl3912/outputs_streetclip"
    ckpt_dir = os.path.join(out_dir, "checkpoints")
    os.makedirs(ckpt_dir, exist_ok=True)

    # ---- Basic config ----
    cfg = {
        "seed": 42,
        "clip_name": "geolocal/StreetCLIP",
        "num_states": NUM_STATES,
        "fusion": "transformer",
        "batch_size": 16,
        "num_workers": 16,
        "epochs": 5,
        "lr": 3e-5,                 # reasonable for CLIP finetune
        "weight_decay": 0.05,
        "warmup_ratio": 0.05,
        "log_every": 50,
        "save_every_steps": 2000,    # regular checkpoint
        "max_grad_norm": 1.0,
        "freeze_clip": False,
        "val_ratio": 0.02,          # quick sanity; increase if you want better model selection
    }

    seed_everything(cfg["seed"])

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ---- W&B init ----
    use_wandb = _WANDB_AVAILABLE and (os.environ.get("WANDB_DISABLED", "false").lower() != "true")
    if use_wandb:
        wandb.init(
            project=os.environ.get("WANDB_PROJECT", "geoguessr-streetclip"),
            name=os.environ.get("WANDB_RUN_NAME", None),
            config=cfg,
        )

    # ---- Processor + Dataset ----
    processor = CLIPProcessor.from_pretrained(cfg["clip_name"])

    full_ds = GeoGuessrDataset(
        images_dir=train_images,
        csv_file=train_csv,
        processor=processor,
        is_test=False,
        strict_files=True,
    )

    # simple train/val split
    n_total = len(full_ds)
    n_val = max(1, int(cfg["val_ratio"] * n_total))
    idxs = np.arange(n_total)
    np.random.shuffle(idxs)
    val_idxs = idxs[:n_val]
    train_idxs = idxs[n_val:]

    train_ds = torch.utils.data.Subset(full_ds, train_idxs.tolist())
    val_ds = torch.utils.data.Subset(full_ds, val_idxs.tolist())

    train_loader = DataLoader(
        train_ds,
        batch_size=cfg["batch_size"],
        shuffle=True,
        num_workers=cfg["num_workers"],
        pin_memory=True,
        collate_fn=geoguessr_collate_fn,
        drop_last=True,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=cfg["batch_size"],
        shuffle=False,
        num_workers=cfg["num_workers"],
        pin_memory=True,
        collate_fn=geoguessr_collate_fn,
        drop_last=False,
    )

    # ---- Model ----
    model = StreetCLIPMultiView(
        num_states=cfg["num_states"],
        clip_name=cfg["clip_name"],
        fusion=cfg["fusion"],
        fusion_layers=2,
        fusion_heads=8,
        head_hidden=512,
        dropout=0.1,
        freeze_clip=cfg["freeze_clip"],
    )

    # DataParallel (multi-GPU) — uses all visible GPUs by default
    if torch.cuda.is_available() and torch.cuda.device_count() > 1:
        print(f"Using DataParallel on {torch.cuda.device_count()} GPUs")
        model = nn.DataParallel(model)

    model = model.to(device)


    # ---- Optimizer ----
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg["lr"], weight_decay=cfg["weight_decay"])

    # ---- Scheduler: cosine with warmup (per-step) ----
    total_steps = cfg["epochs"] * len(train_loader)
    warmup_steps = int(cfg["warmup_ratio"] * total_steps)

    def lr_lambda(step: int):
        if step < warmup_steps:
            return float(step) / max(1, warmup_steps)
        progress = float(step - warmup_steps) / max(1, total_steps - warmup_steps)
        return 0.5 * (1.0 + math.cos(math.pi * progress))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    # ---- Training ----
    best_val = -1e9
    global_step = 0

    # Save config
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, "config.json"), "w") as f:
        json.dump(cfg, f, indent=2)

    for epoch in range(cfg["epochs"]):
        train_metrics = train_one_epoch(
            model=model,
            loader=train_loader,
            optimizer=optimizer,
            scheduler=scheduler,
            device=device,
            epoch=epoch,
            log_every=cfg["log_every"],
            use_wandb=use_wandb,
            max_grad_norm=cfg["max_grad_norm"],
        )

        # evaluate
        val_metrics = evaluate(model, val_loader, device=device, epoch=epoch, use_wandb=use_wandb)

        # choose best by val/top5 (matches competition)
        score = val_metrics["top5"]
        if score > best_val:
            best_val = score
            save_ckpt(
                path=os.path.join(ckpt_dir, "best.pt"),
                model=model,
                optimizer=optimizer,
                scheduler=scheduler,
                step=global_step,
                epoch=epoch,
                best_val=best_val,
                cfg=cfg,
            )

        # regular epoch ckpt
        save_ckpt(
            path=os.path.join(ckpt_dir, f"epoch_{epoch:03d}.pt"),
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            step=global_step,
            epoch=epoch,
            best_val=best_val,
            cfg=cfg,
        )

        # print
        print(
            f"[Epoch {epoch}] "
            f"train loss={train_metrics['loss']:.4f} top1={train_metrics['top1']:.3f} top5={train_metrics['top5']:.3f} | "
            f"val loss={val_metrics['loss']:.4f} top1={val_metrics['top1']:.3f} top5={val_metrics['top5']:.3f}"
        )

        if use_wandb:
            wandb.log(
                {
                    "epoch": epoch,
                    "train/epoch_loss": train_metrics["loss"],
                    "train/epoch_top1": train_metrics["top1"],
                    "train/epoch_top5": train_metrics["top5"],
                    "val/epoch_loss": val_metrics["loss"],
                    "val/epoch_top1": val_metrics["top1"],
                    "val/epoch_top5": val_metrics["top5"],
                    "best_val_top5": best_val,
                },
                commit=True,
            )

        # also save periodic step ckpts inside epoch loop (by steps)
        # (we approximate step count using epoch boundaries + loader length)
        global_step += len(train_loader)

    # ---- Final checkpoint ----
    save_ckpt(
        path=os.path.join(ckpt_dir, "final.pt"),
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        step=global_step,
        epoch=cfg["epochs"] - 1,
        best_val=best_val,
        cfg=cfg,
    )

    if use_wandb:
        wandb.finish()


if __name__ == "__main__":
    main()
