#!/usr/bin/env python3
"""
train_act.py — ATOMIC ACT Training Script
==========================================
Trains ACT (Action Chunking with Transformers) directly from our
HDF5 demo files. Bypasses LeRobot's dataset infrastructure entirely
to avoid HuggingFace dependency issues.

Uses LeRobot's ACTPolicy model weights and forward pass only.

USAGE:
  cd ~/ws_aic/src/aic
  pixi run python3 scripts/train_act.py

  # Or with custom settings:
  pixi run python3 scripts/train_act.py \
    --demo_dir ~/aic_demos \
    --output_dir ~/aic_act_checkpoints \
    --steps 50000 \
    --batch_size 8
"""

import argparse
import glob
import os
import time
import random
from pathlib import Path
from collections import defaultdict

import cv2
import h5py
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

# LeRobot ACT model
from lerobot.policies.act.modeling_act import ACTPolicy
from lerobot.policies.act.configuration_act import ACTConfig
from lerobot.configs.policies import FeatureType, PolicyFeature


# ── Config ────────────────────────────────────────────────────────────────────

IDLE_THRESHOLD_M = 0.0001   # 0.1mm — trim trailing idle steps
IMG_H = 360
IMG_W = 480
STATE_DIM = 16              # tcp(7) + wrench(6) + port_type(2) + port_idx(1)
ACTION_DIM = 7              # target tcp pose


# ── Dataset ───────────────────────────────────────────────────────────────────

class AICDemoDataset(Dataset):
    """
    Loads HDF5 demo episodes, trims idle steps, returns
    (observation_dict, action_chunk) pairs for ACT training.
    """

    def __init__(self, demo_dir: str, chunk_size: int = 50, img_size: tuple = (224, 224)):
        self.chunk_size = chunk_size
        self.img_h, self.img_w = img_size
        self.samples = []   # list of (h5_path, start_idx)

        h5_files = sorted(glob.glob(os.path.join(demo_dir, "episode_*.h5")))
        print(f"Loading {len(h5_files)} episodes from {demo_dir}")

        skipped = 0
        for h5_path in h5_files:
            with h5py.File(h5_path, "r") as f:
                tcp    = f["obs/tcp_pose"][:]
                acts   = f["actions/tcp_pose"][:]

            # Trim idle tail
            if len(tcp) >= 2:
                diffs = np.linalg.norm(np.diff(tcp[:, :3], axis=0), axis=1)
                real  = np.where(diffs >= IDLE_THRESHOLD_M)[0]
                cutoff = int(real[-1]) + 3 if len(real) > 0 else 10
                cutoff = min(cutoff, len(tcp))
            else:
                cutoff = len(tcp)

            # Only keep episodes long enough for at least one chunk
            if cutoff < chunk_size:
                skipped += 1
                continue

            # Each valid starting index becomes one training sample
            for start in range(0, cutoff - chunk_size + 1, 10):  # stride=10
                self.samples.append((h5_path, start, cutoff))

        print(f"  {len(h5_files) - skipped} episodes used, {skipped} skipped")
        print(f"  {len(self.samples)} training samples (chunk_size={chunk_size}, stride=10)")

    def __len__(self):
        return len(self.samples)

    def _load_image(self, img_raw: np.ndarray) -> torch.Tensor:
        """(H, W, 3) uint8 RGB → (3, H, W) float32 normalized tensor."""
        img = cv2.resize(img_raw, (self.img_w, self.img_h), interpolation=cv2.INTER_AREA)
        img = torch.from_numpy(img).permute(2, 0, 1).float() / 255.0
        # ImageNet normalization (LeRobot default)
        mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
        std  = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
        return (img - mean) / std

    def _parse_port(self, port_name: str):
        parts = port_name.lower().split("_")
        port_type  = parts[0]
        port_index = 0
        for p in reversed(parts):
            if p.isdigit():
                port_index = int(p)
                break
        return port_type, port_index

    def __getitem__(self, idx):
        h5_path, start, cutoff = self.samples[idx]

        with h5py.File(h5_path, "r") as f:
            t = start
            left   = f["obs/images/left"][t]
            center = f["obs/images/center"][t]
            right  = f["obs/images/right"][t]
            tcp    = f["obs/tcp_pose"][t]
            wrench = f["obs/wrench"][t]

            # Action chunk — pad with last action if episode ends early
            end   = min(start + self.chunk_size, cutoff)
            acts  = f["actions/tcp_pose"][start:end]
            port_name = str(f["metadata"].attrs.get("port_name", "sfp_port_0"))

        port_type, port_index = self._parse_port(port_name)

        # Pad action chunk if needed
        if len(acts) < self.chunk_size:
            pad = np.tile(acts[-1], (self.chunk_size - len(acts), 1))
            acts = np.concatenate([acts, pad], axis=0)

        # Build state vector (16D)
        state = np.concatenate([
            tcp,
            wrench,
            np.array([
                float(port_type == "sfp"),
                float(port_type == "sc"),
                float(port_index),
            ], dtype=np.float32),
        ])

        return {
            "observation.images.left":   self._load_image(left),
            "observation.images.center": self._load_image(center),
            "observation.images.right":  self._load_image(right),
            "observation.state":         torch.from_numpy(state).float(),
            "action":                    torch.from_numpy(acts).float(),
        }


# ── Normalization stats ───────────────────────────────────────────────────────

def compute_normalization_stats(dataset: AICDemoDataset, n_samples: int = 500):
    """Compute mean/std for state and action from a subset of the dataset."""
    print("Computing normalization statistics...")
    indices = random.sample(range(len(dataset)), min(n_samples, len(dataset)))

    states  = []
    actions = []
    for i in tqdm(indices, desc="  stats", leave=False):
        sample = dataset[i]
        states.append(sample["observation.state"].numpy())
        actions.append(sample["action"].numpy())

    states  = np.stack(states)
    actions = np.concatenate(actions, axis=0)

    stats = {
        "state_mean": states.mean(0),
        "state_std":  states.std(0).clip(min=1e-6),
        "action_mean": actions.mean(0),
        "action_std":  actions.std(0).clip(min=1e-6),
    }
    print(f"  state  mean: {stats['state_mean'][:3].round(4)} ...")
    print(f"  action mean: {stats['action_mean'].round(4)}")
    return stats


# ── Model setup ───────────────────────────────────────────────────────────────

def build_act_policy(chunk_size: int, device: torch.device) -> ACTPolicy:
    """Build ACT policy configured for our 3-camera, 16D state setup."""
    config = ACTConfig(
        # Architecture
        n_obs_steps        = 1,
        chunk_size         = chunk_size,
        n_action_steps     = chunk_size,
        dim_model          = 512,
        n_heads            = 8,
        dim_feedforward    = 3200,
        n_encoder_layers   = 4,
        n_decoder_layers   = 1,
        n_vae_encoder_layers = 4,
        use_vae            = True,
        kl_weight          = 10.0,
        vision_backbone    = "resnet18",
        pretrained_backbone_weights = "ResNet18_Weights.IMAGENET1K_V1",
        # Features — must match our dataset
        input_features = {
            "observation.images.left":   PolicyFeature(type=FeatureType.VISUAL, shape=(3, 224, 224)),
            "observation.images.center": PolicyFeature(type=FeatureType.VISUAL, shape=(3, 224, 224)),
            "observation.images.right":  PolicyFeature(type=FeatureType.VISUAL, shape=(3, 224, 224)),
            "observation.state":         PolicyFeature(type=FeatureType.STATE,  shape=(STATE_DIM,)),
        },
        output_features = {
            "action": PolicyFeature(type=FeatureType.ACTION, shape=(ACTION_DIM,)),
        },
    )

    policy = ACTPolicy(config)
    policy.to(device)
    return policy


# ── Training loop ─────────────────────────────────────────────────────────────

def train(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n{'='*60}")
    print(f"  ATOMIC ACT Training")
    print(f"{'='*60}")
    print(f"  Device     : {device}")
    if device.type == "cuda":
        print(f"  GPU        : {torch.cuda.get_device_name(0)}")
    print(f"  Demo dir   : {args.demo_dir}")
    print(f"  Output dir : {args.output_dir}")
    print(f"  Steps      : {args.steps}")
    print(f"  Batch size : {args.batch_size}")
    print(f"  Chunk size : {args.chunk_size}")
    print(f"{'='*60}\n")

    os.makedirs(args.output_dir, exist_ok=True)

    # Dataset
    dataset = AICDemoDataset(args.demo_dir, chunk_size=args.chunk_size)
    if len(dataset) == 0:
        print("ERROR: No training samples found. Check demo_dir.")
        return

    loader = DataLoader(
        dataset,
        batch_size   = args.batch_size,
        shuffle      = True,
        num_workers  = 2,
        pin_memory   = True,
        drop_last    = True,
    )

    # Normalization stats
    stats = compute_normalization_stats(dataset)
    state_mean  = torch.from_numpy(stats["state_mean"]).float().to(device)
    state_std   = torch.from_numpy(stats["state_std"]).float().to(device)
    action_mean = torch.from_numpy(stats["action_mean"]).float().to(device)
    action_std  = torch.from_numpy(stats["action_std"]).float().to(device)

    # Save stats for inference
    np.save(os.path.join(args.output_dir, "norm_stats.npy"), stats)
    print("✓ Normalization stats saved")

    # Model
    policy = build_act_policy(args.chunk_size, device)
    print(f"✓ ACT policy built")
    n_params = sum(p.numel() for p in policy.parameters() if p.requires_grad)
    print(f"  Trainable parameters: {n_params/1e6:.1f}M")

    # Optimizer
    optimizer = torch.optim.AdamW(
        policy.parameters(),
        lr           = 1e-5,
        weight_decay = 1e-4,
        betas        = (0.9, 0.999),
    )

    # Training
    print(f"\nStarting training for {args.steps} steps...\n")
    policy.train()

    step        = 0
    epoch       = 0
    loss_history = defaultdict(list)
    t_start     = time.time()

    while step < args.steps:
        epoch += 1
        for batch in loader:
            if step >= args.steps:
                break

            # Move to device
            obs = {
                "observation.images.left":   batch["observation.images.left"].to(device),
                "observation.images.center": batch["observation.images.center"].to(device),
                "observation.images.right":  batch["observation.images.right"].to(device),
                "observation.state":         (batch["observation.state"].to(device) - state_mean) / state_std,
            }
            actions_raw = batch["action"].to(device)

            # Normalize actions
            actions_norm = (actions_raw - action_mean) / action_std

            # Forward pass — ACTPolicy expects single batch dict with action_is_pad
            B, T, _ = actions_norm.shape
            action_is_pad = torch.zeros(B, T, dtype=torch.bool, device=device)
            batch = {**obs, "action": actions_norm, "action_is_pad": action_is_pad}
            optimizer.zero_grad()
            output = policy(batch)

            # Extract loss — ACTPolicy returns (loss, loss_dict) tuple
            if isinstance(output, tuple):
                loss = output[0]
            elif isinstance(output, dict):
                loss = output.get("loss", output.get("total_loss"))
            else:
                loss = output

            loss_val = loss.item()

            loss.backward()
            torch.nn.utils.clip_grad_norm_(policy.parameters(), 10.0)
            optimizer.step()

            loss_history["loss"].append(loss_val)
            step += 1

            # Log
            if step % args.log_freq == 0:
                avg_loss = np.mean(loss_history["loss"][-args.log_freq:])
                elapsed  = time.time() - t_start
                steps_per_sec = step / elapsed
                eta_min = (args.steps - step) / steps_per_sec / 60
                print(
                    f"step {step:6d}/{args.steps}  "
                    f"loss={avg_loss:.4f}  "
                    f"epoch={epoch}  "
                    f"eta={eta_min:.0f}min  "
                    f"({steps_per_sec:.1f} steps/s)"
                )

            # Checkpoint
            if step % args.save_freq == 0 or step == args.steps:
                ckpt_dir = os.path.join(args.output_dir, f"checkpoint-{step}")
                os.makedirs(ckpt_dir, exist_ok=True)
                torch.save(policy.state_dict(), os.path.join(ckpt_dir, "policy.pt"))
                torch.save(optimizer.state_dict(), os.path.join(ckpt_dir, "optimizer.pt"))
                np.save(os.path.join(ckpt_dir, "norm_stats.npy"), stats)
                print(f"  ✓ Checkpoint saved → {ckpt_dir}")

    # Save final
    final_dir = os.path.join(args.output_dir, "final")
    os.makedirs(final_dir, exist_ok=True)
    torch.save(policy.state_dict(), os.path.join(final_dir, "policy.pt"))
    np.save(os.path.join(final_dir, "norm_stats.npy"), stats)

    total_time = (time.time() - t_start) / 60
    print(f"\n{'='*60}")
    print(f"  ✓ Training complete!")
    print(f"  Total time  : {total_time:.0f} minutes")
    print(f"  Final loss  : {np.mean(loss_history['loss'][-100:]):.4f}")
    print(f"  Checkpoints : {args.output_dir}")
    print(f"{'='*60}\n")


# ── CLI ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train ACT policy on AIC demos")
    parser.add_argument("--demo_dir",   default=os.path.expanduser("~/aic_demos"))
    parser.add_argument("--output_dir", default=os.path.expanduser("~/aic_act_checkpoints"))
    parser.add_argument("--steps",      default=50000, type=int)
    parser.add_argument("--batch_size", default=8,     type=int)
    parser.add_argument("--chunk_size", default=50,    type=int)
    parser.add_argument("--log_freq",   default=50,    type=int)
    parser.add_argument("--save_freq",  default=5000,  type=int)
    args = parser.parse_args()
    train(args)
