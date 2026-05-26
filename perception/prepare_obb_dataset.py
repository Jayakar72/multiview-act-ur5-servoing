#!/usr/bin/env python3
"""
prepare_obb_dataset.py — Merge all run_ids, split per-config 80/20,
                         write YOLO data.yaml.

Reads from:  ~/aic_yolo_data/<run_id>/{images,labels}/
Writes to:   ~/aic_yolo_dataset/{train,val}/{images,labels}/
             ~/aic_yolo_dataset/data.yaml

Per-config split keeps each config represented in both train and val,
which avoids the "model never saw config X" failure mode.

Usage:
    python prepare_obb_dataset.py
    python prepare_obb_dataset.py --val 0.15           # different val ratio
    python prepare_obb_dataset.py --runs default rail0 # only specific runs
    python prepare_obb_dataset.py --hard-link          # symlink instead of copy (saves disk)
"""

import argparse
import os
import random
import shutil
import sys
from pathlib import Path

DATA_ROOT_DEFAULT = Path.home() / "aic_yolo_data"
DATASET_ROOT_DEFAULT = Path.home() / "aic_yolo_dataset"


def collect_samples(data_root, runs_filter=None):
    samples = []  # list of (img_path, lbl_path, run_id)
    if not data_root.is_dir():
        sys.exit(f"data_root not found: {data_root}")

    for run_dir in sorted(data_root.iterdir()):
        if not run_dir.is_dir():
            continue
        run_id = run_dir.name
        if runs_filter is not None and run_id not in runs_filter:
            continue
        images_dir = run_dir / "images"
        labels_dir = run_dir / "labels"
        if not images_dir.is_dir() or not labels_dir.is_dir():
            print(f"  [skip] {run_id}: missing images/ or labels/")
            continue
        for img_path in sorted(images_dir.glob("*.png")):
            lbl_path = labels_dir / (img_path.stem + ".txt")
            if lbl_path.exists() and lbl_path.stat().st_size > 0:
                samples.append((img_path, lbl_path, run_id))
    return samples


def split_and_install(samples, dataset_root, val_ratio, mode="copy"):
    # Group by run_id so each config splits independently
    by_run = {}
    for img, lbl, run in samples:
        by_run.setdefault(run, []).append((img, lbl))

    rng = random.Random(42)
    train_items = []
    val_items = []
    print("\nPer-config split:")
    for run, items in sorted(by_run.items()):
        rng.shuffle(items)
        n_val = max(1, int(round(len(items) * val_ratio)))
        n_train = len(items) - n_val
        val_items.extend(items[:n_val])
        train_items.extend(items[n_val:])
        print(f"  {run:>10}: {n_train:>5} train + {n_val:>4} val "
              f"= {len(items)} total")

    print(f"\nTOTAL:  {len(train_items)} train + {len(val_items)} val "
          f"= {len(train_items) + len(val_items)}")

    # Wipe + recreate dataset dirs
    if dataset_root.exists():
        print(f"\nClearing existing dataset at {dataset_root}")
        shutil.rmtree(dataset_root)

    for split, items in [("train", train_items), ("val", val_items)]:
        img_out = dataset_root / split / "images"
        lbl_out = dataset_root / split / "labels"
        img_out.mkdir(parents=True, exist_ok=True)
        lbl_out.mkdir(parents=True, exist_ok=True)
        for img, lbl in items:
            dst_img = img_out / img.name
            dst_lbl = lbl_out / lbl.name
            if mode == "hard-link":
                try:
                    os.link(img, dst_img)
                    os.link(lbl, dst_lbl)
                except OSError:
                    shutil.copy2(img, dst_img)
                    shutil.copy2(lbl, dst_lbl)
            else:
                shutil.copy2(img, dst_img)
                shutil.copy2(lbl, dst_lbl)

    # Write data.yaml
    data_yaml = f"""# Auto-generated dataset config for YOLOv8-OBB SFP/SC port detector
path: {dataset_root}
train: train/images
val: val/images

# OBB classes
names:
  0: sfp_slot
  1: sc_slot
"""
    yaml_path = dataset_root / "data.yaml"
    yaml_path.write_text(data_yaml)
    print(f"\nWrote {yaml_path}")
    print(f"\nDataset ready at: {dataset_root}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", type=Path, default=DATA_ROOT_DEFAULT)
    ap.add_argument("--dataset-root", type=Path, default=DATASET_ROOT_DEFAULT)
    ap.add_argument("--val", type=float, default=0.20,
                    help="validation ratio per-config (default 0.20)")
    ap.add_argument("--runs", nargs="+", default=None,
                    help="restrict to specific run_ids (default: all)")
    ap.add_argument("--hard-link", action="store_true",
                    help="hard-link files instead of copying (saves disk)")
    args = ap.parse_args()

    print(f"Collecting from: {args.data_root}")
    samples = collect_samples(args.data_root, args.runs)
    print(f"Found {len(samples)} (image, label) pairs")

    if not samples:
        sys.exit("No samples found. Check that <data_root>/<run_id>/{images,labels}/ exists.")

    split_and_install(
        samples,
        args.dataset_root,
        val_ratio=args.val,
        mode="hard-link" if args.hard_link else "copy",
    )


if __name__ == "__main__":
    main()
