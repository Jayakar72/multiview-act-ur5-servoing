#!/usr/bin/env python3
"""
train_obb.py — Train YOLOv8-OBB on the SFP/SC port dataset.

Reads dataset from: ~/aic_yolo_dataset/data.yaml
Saves runs to:      ~/aic_yolo_runs/sfp_obb_v1/
Best weights:       ~/aic_yolo_runs/sfp_obb_v1/weights/best.pt

Default hyperparameters are conservative — should give a solid model on
~5000 samples in 1-2 hours on an RTX 4080+ class GPU.

Usage:
    python train_obb.py                   # defaults: yolov8s-obb, 100 epochs, imgsz 640
    python train_obb.py --model n         # use yolov8n-obb (faster)
    python train_obb.py --model m         # use yolov8m-obb (more accurate)
    python train_obb.py --epochs 200      # train longer
    python train_obb.py --imgsz 1024      # higher resolution
    python train_obb.py --batch 8         # smaller batch (less VRAM)
    python train_obb.py --resume          # resume from last checkpoint
"""

import argparse
import os
from pathlib import Path

DATASET_ROOT_DEFAULT = Path.home() / "aic_yolo_dataset"
RUNS_ROOT_DEFAULT = Path.home() / "aic_yolo_runs"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=Path,
                    default=DATASET_ROOT_DEFAULT / "data.yaml")
    ap.add_argument("--runs-root", type=Path, default=RUNS_ROOT_DEFAULT)
    ap.add_argument("--name", default="sfp_obb_v1")
    ap.add_argument("--model", choices=["n", "s", "m", "l", "x"], default="s",
                    help="YOLOv8-OBB size (default 's' = small)")
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--patience", type=int, default=25,
                    help="early stopping patience (epochs without improvement)")
    ap.add_argument("--device", default="0", help="GPU id, or 'cpu'")
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--workers", type=int, default=8)
    args = ap.parse_args()

    # Lazy import — keeps pip install error visible only when actually training
    try:
        from ultralytics import YOLO
    except ImportError:
        print("\nUltralytics not found. Install with:")
        print("    pixi run pip install ultralytics")
        print("    # or:  pip install ultralytics")
        return

    if not args.data.is_file():
        print(f"data.yaml not found: {args.data}")
        print("Run prepare_obb_dataset.py first.")
        return

    weights = f"yolov8{args.model}-obb.pt"
    print(f"Starting from pretrained weights: {weights}")
    model = YOLO(weights)

    print(f"\nTraining config:")
    print(f"  data    = {args.data}")
    print(f"  model   = yolov8{args.model}-obb")
    print(f"  epochs  = {args.epochs}")
    print(f"  imgsz   = {args.imgsz}")
    print(f"  batch   = {args.batch}")
    print(f"  device  = {args.device}")
    print(f"  workers = {args.workers}")
    print(f"  output  = {args.runs_root}/{args.name}/")
    print()

    results = model.train(
        data=str(args.data),
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        patience=args.patience,
        device=args.device,
        project=str(args.runs_root),
        name=args.name,
        workers=args.workers,
        resume=args.resume,
        # Augmentation
        degrees=15,           # ±15° rotation
        translate=0.1,        # ±10% translation
        scale=0.5,            # ±50% scale
        shear=0.0,            # no shear (would distort slot rectangles)
        perspective=0.0,      # no perspective warp (don't fake-distort slots)
        flipud=0.0,           # no vertical flip (asymmetric scene)
        fliplr=0.0,           # no horizontal flip (slot is asymmetric in OBB)
        mosaic=0.5,           # mosaic mixing (50% of the time)
        mixup=0.0,
        # Loss weights — defaults are fine
        # Save settings
        save_period=10,       # checkpoint every 10 epochs
        verbose=True,
    )

    print("\n" + "=" * 60)
    print("TRAINING COMPLETE")
    print("=" * 60)
    save_dir = Path(results.save_dir)
    print(f"  Run dir       : {save_dir}")
    print(f"  Best weights  : {save_dir / 'weights' / 'best.pt'}")
    print(f"  Last weights  : {save_dir / 'weights' / 'last.pt'}")
    print(f"  Validation    : see {save_dir / 'results.png'}")
    print(f"  Confusion mat : see {save_dir / 'confusion_matrix.png'}")
    print()
    print("To validate manually:")
    print(f"  yolo obb val model={save_dir / 'weights' / 'best.pt'} "
          f"data={args.data}")


if __name__ == "__main__":
    main()
