#!/usr/bin/env python3
"""
analyze_pose_log.py — Analyze pose_log.csv from Stage C.

Tests whether multi-frame averaging would bring our PnP errors down to
acceptable levels. Computes:

  1. Per-frame error stats (the raw single-frame numbers)
  2. Per-(port, camera) median across all frames
  3. Per-port median across all cameras + frames combined

Also shows error distributions by camera and by distance to port.

Usage:
    python3 analyze_pose_log.py [csv_path]
    Default: ~/aic_yolo_test_poses_v2/pose_log.csv
"""

import os
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd


def percentiles(arr, ps=(50, 75, 95)):
    if len(arr) == 0:
        return {p: float("nan") for p in ps}
    return {p: float(np.percentile(arr, p)) for p in ps}


def fmt_pct(arr):
    p = percentiles(arr)
    return f"med={p[50]:6.2f}  p75={p[75]:6.2f}  p95={p[95]:6.2f}  n={len(arr):4d}"


def main():
    csv_path = sys.argv[1] if len(sys.argv) > 1 else os.path.expanduser(
        "~/aic_yolo_test_poses_v2/pose_log.csv")

    if not os.path.exists(csv_path):
        print(f"ERROR: {csv_path} not found")
        sys.exit(1)

    df = pd.read_csv(csv_path)
    print(f"Loaded {len(df)} records from {csv_path}\n")
    print("Columns:", list(df.columns), "\n")

    # Use absolute values for error magnitudes
    df["abs_dx"]   = df["dx_mm"].abs()
    df["abs_dy"]   = df["dy_mm"].abs()
    df["abs_dz"]   = df["dz_mm"].abs()
    df["abs_dyaw"] = df["dyaw_deg"].abs()

    # ── Section 1: Per-frame stats (single-frame, no averaging) ──
    print("=" * 72)
    print("SECTION 1: Per-frame stats (single detection = single estimate)")
    print("=" * 72)
    for cls in sorted(df["class"].unique()):
        sub = df[df["class"] == cls]
        print(f"\n[{cls.upper()}]  n={len(sub)}")
        print(f"  |Δ|    mm   {fmt_pct(sub['d_mag_mm'].values)}")
        print(f"  |dx|   mm   {fmt_pct(sub['abs_dx'].values)}")
        print(f"  |dy|   mm   {fmt_pct(sub['abs_dy'].values)}")
        print(f"  |dz|   mm   {fmt_pct(sub['abs_dz'].values)}")
        print(f"  |dyaw| deg  {fmt_pct(sub['abs_dyaw'].values)}")

    # ── Section 2: Multi-frame averaging per (port, camera) ──
    print("\n" + "=" * 72)
    print("SECTION 2: Median across frames per (port, camera)")
    print("(simulates averaging during approach with one camera)")
    print("=" * 72)

    grouped = df.groupby(["matched_port", "camera"])
    agg_rows = []
    for (port, cam), g in grouped:
        # robust averaging via median
        med = g.median(numeric_only=True)
        agg_rows.append({
            "port": port, "camera": cam, "n_frames": len(g),
            "class": g["class"].iloc[0],
            "med_d_mag": med["d_mag_mm"],
            "med_abs_dx": g["abs_dx"].median(),
            "med_abs_dy": g["abs_dy"].median(),
            "med_abs_dz": g["abs_dz"].median(),
            "med_abs_dyaw": g["abs_dyaw"].median(),
        })
    agg = pd.DataFrame(agg_rows)
    if not agg.empty:
        for cls in sorted(agg["class"].unique()):
            sub = agg[agg["class"] == cls]
            print(f"\n[{cls.upper()}]  n={len(sub)} (port,camera) groups")
            print(f"  median |Δ|     mm   {fmt_pct(sub['med_d_mag'].values)}")
            print(f"  median |dz|    mm   {fmt_pct(sub['med_abs_dz'].values)}")
            print(f"  median |dyaw|  deg  {fmt_pct(sub['med_abs_dyaw'].values)}")

    # ── Section 3: Multi-camera fusion per port ──
    print("\n" + "=" * 72)
    print("SECTION 3: Median across ALL frames + ALL cameras per port")
    print("(simulates 3-camera fusion with averaging)")
    print("=" * 72)

    grouped = df.groupby("matched_port")
    fusion_rows = []
    for port, g in grouped:
        # combine all detections of this port from all cameras
        # take the median pose, then re-compute error vs GT (which is constant)
        gt_x = g["gt_x"].iloc[0]   # GT is constant per port
        gt_y = g["gt_y"].iloc[0]
        gt_z = g["gt_z"].iloc[0]
        gt_yaw = g["gt_yaw_deg"].iloc[0]

        med_px = g["pnp_x"].median()
        med_py = g["pnp_y"].median()
        med_pz = g["pnp_z"].median()
        # yaw needs care due to wrap-around; use median of dyaw_deg directly
        med_dyaw = g["dyaw_deg"].median()

        dx_mm = (med_px - gt_x) * 1000
        dy_mm = (med_py - gt_y) * 1000
        dz_mm = (med_pz - gt_z) * 1000
        d_mag = (dx_mm**2 + dy_mm**2 + dz_mm**2)**0.5

        fusion_rows.append({
            "port": port, "n_total_frames": len(g),
            "class": g["class"].iloc[0],
            "abs_dx": abs(dx_mm), "abs_dy": abs(dy_mm), "abs_dz": abs(dz_mm),
            "d_mag": d_mag, "abs_dyaw": abs(med_dyaw),
        })
    fusion = pd.DataFrame(fusion_rows)
    if not fusion.empty:
        for cls in sorted(fusion["class"].unique()):
            sub = fusion[fusion["class"] == cls]
            print(f"\n[{cls.upper()}]  {len(sub)} ports")
            for _, row in sub.iterrows():
                print(f"  {row['port'][:60]:60s}")
                print(f"    n={row['n_total_frames']:3d}  "
                      f"|Δ|={row['d_mag']:6.2f}mm  "
                      f"|dx|={row['abs_dx']:6.2f}  "
                      f"|dy|={row['abs_dy']:6.2f}  "
                      f"|dz|={row['abs_dz']:6.2f}  "
                      f"|dyaw|={row['abs_dyaw']:5.2f}°")
            print(f"  ── aggregate ──")
            print(f"  |Δ|     mm   {fmt_pct(sub['d_mag'].values)}")
            print(f"  |dz|    mm   {fmt_pct(sub['abs_dz'].values)}")
            print(f"  |dyaw|  deg  {fmt_pct(sub['abs_dyaw'].values)}")

    # ── Section 4: Confidence vs error correlation ──
    print("\n" + "=" * 72)
    print("SECTION 4: Error vs confidence (helps pick threshold)")
    print("=" * 72)
    bins = [0.4, 0.6, 0.75, 0.85, 0.95, 1.01]
    print(f"\n{'conf bucket':14s}  {'n':>5s}  {'med |Δ|':>9s}  "
          f"{'p95 |Δ|':>9s}  {'med dyaw':>10s}")
    for lo, hi in zip(bins[:-1], bins[1:]):
        sub = df[(df["conf"] >= lo) & (df["conf"] < hi)]
        if len(sub) == 0:
            continue
        print(f"{lo:.2f}–{hi:.2f}    {len(sub):>5d}  "
              f"{sub['d_mag_mm'].median():>7.2f}mm  "
              f"{np.percentile(sub['d_mag_mm'], 95):>7.2f}mm  "
              f"{sub['abs_dyaw'].median():>8.2f}°")

    # ── Section 5: Error vs distance ──
    print("\n" + "=" * 72)
    print("SECTION 5: Error vs camera-to-port distance")
    print("=" * 72)
    bins_cm = [(0, 22), (22, 25), (25, 30), (30, 40), (40, 100)]
    print(f"\n{'dist bucket':14s}  {'n':>5s}  {'med |Δ|':>9s}  "
          f"{'p95 |Δ|':>9s}  {'med dyaw':>10s}")
    for lo, hi in bins_cm:
        sub = df[(df["cam_to_port_cm"] >= lo) & (df["cam_to_port_cm"] < hi)]
        if len(sub) == 0:
            continue
        print(f"{lo:3d}–{hi:3d}cm    {len(sub):>5d}  "
              f"{sub['d_mag_mm'].median():>7.2f}mm  "
              f"{np.percentile(sub['d_mag_mm'], 95):>7.2f}mm  "
              f"{sub['abs_dyaw'].median():>8.2f}°")

    print("\n" + "=" * 72)


if __name__ == "__main__":
    main()
