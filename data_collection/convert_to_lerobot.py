#!/usr/bin/env python3
"""
convert_to_lerobot.py
─────────────────────
Converts AtomicRecorder HDF5 episodes to LeRobot dataset format.
Trims idle trailing steps (TCP movement < 0.1mm) before converting.

Output layout:
  ~/aic_lerobot_dataset/
    data/chunk-000/episode_000000.parquet ...
    videos/chunk-000/observation.images.{left,center,right}/episode_000000.mp4
    meta/info.json  episodes.jsonl  stats.json

USAGE:
  python3 convert_to_lerobot.py
  python3 convert_to_lerobot.py --demo-dir ~/aic_demos --out-dir ~/aic_lerobot_dataset --fps 20

REQUIREMENTS:
  pixi run pip install pyarrow tqdm
  (h5py, opencv, numpy already installed)
"""

import argparse
import json
import os
import shutil
from pathlib import Path
from collections import Counter

import cv2
import h5py
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
from tqdm import tqdm


# ── Defaults ──────────────────────────────────────────────────────────────────

DEFAULT_DEMO_DIR = os.path.expanduser("~/aic_demos")
DEFAULT_OUT_DIR  = os.path.expanduser("~/aic_lerobot_dataset")
DEFAULT_FPS      = 20
CHUNK_SIZE       = 1000   # episodes per chunk folder

# Minimum TCP movement per step to be considered "real motion" (metres)
IDLE_THRESHOLD_M = 0.0001  # 0.1 mm


# ── Helpers ───────────────────────────────────────────────────────────────────

def _port_to_task_str(port_type: str, port_index: int) -> str:
    names = {"sfp": "SFP module", "sc": "SC fiber plug"}
    connector = names.get(port_type, port_type.upper())
    return f"insert {connector} into port {port_index}"


def _trim_idle_tail(tcp, wrench, acts, left, center, right):
    """
    Remove trailing idle steps where the TCP barely moves.
    Keeps everything up to and including the last step with
    real motion (> IDLE_THRESHOLD_M per step).
    Returns trimmed arrays and the number of steps removed.
    """
    if len(tcp) < 2:
        return tcp, wrench, acts, left, center, right, 0

    diffs = np.linalg.norm(np.diff(tcp[:, :3], axis=0), axis=1)
    real_indices = np.where(diffs >= IDLE_THRESHOLD_M)[0]

    if len(real_indices) == 0:
        # Entire episode is idle — keep first 10 steps as minimum
        cutoff = min(10, len(tcp))
    else:
        # Keep up to 2 steps past the last real motion step
        cutoff = min(real_indices[-1] + 3, len(tcp))

    trimmed = len(tcp) - cutoff
    return (
        tcp[:cutoff],
        wrench[:cutoff],
        acts[:cutoff],
        left[:cutoff],
        center[:cutoff],
        right[:cutoff],
        trimmed,
    )


def _write_mp4(frames: np.ndarray, path: Path, fps: int):
    """Write (N, H, W, 3) uint8 RGB array to MP4."""
    path.parent.mkdir(parents=True, exist_ok=True)
    n, h, w, _ = frames.shape
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(path), fourcc, fps, (w, h))
    for frame in frames:
        writer.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
    writer.release()


def _episode_to_parquet(
    ep_idx, obs_tcp, obs_wrench, actions,
    port_type, port_index, chunk_dir
):
    """Write one episode as a Parquet file."""
    n = len(actions)

    # State: TCP pose (7) + wrench (6) + port_type one-hot (2) + port_index (1) = 16
    state_rows = []
    for i in range(n):
        state = np.concatenate([
            obs_tcp[i],
            obs_wrench[i],
            np.array([
                float(port_type == "sfp"),
                float(port_type == "sc"),
                float(port_index),
            ], dtype=np.float32),
        ])
        state_rows.append(state.tolist())

    table = pa.table({
        "episode_index": np.full(n, ep_idx, dtype=np.int64),
        "frame_index":   np.arange(n, dtype=np.int64),
        "timestamp":     np.arange(n, dtype=np.float64) / DEFAULT_FPS,
        "task_index":    np.zeros(n, dtype=np.int64),
        "observation.state": pa.array(state_rows),
        "action":            pa.array([actions[i].tolist() for i in range(n)]),
        "next.done":         np.array([False] * (n - 1) + [True]),
        "next.reward":       np.zeros(n, dtype=np.float32),
    })

    out_path = chunk_dir / f"episode_{ep_idx:06d}.parquet"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(table, str(out_path))
    return out_path


def _parse_port_meta(port_name: str):
    parts = port_name.lower().split("_")
    port_type = parts[0]  # 'sfp' or 'sc'
    port_index = 0
    for part in reversed(parts):
        if part.isdigit():
            port_index = int(part)
            break
    return port_type, port_index


# ── Main converter ─────────────────────────────────────────────────────────────

def convert(demo_dir: str, out_dir: str, fps: int):
    demo_dir = Path(demo_dir)
    out_dir  = Path(out_dir)

    episodes = sorted(demo_dir.glob("episode_*.h5"))
    if not episodes:
        print(f"No episodes found in {demo_dir}. Run AtomicRecorder first.")
        return

    print(f"\n{'='*60}")
    print(f"  ATOMIC LeRobot Converter")
    print(f"{'='*60}")
    print(f"  Input  : {demo_dir}  ({len(episodes)} episodes)")
    print(f"  Output : {out_dir}")
    print(f"  FPS    : {fps}")
    print(f"  Trim   : idle steps < {IDLE_THRESHOLD_M*1000:.1f}mm removed")
    print(f"{'='*60}\n")

    if out_dir.exists():
        ans = input(f"{out_dir} already exists. Overwrite? [y/N] ").strip().lower()
        if ans != "y":
            print("Aborted.")
            return
        shutil.rmtree(out_dir)

    (out_dir / "data").mkdir(parents=True)
    (out_dir / "videos").mkdir(parents=True)
    (out_dir / "meta").mkdir(parents=True)

    ep_meta       = []
    all_states    = []
    all_actions   = []
    total_frames  = 0
    total_trimmed = 0
    port_counts   = Counter()
    skipped       = 0

    for ep_idx, h5_path in enumerate(tqdm(episodes, desc="Converting")):
        chunk_idx  = ep_idx // CHUNK_SIZE
        chunk_name = f"chunk-{chunk_idx:03d}"
        data_dir   = out_dir / "data"  / chunk_name
        vid_base   = out_dir / "videos" / chunk_name

        with h5py.File(h5_path, "r") as f:
            left   = f["obs/images/left"][:]
            center = f["obs/images/center"][:]
            right  = f["obs/images/right"][:]
            tcp    = f["obs/tcp_pose"][:]
            wrench = f["obs/wrench"][:]
            acts   = f["actions/tcp_pose"][:]

            port_name  = str(f["metadata"].attrs.get("port_name",  "sfp_port_0"))
            port_type, port_index = _parse_port_meta(port_name)

        # ── Trim idle tail ───────────────────────────────────────────────
        tcp, wrench, acts, left, center, right, n_trimmed = _trim_idle_tail(
            tcp, wrench, acts, left, center, right
        )
        total_trimmed += n_trimmed

        n = len(acts)
        if n < 5:
            print(f"  Skipping {h5_path.name} — only {n} steps after trim")
            skipped += 1
            continue

        task_str = _port_to_task_str(port_type, port_index)

        # ── Write videos ─────────────────────────────────────────────────
        _write_mp4(left,   vid_base / "observation.images.left"   / f"episode_{ep_idx:06d}.mp4", fps)
        _write_mp4(center, vid_base / "observation.images.center" / f"episode_{ep_idx:06d}.mp4", fps)
        _write_mp4(right,  vid_base / "observation.images.right"  / f"episode_{ep_idx:06d}.mp4", fps)

        # ── Write parquet ────────────────────────────────────────────────
        _episode_to_parquet(ep_idx, tcp, wrench, acts, port_type, port_index, data_dir)

        # ── Accumulate stats ─────────────────────────────────────────────
        state_with_task = np.concatenate([
            tcp, wrench,
            np.tile(
                [float(port_type == "sfp"), float(port_type == "sc"), float(port_index)],
                (n, 1)
            )
        ], axis=1)
        all_states.append(state_with_task)
        all_actions.append(acts)
        total_frames += n
        port_counts[port_name] += 1

        ep_meta.append({
            "episode_index": ep_idx,
            "tasks":         [task_str],
            "length":        n,
            "port_type":     port_type,
            "port_index":    port_index,
        })

    n_episodes = len(ep_meta)
    if n_episodes == 0:
        print("No valid episodes after conversion.")
        return

    # ── Get image shape from last valid episode ───────────────────────────
    with h5py.File(episodes[-1], "r") as f:
        img_h = int(f["metadata"].attrs.get("image_h", 360))
        img_w = int(f["metadata"].attrs.get("image_w", 480))

    # ── info.json ─────────────────────────────────────────────────────────
    info = {
        "codebase_version": "v2.0",
        "robot_type":       "ur5e",
        "total_episodes":   n_episodes,
        "total_frames":     total_frames,
        "fps":              fps,
        "splits":           {"train": f"0:{n_episodes}"},
        "data_path":        "data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.parquet",
        "video_path":       "videos/chunk-{episode_chunk:03d}/{video_key}/episode_{episode_index:06d}.mp4",
        "features": {
            "observation.images.left": {
                "dtype": "video",
                "shape": [img_h, img_w, 3],
                "video_info": {"video.fps": fps, "video.codec": "mp4v",
                               "video.pix_fmt": "rgb24", "has_audio": False}
            },
            "observation.images.center": {
                "dtype": "video",
                "shape": [img_h, img_w, 3],
                "video_info": {"video.fps": fps, "video.codec": "mp4v",
                               "video.pix_fmt": "rgb24", "has_audio": False}
            },
            "observation.images.right": {
                "dtype": "video",
                "shape": [img_h, img_w, 3],
                "video_info": {"video.fps": fps, "video.codec": "mp4v",
                               "video.pix_fmt": "rgb24", "has_audio": False}
            },
            "observation.state": {
                "dtype": "float32",
                "shape": [16],
                "names": {
                    "motors": [
                        "tcp_x", "tcp_y", "tcp_z",
                        "tcp_qx", "tcp_qy", "tcp_qz", "tcp_qw",
                        "fx", "fy", "fz", "tx", "ty", "tz",
                        "is_sfp", "is_sc", "port_index",
                    ]
                }
            },
            "action": {
                "dtype": "float32",
                "shape": [7],
                "names": {
                    "motors": [
                        "target_x", "target_y", "target_z",
                        "target_qx", "target_qy", "target_qz", "target_qw",
                    ]
                }
            },
        },
    }
    with open(out_dir / "meta" / "info.json", "w") as f:
        json.dump(info, f, indent=2)

    # ── episodes.jsonl ────────────────────────────────────────────────────
    with open(out_dir / "meta" / "episodes.jsonl", "w") as f:
        for ep in ep_meta:
            f.write(json.dumps(ep) + "\n")

    # ── stats.json ────────────────────────────────────────────────────────
    all_s = np.concatenate(all_states,  axis=0)
    all_a = np.concatenate(all_actions, axis=0)

    def _stats(arr):
        return {
            "mean": arr.mean(axis=0).tolist(),
            "std":  arr.std(axis=0).clip(min=1e-6).tolist(),
            "min":  arr.min(axis=0).tolist(),
            "max":  arr.max(axis=0).tolist(),
        }

    stats = {
        "observation.state": _stats(all_s),
        "action":            _stats(all_a),
    }
    with open(out_dir / "meta" / "stats.json", "w") as f:
        json.dump(stats, f, indent=2)

    # ── Summary ───────────────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print(f"  ✓ Conversion complete")
    print(f"{'='*60}")
    print(f"  Episodes converted : {n_episodes}")
    print(f"  Episodes skipped   : {skipped}")
    print(f"  Total frames       : {total_frames}  ({total_frames/fps:.0f}s at {fps}Hz)")
    print(f"  Idle steps trimmed : {total_trimmed}  ({total_trimmed/(total_frames+total_trimmed)*100:.1f}% removed)")
    print(f"\n  Port breakdown:")
    for port, count in sorted(port_counts.items()):
        bar = "█" * count
        print(f"    {port:20s}: {count:3d}  {bar}")
    print(f"\n  Dataset saved to: {out_dir}")
    print(f"\n  To train ACT:")
    print(f"    lerobot-train \\")
    print(f"      --policy=act \\")
    print(f"      --dataset.repo_id=local/aic_lerobot_dataset \\")
    print(f"      --dataset.root={out_dir} \\")
    print(f"      --policy.chunk_size=50 \\")
    print(f"      --training.num_epochs=5000 \\")
    print(f"      --output_dir=~/aic_act_checkpoints")
    print(f"{'='*60}\n")


# ── CLI ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Convert AtomicRecorder HDF5 demos to LeRobot format"
    )
    parser.add_argument("--demo-dir", default=DEFAULT_DEMO_DIR,
                        help=f"Directory containing episode_*.h5 files (default: {DEFAULT_DEMO_DIR})")
    parser.add_argument("--out-dir",  default=DEFAULT_OUT_DIR,
                        help=f"Output directory for LeRobot dataset (default: {DEFAULT_OUT_DIR})")
    parser.add_argument("--fps",      default=DEFAULT_FPS, type=int,
                        help=f"Recording frame rate (default: {DEFAULT_FPS})")
    args = parser.parse_args()
    convert(args.demo_dir, args.out_dir, args.fps)
