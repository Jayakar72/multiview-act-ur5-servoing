# Data Collection

This folder houses everything needed to collect the two datasets the pipeline depends on:

1. **ACT demonstrations** — 302 teleop episodes, saved as HDF5
2. **YOLO-OBB training data** — ~5,160 auto-labeled wrist-camera images

Both happen *inside* the AIC simulation environment, but they use different driver policies.

## What's in this folder

| File | Purpose |
|------|---------|
| `AtomicRecorder.py` | The recorder policy. Mounted into the AIC engine, records every observation/action pair during teleoperation. Saves one HDF5 per trial to `~/aic_demos/`. |
| `auto_collect.sh` | Loops the eval pipeline against a given config until the demo directory reaches a target episode count. |
| `collect_port_data.py` | Standalone ROS 2 node that subscribes to the 3 wrist cameras + `/tf`, projects ground-truth port corners onto each image, and saves YOLO-OBB labels alongside the JPEG. |
| `convert_to_lerobot.py` | Optional converter: reads HDF5 demos and writes a LeRobot-compatible dataset (parquet + meta JSONL). Useful if you want to use LeRobot's training stack instead of our custom `train_act.py`. |

---

## 1. Collecting ACT demonstrations

### How `AtomicRecorder` works

`AtomicRecorder` is a Policy subclass that does **not** generate actions on its own — it expects an external driver (teleoperation or `CheatCode`) to produce actions. On every call, the recorder:

1. Captures the observation (3 cameras + state vector)
2. Captures the action being applied this step
3. Appends both to an in-memory buffer
4. On trial-end, flushes the buffer to a numbered HDF5 file:

```
~/aic_demos/
  episode_000001.h5
  episode_000002.h5
  ...
```

Each HDF5 file contains arrays of shape `(T, ...)` where `T` is the trial length in policy ticks:

| Key | Shape | dtype | Meaning |
|-----|-------|-------|---------|
| `images/cam_0` | (T, 224, 224, 3) | uint8 | Wrist camera 0 |
| `images/cam_1` | (T, 224, 224, 3) | uint8 | Wrist camera 1 |
| `images/cam_2` | (T, 224, 224, 3) | uint8 | Wrist camera 2 |
| `state` | (T, 16) | float32 | TCP pose (7) + wrench (6) + port encoding (3) |
| `action` | (T, 7) | float32 | TCP target pose |
| `meta/port_type` | scalar | str | "sfp" or "sc" |
| `meta/rail_idx` | scalar | int | Mount index, 0–4 |
| `meta/success` | scalar | bool | Trial result |

### Running a collection session

The recorder is mounted as the policy via the standard AIC engine mechanism, with an external driver doing the actual control.

For **CheatCode-driven collection** (most automated, what `auto_collect.sh` uses):

```bash
# Terminal 1 — eval sim
distrobox enter -r aic_eval -- /entrypoint.sh \
  start_aic_engine:=true \
  aic_engine_config_file:=<your-workspace>/src/aic/training/configs/diverse/config_rail0.yaml

# Terminal 2 — recorder mounted, CheatCode drives the robot
cd <your-workspace>/src/aic
pixi run ros2 run aic_model aic_model --ros-args \
  -p use_sim_time:=true \
  -p policy:=aic_example_policies.ros.AtomicRecorder \
  -p driver_policy:=aic_example_policies.ros.CheatCode
```

For **teleoperated collection** (higher quality, requires a teleop setup):

Replace `driver_policy` with your teleop policy. The recorder's interface is the same.

### Auto-looped collection

The `auto_collect.sh` script automates the "run trial → count files → run another" loop:

```bash
bash data_collection/auto_collect.sh \
  --config training/configs/diverse/config_rail0.yaml \
  --target 80
```

It tears down and restarts the sim cleanly between trials. Run it overnight per config.

### The 302-episode distribution

| Config | Target demos | Why |
|--------|------------:|-----|
| `config_rail0.yaml` | 77 | Rail 0 is heavily sampled by the eval portal |
| `config_rail1.yaml` | 75 | Same |
| `config_rail2.yaml` | 73 | Common in eval |
| `config_rail3_4.yaml` | ~50 | Less common, smaller weight |
| `config_sc.yaml` | 98 | All SC trials |
| **Total** | **302** | |

The weighting matches what Intrinsic's `sample_config.yaml` shows the portal draws from. Don't waste demos on configurations the evaluator rarely samples.

### Common pitfalls

- **Don't move the demos directory mid-training.** The HDF5 files are read by file index; renaming or reordering breaks the dataset.
- **Watch for zero-length episodes.** If the eval sim crashes before the trial starts, you'll get a 0-row HDF5. Filter these out before training:
  ```bash
  for f in ~/aic_demos/*.h5; do
    n=$(python -c "import h5py; print(h5py.File('$f')['state'].shape[0])")
    [[ $n -lt 50 ]] && rm "$f"
  done
  ```
- **Don't mix successful and failed trials.** `AtomicRecorder` saves both. Train only on `meta/success == True` — `train_act.py` does this filtering by default.

---

## 2. Collecting YOLO-OBB training data

### Why a separate collector

YOLO-OBB needs **per-image OBB labels** in the Ultralytics format:

```
class_id x1 y1 x2 y2 x3 y3 x4 y4    (normalized to [0, 1])
```

The four corners must be in a known order. Manual labeling at the scale we needed (5,000+ images) was impractical. We instead used the simulation's ground-truth TF tree to project known 3D port corners onto each camera frame, automatically.

### How `collect_port_data.py` works

It's a ROS 2 node that subscribes to:

- `/cam_0/image_raw`, `/cam_1/image_raw`, `/cam_2/image_raw`
- `/cam_0/camera_info`, etc. — for intrinsics
- `/tf` — for ground-truth port poses (only available when `ground_truth:=true`)

On every received image:

1. Look up the GT pose of every port_link_entrance frame visible from the current camera
2. For each visible port, generate the 4 corners of its known 3D rectangle in port_link_entrance frame:
   - SFP: 13.4 × 8.4 mm
   - SC: 10.2 × 10.2 mm
3. Transform corners to camera frame, then project to image plane using `K` from `/camera_info`
4. If all 4 corners fall inside the image bounds (with a margin), save:
   - `image_<run_id>_<frame_idx>.jpg`
   - `image_<run_id>_<frame_idx>.txt` (YOLO label)
   - Optionally a debug overlay image with corners drawn

The output directory structure:

```
~/aic_yolo_dataset/
├── images/
│   ├── image_rail0_000001.jpg
│   ├── image_rail0_000002.jpg
│   └── ...
├── labels/
│   ├── image_rail0_000001.txt
│   ├── image_rail0_000002.txt
│   └── ...
└── debug/                              # optional, can disable
    └── image_rail0_000001_overlay.jpg
```

### Running the collector

Three terminals, one config at a time. **The key flag is `ground_truth:=true`** — without it, the `/tf` tree won't include port poses.

```bash
# Terminal 1 — eval sim with GT TFs published
distrobox enter -r aic_eval -- /entrypoint.sh \
  ground_truth:=true \
  start_aic_engine:=true \
  aic_engine_config_file:=<your-workspace>/src/aic/training/configs/diverse/config_rail0.yaml

# Terminal 2 — CheatCode drives the robot through varied scenes
cd <your-workspace>/src/aic
pixi run ros2 run aic_model aic_model --ros-args \
  -p use_sim_time:=true \
  -p policy:=aic_example_policies.ros.CheatCode

# Terminal 3 — collector
cd <your-workspace>/src/aic
pixi run python data_collection/collect_port_data.py \
  --ros-args -p run_id:=rail0
```

Let it run until you have ~1,000 images per config. Repeat for `rail1`, `rail2`, `rail3_4`, and the default `sample_config.yaml`.

### Dataset sizes we ended up with

| Config | Images |
|--------|-------:|
| rail0 | ~1,000 |
| rail1 | ~1,000 |
| rail2 | ~1,000 |
| rail3_4 | ~1,000 |
| default (sample_config) | ~1,160 |
| **Total** | **~5,160** |

After collection, prep into train/val splits with `perception/prepare_obb_dataset.py` (see [`../perception/README.md`](../perception/README.md)).

### Why this beat the manual route

Our first YOLO attempt used 190 manually-labeled images and a YOLOv8n. mAP50 was excellent (0.995) but mAP50-95 was only 0.700, and crucially — **standard YOLO doesn't give you orientation**. We'd have had to manually rotate-label every box, which the Flask labeler we built didn't support.

Auto-labeling via 3D projection:
- Removed the per-image labor cost entirely
- Gave us pixel-perfect 4-corner labels (no human ambiguity)
- Let us scale to 5,000+ images for a larger YOLOv8s-OBB model
- Final mAP50-95 ≈ 0.79–0.80 — far better than the manual nano

This is the most important lesson from the perception side of the project: **if you have ground-truth in simulation, use it to generate supervision rather than manually labeling.**

---

## 3. Optional: convert HDF5 demos to LeRobot dataset format

Our training script (`training/train_act.py`) reads HDF5 directly. But if you'd prefer to use LeRobot's full training stack (their `Trainer`, dataset loader, evaluation framework), you can convert your demos:

```bash
pixi run python data_collection/convert_to_lerobot.py \
  --demo_dir ~/aic_demos \
  --output ~/aic_lerobot_dataset \
  --repo_id <your-namespace>/aic-cable-insertion
```

The converter:
- Reads every `episode_*.h5` from `--demo_dir`
- Writes parquet + meta JSONL in LeRobot's [v2.0 dataset format](https://github.com/huggingface/lerobot/blob/main/docs/source/lerobot_dataset.mdx)
- Preserves all observation keys (3 images + state) and the action key
- Optionally pushes to the Hugging Face Hub via the `--repo_id` argument

Use this if:
- You want to share your dataset on the HF Hub
- You're switching to LeRobot's training loop for ablation studies
- You want to use other LeRobot policies (Diffusion Policy, VQ-BeT, etc.) on the same data

Skip this if you're happy with our custom `train_act.py`.

---

## Cross-references

- ACT training: [`../training/README.md`](../training/README.md)
- YOLO training: [`../perception/README.md`](../perception/README.md)
- Where this data fits in the pipeline: [`../docs/architecture.md`](../docs/architecture.md)
