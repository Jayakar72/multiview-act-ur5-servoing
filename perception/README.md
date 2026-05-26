# Perception: YOLO-OBB + PnP

This folder is the **6-DoF port pose recovery pipeline**. Given a wrist-camera image, it produces world-frame (x, y, z, yaw) for every visible SFP or SC port.

The pipeline has two stages:

1. **YOLOv8-OBB** detects ports as oriented bounding boxes (4 corners, not just a 2D box)
2. **PnP** solves for the 6-DoF pose from those 4 corners using the port's known 3D dimensions

## What's in this folder

### Training pipeline

| File | Purpose |
|------|---------|
| `train_obb.py` | Ultralytics training driver for YOLOv8s-OBB. |
| `prepare_obb_dataset.py` | Takes the raw collected dataset, builds train/val splits and the Ultralytics YAML config. |

### Test and debug utilities

| File | Purpose |
|------|---------|
| `yolo_obb_live_test.py` | Runs the trained model on a live camera feed from the AIC simulator. Shows annotated frames in an OpenCV window for sanity-checking detection quality. |
| `yolo_obb_pose_test.py` | End-to-end test: runs YOLO → PnP → world-frame pose on each detection. Prints recovered poses for visual comparison with the GT TF tree (when `ground_truth:=true`). |
| `yolo_obb_pose_test_v2.py` | Version 2 of the pose test. Adds per-frame pose logging to disk for later batch analysis. |
| `analyze_pose_log.py` | Reads the pose logs produced by `yolo_obb_pose_test_v2.py` and computes per-detection error statistics (mean/median XYZ error, yaw error, reprojection error histograms). Use this to quantify perception accuracy before deploying. |

Recommended workflow for new contributors: train via `train_obb.py`, sanity-check with `yolo_obb_live_test.py`, quantify with `yolo_obb_pose_test_v2.py` → `analyze_pose_log.py`.

The PnP solver itself lives in the [`MultiViewACTPolicy`](../policy/MultiViewACTPolicy.py) file (see methods `_solve_pnp_best_perm`, `_make_corners_3d`, `_detect_ports_in_image`). It's tightly coupled to the runtime — we don't carve it out into a separate module because that would force loading ROS/AIC dependencies at perception-development time.

---

## 1. The YOLO journey: from nano to OBB

This was the most important pivot in the project. Documenting it because the *first* approach reads like an obviously-correct idea, and the failure mode wasn't visible until we tried using the outputs downstream.

### Attempt 1 — YOLOv8n with manual labels (deprecated)

**Setup:**
- Saved ~408 wrist-camera images during teleop
- Built a Flask-based labeling tool, drew bounding boxes on 190 of them
- Class: 1 (`nic_port`)
- Trained YOLOv8n for 50 epochs at imgsz 480, batch 16
- 8 minutes on the RTX 4070

**Results:** mAP50 = 0.995, mAP50-95 = 0.700

**Why we threw it away:** The 2D bounding box gives you the port's **center** and rough **extent**, but nothing about its **orientation**. When the port is viewed obliquely (the typical case for our wrist-camera geometry), the box corners are **not** the port corners — they're the axis-aligned bounding rectangle of a rotated quadrilateral.

PnP needs the actual port corners, in known correspondence with the 3D model, to recover orientation. A 2D box gives you 2 unknowns short of what you need.

We considered:
- Adding a "yaw" regression head to YOLO — non-standard, requires custom training code
- Using corner heatmaps — significantly more annotation work
- Manually rotating each labeled box — tedious and error-prone

The cleaner solution was sitting one model away: **YOLOv8-OBB**, which natively outputs 4 corners per detection.

### Attempt 2 — YOLOv8s-OBB with auto-labels (production)

OBB models output `(x_center, y_center, width, height, angle)` — equivalent to 4 corners with a known order. This is exactly what PnP needs.

The catch: OBB training needs OBB labels. Manually annotating oriented rectangles is even more tedious than axis-aligned ones, so we built the **auto-labeling pipeline** described in [`../data_collection/README.md`](../data_collection/README.md).

**Setup:**
- 5,160 auto-labeled images from CheatCode driving with `ground_truth:=true`
- 80/20 train/val split → 4,129 train + 1,031 val
- Classes: 2 (`sfp_slot`, `sc_slot`)
- Model: YOLOv8s-OBB (small, ~11M params — 3× the nano)
- Trained 100 epochs at imgsz 640, batch 16 (early-stopped at epoch ~10–12)

**Results:** mAP50 ≈ 0.965, mAP50-95 ≈ 0.79–0.80

The mAP50 dropped slightly from the nano (0.995 → 0.965) but mAP50-95 jumped from 0.700 → 0.79–0.80. The latter is what matters for downstream PnP — it measures localization tightness at higher IoU thresholds, which directly correlates with corner-pixel accuracy.

---

## 2. Training the OBB model

### Dependencies

Add Ultralytics to your Pixi environment:

```bash
pixi add --pypi ultralytics
```

### Prepare the dataset

After collecting raw images and labels into `~/aic_yolo_dataset/` (see [data_collection](../data_collection/README.md)), build the training-ready dataset:

```bash
cd <your-workspace>/src/aic
pixi run python perception/prepare_obb_dataset.py
```

This produces:

```
~/aic_yolo_dataset/
├── images/train/   ← 80% of images
├── images/val/     ← 20% of images
├── labels/train/
├── labels/val/
└── data.yaml       ← Ultralytics dataset config
```

The `data.yaml` looks like:

```yaml
path: /home/<user>/aic_yolo_dataset
train: images/train
val:   images/val
names:
  0: sfp_slot
  1: sc_slot
```

### Train

```bash
pixi run python perception/train_obb.py
```

Defaults (in `train_obb.py`):
- `--model s` (YOLOv8s-OBB)
- `--epochs 100`
- `--imgsz 640`
- `--batch 16`
- `--device 0`

If you'd rather skip the wrapper script entirely:

```bash
pixi run yolo obb train \
  data=~/aic_yolo_dataset/data.yaml \
  model=yolov8s-obb.pt \
  epochs=100 imgsz=640 batch=16 device=0
```

### Watch for convergence

Set `epochs=100` as a ceiling but watch `mAP50-95` in the validation metrics every epoch:

```
Epoch 1   mAP50: 0.62   mAP50-95: 0.42
Epoch 5   mAP50: 0.91   mAP50-95: 0.71
Epoch 10  mAP50: 0.96   mAP50-95: 0.79     ← plateau begins
Epoch 12  mAP50: 0.965  mAP50-95: 0.80
Epoch 20  mAP50: 0.965  mAP50-95: 0.80     ← no further gains
```

When mAP50-95 stops improving for 3–5 epochs, kill training. The auto-labels are so consistent that the model converges fast.

Total wall-clock training time on RTX 4070: well under one hour.

### Output

```
~/aic_yolo_runs/sfp_obb_v1/
├── weights/
│   ├── best.pt      ← what you ship
│   └── last.pt
├── results.csv
└── val_batch*.jpg   ← visualizations of validation predictions
```

`best.pt` is copied into the Docker image as `/root/aic_yolo_models/best.pt`. The policy reads it via the `AIC_YOLO_CHECKPOINT` env var.

---

## 3. PnP: 4 corners → 6-DoF pose

Once YOLO gives us 4 ordered corners, we recover the port's 6-DoF pose in the camera frame, then transform to the robot base frame.

### The 3D model

Each port is a flat rectangle with known dimensions, measured from Intrinsic's URDF:

| Port type | Width | Height |
|-----------|-------|--------|
| SFP | 13.4 mm | 8.4 mm |
| SC | 10.2 mm | 10.2 mm |

The 3D corners in port frame (CCW, bottom-left first):

```python
def _make_corners_3d(w, h):
    return np.array([
        [-w/2, -h/2, 0.0],   # bottom-left
        [+w/2, -h/2, 0.0],   # bottom-right
        [+w/2, +h/2, 0.0],   # top-right
        [-w/2, +h/2, 0.0],   # top-left
    ], dtype=np.float64)
```

### The PnP call

```python
ok, rvec, tvec = cv2.solvePnP(
    objectPoints=corners_3d,
    imagePoints=corners_2d,
    cameraMatrix=K,           # from /camera_info
    distCoeffs=D,             # zeros in sim
    flags=cv2.SOLVEPNP_ITERATIVE,
)
```

`rvec` and `tvec` give us the port-frame-to-camera-frame transform. Compose with `T_base_camera` (from the TF tree) to get the port pose in the robot base frame.

### The corner-order ambiguity

This caused us hours of debugging. YOLO-OBB outputs 4 corners in *some* order, but **the order isn't guaranteed to match the 3D model's order**. The convention rotates depending on the box orientation:

- A box at 0° might output corners as `[BL, BR, TR, TL]` (matching our 3D model)
- The same box rotated 90° might output `[BR, TR, TL, BL]`
- Or `[TR, TL, BL, BR]`
- Or `[TL, BL, BR, TR]`

If you naively pass mismatched correspondences to PnP, you get a pose that's rotated 90°/180°/270° from the truth. The XY translation is approximately correct (the box center is invariant) but yaw is wrong — and yaw is what your downstream planning cares about.

The fix is to **try all 4 cyclic permutations** and pick the one with the lowest reprojection error:

```python
def _solve_pnp_best_perm(corners_3d, corners_2d, K, D):
    best = None
    for perm in range(4):
        rotated = np.roll(corners_2d, -perm, axis=0)
        ok, rvec, tvec = cv2.solvePnP(corners_3d, rotated, K, D,
                                       flags=cv2.SOLVEPNP_ITERATIVE)
        if not ok:
            continue
        proj, _ = cv2.projectPoints(corners_3d, rvec, tvec, K, D)
        err = np.linalg.norm(
            proj.reshape(-1, 2) - rotated, axis=1
        ).mean()
        if best is None or err < best[2]:
            best = (rvec, tvec, err, perm)
    return best
```

The correct permutation gives mean reprojection error of ~0.5–2 pixels. The wrong permutations typically give 20+ pixels. Picking by minimum error is reliable.

### Camera intrinsics

In simulation, `/camera_info` publishes exact intrinsics — the `K` matrix and zero distortion. No calibration needed. For real-robot deployment you'd run a checkerboard calibration first.

### Where the output goes

Each successful detection produces:
- `port_xyz_world` — 3D position in robot base frame
- `port_yaw_world` — rotation around vertical axis
- `pnp_reproj_err` — used as a quality metric for downstream clustering

These per-detection results stream into the multi-view scene registration logic (see [`../docs/multi_view_perception.md`](../docs/multi_view_perception.md)).

---

## 4. Sample detections

Snapshots from a live trial showing the OBB detector's output across the three wrist cameras. Each oriented box is a port; the confidence value and class label (sfp_slot or sc_slot) are drawn above each box.

**Scene A — initial spawn pose**

| Left | Center | Right |
|:-:|:-:|:-:|
| ![](../assets/YoloLogs/left_00001.jpg) | ![](../assets/YoloLogs/center_00001.jpg) | ![](../assets/YoloLogs/right_00001.jpg) |

**Scene B — partway through the survey rectangle**

| Left | Center | Right |
|:-:|:-:|:-:|
| ![](../assets/YoloLogs/left_00033.jpg) | ![](../assets/YoloLogs/center_00032.jpg) | ![](../assets/YoloLogs/right_00029.jpg) |

These are typical detections — ports near the image center yield 90%+ confidence; ports near edges or in oblique views drop to 60–80% (still well above the `DETECT_CONF_THRESH = 0.50` floor).

---

## Cross-references

- The PnP code itself: [`../policy/MultiViewACTPolicy.py`](../policy/MultiViewACTPolicy.py) (search for `_solve_pnp_best_perm`)
- Multi-view registry construction: [`../docs/multi_view_perception.md`](../docs/multi_view_perception.md)
- How the registry feeds the visual servo: [`../docs/visual_servoing.md`](../docs/visual_servoing.md)
- Data collection (the auto-labeling pipeline): [`../data_collection/README.md`](../data_collection/README.md)
