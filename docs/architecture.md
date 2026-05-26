# Architecture

This document describes the full runtime pipeline of `MultiViewACTPolicy` — what runs when, what each phase produces, and how the phases hand off to one another.

For a less code-centric overview, see the [main README](../README.md).

## Phase chart

```
                ┌──────────────────────────────────────────────────────────────┐
                │   Phase 0  ─  Initialization                                 │
                │                                                              │
                │   ─ Load ACT checkpoint (policy.pt + norm_stats.npy)        │
                │   ─ Load YOLO-OBB checkpoint (best.pt)                       │
                │   ─ Build camera-intrinsics cache                            │
                │   ─ Subscribe to wrench, joints, /tf, images, camera_info    │
                └─────────────────────────────┬────────────────────────────────┘
                                              │ ready to take tasks
                                              ▼
                ┌──────────────────────────────────────────────────────────────┐
                │   Phase 1  ─  Pre-homing scene survey                        │
                │                                                              │
                │   For each of 8 survey waypoints:                            │
                │     1. Move TCP to waypoint (light stiffness, slow)          │
                │     2. Wait SURVEY_MOVE_SETTLE_S seconds                     │
                │     3. For SURVEY_DETECT_DURATION_S seconds:                 │
                │        For each camera:                                      │
                │          ─ Run YOLO-OBB on latest frame                      │
                │          ─ Form same-NIC port pairs (2 ports per NIC)        │
                │          ─ Run PnP per pair, transform to base frame         │
                │          ─ Append result to detection list                   │
                │                                                              │
                │   Output:  raw_detections = List[{xyz, yaw, conf, ...}]      │
                └─────────────────────────────┬────────────────────────────────┘
                                              │
                                              ▼
                ┌──────────────────────────────────────────────────────────────┐
                │   Phase 1.5  ─  Registry construction                        │
                │                                                              │
                │   1. Cluster raw_detections by XY within 14 mm radius        │
                │   2. Drop clusters with n_obs < 2 (likely YOLO noise)        │
                │   3. For each cluster:                                       │
                │        ─ port_0_world  = median XY of port-0 in cluster      │
                │        ─ port_1_world  = median XY of port-1 in cluster      │
                │        ─ midpoint, yaw, n_obs, conf_median                   │
                │   4. Solve mount-index assignment via:                       │
                │        ─ Anchor: target_module_name from task message        │
                │        ─ Rail-spacing bands (40/80/120/160 mm)               │
                │        ─ Backtracking search                                 │
                │        ─ Yaw-direction monotonicity filter                   │
                │        ─ TCP-distance tiebreak                               │
                │                                                              │
                │   Output:  self.nic_registry = Dict[mount_idx, registry_row] │
                └─────────────────────────────┬────────────────────────────────┘
                                              │ if target mount in registry
                                              ▼
                ┌──────────────────────────────────────────────────────────────┐
                │   Phase 1.6  ─  Cable-aware approach to homing               │
                │                                                              │
                │   1. Pick nearest survey-rectangle corner to target          │
                │      (excluding right-side corners to keep cable safe)       │
                │   2. Walk TCP to that corner along survey edges              │
                │      (rectangle perimeter, not straight-line)                │
                │   3. Compose plug-straight tilt into homing orientation      │
                │      (skip the tilt-in-place phase entirely)                 │
                │   4. Home above target port at ~25 mm Z standoff             │
                └─────────────────────────────┬────────────────────────────────┘
                                              │
                                              ▼
                ┌──────────────────────────────────────────────────────────────┐
                │   Phase 1.7  ─  Hybrid visual servo                          │
                │                                                              │
                │   Loop until pixel error ≤ tol or max iterations:            │
                │                                                              │
                │     mode = "yolo" initially                                  │
                │     ┌─ if mode == "yolo":                                    │
                │     │    detect on center camera                             │
                │     │    if detection within 25mm of registered target:      │
                │     │       pixel_err = |target_px - detected_center_px|     │
                │     │       tol = 7px                                        │
                │     │    else:                                               │
                │     │       miss_count += 1                                  │
                │     │       if miss_count ≥ 4: mode = "reg"                  │
                │     │                                                        │
                │     └─ if mode == "reg":                                     │
                │          project registered world-pose through camera K      │
                │          pixel_err = |target_px - projected_center_px|       │
                │          tol = 2px                                           │
                │                                                              │
                │     command TCP correction in image plane direction          │
                │     light stiffness (150 N/m XY)                             │
                │                                                              │
                │   Record self._last_servo_mode = mode at every exit          │
                └─────────────────────────────┬────────────────────────────────┘
                                              │
                                              ▼
                ┌──────────────────────────────────────────────────────────────┐
                │   Phase 2  ─  Failure-aware descent                          │
                │                                                              │
                │   1. Switch to descent stiffness (800 N/m XY, 70 N/m Z)      │
                │   2. Command Z downward at ~5 mm/s                           │
                │   3. Each tick:                                              │
                │        ─ Check Z motion: |dz| < 0.2 mm?                      │
                │        ─ If 5 consecutive stalls:                            │
                │            classify by actual TCP Z:                         │
                │              Z < PHASE2_STALL_MIN_Z  → real rim → exit       │
                │              Z ≥ PHASE2_STALL_MIN_Z  → cable tension stall:  │
                │                  reset stall counter, cap |cmd_z - act_z|    │
                │                  keep pushing                                │
                │   4. If Z reaches insertion target → SUCCESS                 │
                └─────────────────────────────┬────────────────────────────────┘
                                              │ rim contact
                                              ▼
                ┌──────────────────────────────────────────────────────────────┐
                │   Phase 3  ─  Mode-gated spiral recovery                     │
                │                                                              │
                │   if self._last_servo_mode == "reg":                         │
                │     apply 4mm forward nudge (compensate PnP bias)            │
                │                                                              │
                │   For idx in 0..5 (6 spirals):                               │
                │     spiral(70 N/m soft, ~6-8mm radius, ~5sec, push down)     │
                │     if Z drops past rim threshold: descend → SUCCESS         │
                │                                                              │
                │     # Inter-spiral nudge (alternates by mode):               │
                │     if last_servo_mode == "reg":                             │
                │       even idx: RIGHT 2.5mm   odd idx: RETRACT 2.5mm         │
                │     else (yolo):                                             │
                │       even idx: RETRACT 2.5mm odd idx: RIGHT 2.5mm           │
                │                                                              │
                │   if narrow-band Z stall (0.234 < Z < 0.235):                │
                │     lift to 0.238 Z, forward 8mm, retry                      │
                │                                                              │
                │   Final attempt: wiggle-while-pushing descent                │
                │     XY oscillate ±1.5mm at 2Hz while pushing Z down          │
                │                                                              │
                │   Exit: SUCCESS, MAX_RECOVERY, or PHYSICAL_LIMIT             │
                └──────────────────────────────────────────────────────────────┘
```

## State carried across phases

A handful of variables in `self` carry information between phases:

| Variable | Set by | Read by | Meaning |
|----------|--------|---------|---------|
| `self.nic_registry` | Phase 1.5 | Phases 1.6, 1.7, 3 | World-frame pose registry |
| `self.target_mount_idx` | task message | Phases 1.5–3 | Which NIC to drive to |
| `self._target_world_pose` | Phase 1.5 | Phases 1.6, 1.7, 3 | Registered pose of target |
| `self._last_servo_mode` | Phase 1.7 (every exit) | Phase 3 | `"yolo"` or `"reg"` — gates spiral pattern + REG nudge |
| `self._homed_corner` | Phase 1.6 | Phase 3 (retract direction) | Which rectangle corner we approached from |

## Stiffness modes

Three named modes get pushed to the impedance controller:

| Mode | XY stiffness | Z stiffness | When |
|------|-------------:|------------:|------|
| `SURVEY` | 150 N/m | 150 N/m | Phase 1 (waypoint moves) |
| `SERVO` | 150 N/m | 150 N/m | Phase 1.7 (visual servo) |
| `DESCENT_LOCK` | 800 N/m | 70 N/m | Phase 2 (forced descent) |
| `SPIRAL_SOFT` | 70 N/m | 70 N/m | Phase 3 (spiral recovery) |

The XY split between `DESCENT_LOCK` (800) and `SPIRAL_SOFT` (70) is the most important: during forced descent we need rigid XY to resist cable forces; during spiral search we need compliance so the plug actually slips into the hole rather than skating across the rim.

## Configuration knobs

The class header of `MultiViewACTPolicy.py` exposes ~50 configurable constants. The ones most likely to matter for tuning:

| Constant | Default | Effect |
|----------|--------:|--------|
| `SURVEY_RECT_WIDTH` | 0.145 m | XY rectangle size — too small misses outer NICs |
| `SURVEY_RECT_HEIGHT` | 0.10 m | Same, other axis |
| `SURVEY_MOVE_SETTLE_S` | 3.0 s | Wait between move and detect (motion blur) |
| `SURVEY_DETECT_DURATION_S` | 2.0 s | How long to accumulate detections at each waypoint |
| `DETECT_CONF_THRESH` | 0.50 | YOLO confidence floor |
| `XY_CLUSTER_RADIUS_M` | 0.014 m | Registry-clustering radius |
| `SERVO_PIXEL_TOLERANCE` | 7 px | YOLO-mode servo convergence |
| `SERVO_PIXEL_TOLERANCE_REG` | 2 px | REG-mode servo convergence |
| `SERVO_YOLO_FALLBACK_MISSES` | 4 | YOLO misses before mode switch |
| `SERVO_MAX_ITERATIONS` | 45 | Hard cap on servo loop |
| `WORLD_POSE_LOCK_RADIUS_M` | 0.025 m | Reject YOLO detections this far from registered target |
| `PHASE2_STALL_MIN_Z` | 0.2499 m | Real-rim threshold |
| `REG_PRE_SPIRAL_NUDGE_M` | 0.004 m | Forward nudge before REG-mode spirals |
| `SPIRAL_STEP_NUDGE_M` | 0.0025 m | Inter-spiral lateral nudge size |

## Cross-references

- ACT training: [act_training.md](act_training.md)
- Multi-view scene registration: [multi_view_perception.md](multi_view_perception.md)
- Visual servoing: [visual_servoing.md](visual_servoing.md)
- Recovery logic: [failure_recovery.md](failure_recovery.md)
