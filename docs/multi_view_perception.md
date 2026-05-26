# Multi-View Scene Perception — Deep Dive

The most novel contribution of the project. This document explains *how* the survey pattern, PnP, clustering, and constraint solver come together to produce a world-frame NIC registry from raw images.

For training details (YOLO + auto-labels), see [`../perception/README.md`](../perception/README.md).

## The problem statement

Given:
- A UR5e arm with 3 RGB wrist cameras (intrinsics known)
- A task message: `target_module_name = "nic_card_mount_2"`
- A scene with 2–5 NICs at unknown rail indices and yaws

Produce:
- World-frame XYZ + yaw of the target NIC, accurate to <5 mm and <2°

The naïve approach — "look at the scene once, run YOLO, pick the most-confident detection" — fails for several reasons:

1. **Occlusion.** From any single TCP pose, some NICs are hidden behind others.
2. **PnP noise from oblique angles.** A port viewed at >60° from normal gives noisy 6-DoF poses (the 4-corner geometry degenerates).
3. **Adjacent-NIC confusion.** Two NICs on adjacent rails are 40 mm apart. YOLO can match either to a target description.
4. **No mount-index information from detection alone.** YOLO tells you "I see SFP ports here" — not "those are mount 0 and that's mount 2."

Tier 2's job is to fuse multiple views, separate physical NICs cleanly, and assign mount indices using geometric priors.

## Step 1 — Survey rectangle pattern

### Why a rectangle, not a circle or grid

We tried several survey patterns:
- **Single pose at spawn** — fast, but ~25% of trials had one NIC heavily occluded
- **Linear sweep along Y** — better, but pure forward motion gives no XY parallax
- **Grid of 9 waypoints** — best coverage, but ~30 seconds total — too slow
- **Rectangle with 8 waypoints** — what we ship; 12-15 sec total with good XY+Y coverage
- **Hexagon** — equivalent quality to rectangle but more code

The rectangle wins because:
- 4 corners give wide XY parallax
- 4 midpoints sit closer to the scene, where YOLO confidence is highest
- The rectangle's *edges* double as cable-aware approach paths in Tier 3 — reusing the same path for two purposes

### The waypoint layout

In TCP-frame relative to spawn (numbers from `MultiViewACTPolicy.py`):

```
                +Y forward (toward board)
                          │
                          │
       (-X,+Y)─────────(0,+Y)──────────(+X,+Y)
            │                                │
            │                                │
       (-X,0)         (0,0) spawn         (+X,0)
            │                                │
            │                                │
       (-X,-Y)─────────(0,-Y)──────────(+X,-Y)
                          │
                          │
                -Y away (back to spawn)
```

With `SURVEY_RECT_WIDTH = 0.145 m`, `SURVEY_RECT_HEIGHT = 0.10 m`, the rectangle is ~145 mm × 100 mm centered at spawn. The order traversed (8 waypoints + return) is: spawn → front-left → front-mid → front → right-mid → right-back → back-mid → back-left → left-mid → spawn.

### Settling and detection cadence

At each waypoint:
- `SURVEY_MOVE_SETTLE_S = 3.0 s` after arrival — wait for the wrist to stop vibrating; motion blur ruins YOLO confidence
- `SURVEY_DETECT_DURATION_S = 2.0 s` — collect frames at the camera's natural rate (10–15 Hz), feed each through YOLO
- All 3 cameras processed per frame

Total survey time: 8 waypoints × 5 sec ≈ 40 sec of which ~24 sec is settle+detect (the rest is motion). Total detections per trial: 100–300 raw detections.

## Step 2 — Per-frame: YOLO → port pairs → PnP

YOLO-OBB outputs 4-corner detections per image (see [`../perception/README.md`](../perception/README.md)). The next steps run per frame:

### Pair ports into NICs

A NIC card has **two visible SFP ports** side by side (top and bottom). The registry tracks NICs, not individual ports, so we pair detections that are likely on the same card:

- Iterate all pairs of detections in the same image
- For each pair, check:
  - Both must be `sfp_slot` class
  - Centers within 30 px vertical distance
  - Centers within 80 px horizontal distance
  - Confidence both >= `DETECT_CONF_THRESH` (0.50)
- If multiple pairs match a single port, prefer the one with smallest distance

Unmatched ports are discarded — a NIC with only one visible port is unreliable.

### PnP per pair

For each paired NIC:
- Take the 4 corners of port_0 and 4 corners of port_1 (8 total points)
- Run `_solve_pnp_best_perm` on each port separately (see [`../perception/README.md`](../perception/README.md) for corner-order handling)
- Median the two pose estimates (typically agree within 1–2 mm)
- Transform to robot base frame using current TF lookup

### What gets appended to `raw_detections`

```python
{
    "port_0_world_xyz":  (3,) numpy array,
    "port_1_world_xyz":  (3,) numpy array,
    "midpoint_world_xyz": (3,) numpy array,
    "yaw_world":         float,
    "conf":              float (min of the two ports),
    "reproj_err":        float (mean of port_0 and port_1 reproj errs),
    "from_camera_id":    int (0/1/2),
    "from_waypoint":     int (0-7),
}
```

After the full survey: typically 50–200 entries in `raw_detections` (much fewer than the raw 100–300 per-image detections, because pairing filters most out).

## Step 3 — Clustering into registry entries

Each `raw_detection` is a noisy world-frame measurement of one physical NIC. We need to group them.

### Clustering algorithm

```python
def cluster_detections(raw_detections, radius=0.014):
    clusters = []
    for det in raw_detections:
        for cluster in clusters:
            if np.linalg.norm(det["midpoint_world_xyz"][:2]
                            - cluster["centroid_xy"]) < radius:
                cluster["members"].append(det)
                cluster["centroid_xy"] = np.median(
                    [m["midpoint_world_xyz"][:2] for m in cluster["members"]],
                    axis=0
                )
                break
        else:
            clusters.append({
                "centroid_xy": det["midpoint_world_xyz"][:2].copy(),
                "members": [det],
            })
    return clusters
```

Simple greedy assignment within a 14 mm XY radius (Z is dropped because PnP Z is noisier).

### Why 14 mm?

The nearest possible inter-NIC distance is 40 mm (adjacent rails). With per-detection PnP noise of σ ≈ 4 mm, two detections of the same NIC will scatter within ~3σ = 12 mm. The radius needs to be larger than expected intra-cluster spread (~12 mm) but smaller than the minimum inter-NIC spread (40 mm).

14 mm gives a comfortable margin both ways:
- Cluster radius (14) > 3σ spread (12) — same-NIC detections cluster correctly
- Adjacent-NIC distance (40) - cluster diameter (28) = 12 mm separation — adjacent NICs stay distinct

### Cluster filtering

After clustering:
- Drop clusters with `n_members < 2` (single-detection clusters are usually false positives)
- For each remaining cluster, compute:
  - Median XY of `port_0_world_xyz`, `port_1_world_xyz`, `midpoint_world_xyz`
  - Circular median of `yaw_world` (use atan2 of mean sin/cos to avoid wraparound)
  - Number of members `n_obs`
  - Median confidence

These become **registry rows** indexed eventually by mount index (next step).

## Step 4 — Mount-index assignment

Registry rows have positions but no labels yet. The AIC engine refers to NICs by mount index (`nic_card_mount_0` through `nic_card_mount_4`), with the target specified in the task message. We need to map registry rows to mount indices.

### Geometric constraints

The task board has 5 rails, 40 mm apart. Each NIC slides on its rail with ±22.5 mm translation. So the world-frame distance between any two NICs falls into bands depending on their rail-index difference:

| `|i - j|` | Min distance | Max distance |
|-----------|-------------:|-------------:|
| 1 | 40.0 mm | 45.6 mm |
| 2 | 80.0 mm | 82.8 mm |
| 3 | 120.0 mm | 121.9 mm |
| 4 | 160.0 mm | 161.4 mm |

The bands are nearly non-overlapping (smallest gap is 80.0 - 45.6 = 34 mm) — pairwise distances unambiguously identify rail-index differences.

### Backtracking search

For each registry row, we want a mount index in `{0, 1, 2, 3, 4}`. The anchor constraint is from the task: one row's mount index is fixed by `target_module_name`.

We then enumerate all assignments of mount indices to the other rows that are consistent with the pairwise distance bands:

```python
def solve_mount_indices(rows, anchor_row_idx, anchor_mount_idx):
    n = len(rows)
    pairwise = compute_pairwise_distances(rows)

    candidates = []
    for assignment in itertools.permutations(range(5), n):
        if assignment[anchor_row_idx] != anchor_mount_idx:
            continue
        ok = all(
            distance_band(pairwise[i,j]) == abs(assignment[i] - assignment[j])
            for i, j in itertools.combinations(range(n), 2)
        )
        if ok:
            candidates.append(assignment)
    return candidates
```

Most scenes admit a unique solution (1 candidate). When multiple survive, we apply the yaw filter next.

### Yaw-direction monotonicity filter

The board has a clear "increasing mount index" direction in world frame. If `yaw` is the median NIC yaw, this direction is `(sin(yaw), -cos(yaw), 0)` (depends on URDF convention).

A valid assignment must produce mount indices that are monotonic when registry rows are projected onto this direction. Non-monotonic assignments are rejected — they imply the board is upside-down or that the registry has duplicates.

In code:

```python
def yaw_monotonicity_check(rows, assignment, yaw):
    direction = np.array([np.sin(yaw), -np.cos(yaw)])
    projections = [(np.dot(row["midpoint_world_xy"], direction), mnt)
                   for row, mnt in zip(rows, assignment)]
    projections.sort(key=lambda p: p[1])  # sort by mount index
    return all(projections[i][0] < projections[i+1][0]
               for i in range(len(projections)-1))
```

### Tiebreak by TCP distance

If two assignments survive the yaw filter (rare — happens in symmetric 3-NIC scenes), break the tie by choosing the assignment whose target row is closest to the current TCP position. The reasoning: the spawn TCP is biased to be slightly forward of the board center, so the "correct" target is usually the closer one.

This last step is the only place we admit ambiguity remains.

## Step 5 — Registry as canonical truth

After all steps, `self.nic_registry` is a dictionary:

```python
{
    0: {
        "port_0_world_xyz":  (3,) array,
        "port_1_world_xyz":  (3,) array,
        "midpoint_world_xyz": (3,) array,
        "yaw_world":         float,
        "n_obs":             int,
        "conf_median":       float,
    },
    2: { ... },
    3: { ... },
    # mount indices for which we have registry rows; not necessarily 0..4
}
```

For the rest of the policy run (Phases 1.6, 1.7, 3), this dict is the canonical scene model. Live YOLO detections during the servo phase are *validated against* it (the 25 mm world-pose lock), not allowed to override it.

This is intentional: the registry was built from 8 viewpoints over 24 seconds of focused observation. A live YOLO detection from a single camera at one pose can't compete with that statistical evidence.

## Known limitations

- **Heavy initial occlusion.** If at all 8 survey waypoints, a NIC is fully occluded by another NIC, it never enters the registry. We have not yet implemented an active scene-completion mechanism.
- **Trial 2's 3-NIC ambiguity.** When 3 NICs are visible with target = mount 1, the yaw-monotonicity filter is consistent with two distinct assignments (one rotated 180°). The TCP-distance tiebreak picks wrong roughly 25% of the time on these specific configurations.
- **Static scene assumption.** The registry doesn't update during Tier 3. If the board is moved mid-trial (it isn't, in the eval environment) the registry would be stale.

## Cross-references

- YOLO/PnP details: [`../perception/README.md`](../perception/README.md)
- Visual servo (consumer of registry): [visual_servoing.md](visual_servoing.md)
- Spiral recovery (consumer of registry): [failure_recovery.md](failure_recovery.md)
- Architecture overview: [architecture.md](architecture.md)
