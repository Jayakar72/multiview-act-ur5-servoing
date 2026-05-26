# Visual Servoing — Deep Dive

The closed-loop alignment phase that brings the TCP from "approximately above the port" to "aligned within 2 px of the port center" before descent begins. Lives in Phase 1.7 of the pipeline.

For broader context, see [architecture.md](architecture.md).

## Why visual servoing at all

After homing (Phase 1.6), we're typically within 5–15 mm of the port. Why not just descend?

Because:
- **5–15 mm at 0.5 mm port clearance is 10× the tolerance.** Plug will jam every time.
- **Homing accuracy is limited by registry accuracy.** Registry XY is typically ±3 mm; cumulative pose chain (base → flange → tcp) adds another 1–2 mm. We can't expect homing alone to put us within 0.5 mm.
- **Image-based feedback can close the loop to sub-pixel accuracy.** The wrist's center camera can see the port at ~1 mm/pixel at the homing standoff. Sub-pixel alignment converts to sub-mm world accuracy.

The servo's job is to consume the remaining 5–15 mm of XY error using camera images directly, without needing to update the registry.

## Two-mode architecture

The servo operates in one of two modes. The mode is chosen dynamically per iteration and recorded at exit (Phase 3 reads it).

### YOLO mode (default)

Inputs:
- Live YOLO detection on the center camera's most recent frame
- The registered target's world-frame XYZ (from `self.nic_registry`)

Per iteration:
1. Run YOLO-OBB on the latest frame
2. For each detection, run PnP and transform to base frame
3. **World-pose lock check**: reject any detection whose world position is more than 25 mm from the registered target (it's an adjacent NIC, not our target)
4. If a valid detection remains: compute pixel error `(detected_center - target_pixel)`
5. Convergence tolerance: 7 px
6. Command a TCP correction proportional to pixel error, scaled by depth

### REG mode (fallback)

Inputs:
- The registered target's world-frame XYZ (from `self.nic_registry`)
- Live camera intrinsics
- Current TCP pose

Per iteration:
1. Project the registered world point through the camera's intrinsic + extrinsic chain to get its expected pixel position
2. Compute pixel error `(projected_center - target_pixel)`
3. Convergence tolerance: 2 px (tighter — projection is deterministic, no detection noise)
4. Command a TCP correction proportional to pixel error

### Mode transition

```
Start: mode = "yolo", miss_count = 0

Each iteration:
  if mode == "yolo":
    if YOLO finds a valid (within 25mm of registry) detection:
      use it, reset miss_count
    else:
      miss_count += 1
      if miss_count >= SERVO_YOLO_FALLBACK_MISSES (= 4):
        mode = "reg"

  if mode == "reg":
    use projected registered pose (always succeeds)
```

Once switched to REG, the servo stays there for the remainder of this phase. (We don't try to switch back — REG is more reliable once we've given up on YOLO, and the cost of a switch oscillation isn't worth it.)

## Why two modes?

### Why YOLO is the default

YOLO at convergence gives **sub-millimeter accuracy** when it works. The detection localizes the actual port (not the registry's noisy estimate of it), so any inaccuracy in the registry is eliminated.

### Why REG is the fallback

YOLO fails when:
- The port falls outside the center camera's field of view (camera angle changed during approach)
- Cable, plug body, or another NIC partially occludes the port from this viewpoint
- Confidence drops below the threshold (sometimes due to motion or unfortunate lighting)

In these cases, REG mode uses the registry — which was built from 8 viewpoints of fused evidence — to compute pixel error analytically. The servo can still converge even if YOLO has given up.

The cost: REG mode inherits the registry's accuracy. If the registry's target XYZ is 4 mm off, REG mode will park the wrist 4 mm off. That's worse than YOLO's sub-mm convergence — but better than failing entirely.

### Why not always REG?

Why not just always use REG (it's always available)? Because YOLO + world-pose lock gives **higher accuracy at the same complexity**. The world-pose lock prevents YOLO from latching onto neighbors, so the only failure mode for YOLO mode is "YOLO doesn't see the port." Until that happens, prefer YOLO.

## The world-pose lock

This single mechanism prevented an entire class of failures: the servo latching onto an adjacent NIC.

### The failure mode (before the lock)

Picture this: registered target is mount 2. Mount 2 is at world XY = (-0.42, +0.30). Mount 3 is at (-0.42, +0.34) — 40 mm to the side. After homing, the wrist is 5 mm off-target in the +Y direction. The center camera sees both mount 2 (the target) and mount 3 (adjacent), both fully in frame.

Without a lock: YOLO returns the **highest confidence** detection. If mount 3's view is slightly better (sharper angle, less cable in frame), the servo latches onto it. The wrist moves toward mount 3 — and now mount 3 has even higher confidence. The servo converges on the wrong NIC.

### The fix

```python
def is_target_detection(detection, registry_target_xyz):
    det_world = pnp_to_world(detection)
    return np.linalg.norm(det_world[:2] - registry_target_xyz[:2]) < 0.025  # 25mm
```

A detection is only used if its PnP-recovered world position is within 25 mm of the registered target. Adjacent NICs at 40 mm separation are reliably rejected.

### Why 25 mm?

- Too small (e.g., 10 mm): rejects valid detections during early servo iterations when alignment is rough.
- Too large (e.g., 50 mm): allows adjacent NICs back in.
- 25 mm: comfortably accepts the actual target through the full 0–15 mm initial error range, while still rejecting adjacent NICs.

## Control law

For both modes, the pixel error is converted to a TCP correction:

```python
# Compute pixel error
pixel_err_x = target_pixel_x - actual_or_projected_pixel_x  # px
pixel_err_y = target_pixel_y - actual_or_projected_pixel_y  # px

# Convert to world-frame meters using known camera-to-target distance
mm_per_pixel = depth_m / focal_length_px
world_corr_x = pixel_err_x * mm_per_pixel
world_corr_y = pixel_err_y * mm_per_pixel

# Camera frame to TCP frame transform
tcp_corr_x = -world_corr_x   # camera X = TCP +X (signs from URDF)
tcp_corr_y = -world_corr_y   # camera Y = TCP +Y

# Apply gain — fraction of error per step
gain = 0.7
tcp_target_x = current_tcp_x + gain * tcp_corr_x
tcp_target_y = current_tcp_y + gain * tcp_corr_y
```

A gain of 0.7 means we move 70% of the way to the target per step. Conservative enough to avoid overshoot at low impedance; aggressive enough to converge in a handful of iterations.

## Stiffness mode during servo

The impedance controller is set to:
- **XY stiffness: 150 N/m** — light. We're not yet contacting anything; over-stiff XY would cause overshoot at the low control gain.
- **Z stiffness: 150 N/m** — light. We're holding altitude, not pushing.
- **Damping: 5 N·s/m** — slight; just enough to suppress oscillation.

If we switched to descent stiffness (800 N/m XY) during servoing, small commanded corrections would produce sharp transients that overshoot.

## Iteration cap

`SERVO_MAX_ITERATIONS = 45`. The servo loop runs at the policy tick rate (~20 Hz), so 45 iterations = ~2.25 seconds wall time.

If convergence hasn't happened by then, we exit anyway. The last mode (`yolo` or `reg`) is recorded, and Phase 2 (descent) takes over with whatever alignment we achieved. If alignment is bad, Phase 3 (spiral recovery) catches it.

## Exit conditions and recorded state

Three ways the servo exits:

1. **Converged** — pixel error below mode-specific tolerance. Best outcome.
2. **Iteration cap** — 45 iterations reached. Reasonable alignment, might still succeed at descent.
3. **No detection persistent** — YOLO never finds the target *and* the registered pose projects outside the image. Rare. The servo exits early with a flag indicating this; Phase 2 then uses extra-conservative descent.

In all three cases, `self._last_servo_mode` is set to the current `mode` value. Phase 3 reads this flag to gate its spiral pattern + REG nudge.

## Cross-references

- The flag `self._last_servo_mode` is consumed by [failure_recovery.md](failure_recovery.md)
- The registry the servo reads from is built in [multi_view_perception.md](multi_view_perception.md)
- Stiffness modes overview: [architecture.md](architecture.md)
