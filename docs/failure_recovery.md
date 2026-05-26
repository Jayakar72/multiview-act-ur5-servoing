# Failure-Aware Descent and Recovery — Deep Dive

The Phase 2 / Phase 3 logic that handles everything from the servo's exit point to "plug seated" or "give up." For broader context see [architecture.md](architecture.md).

## What this layer is solving for

By the time descent begins, the wrist is aligned within ~1 mm of the target port. Most descents should succeed without intervention.

But "most" isn't "all." The hard cases:

- **Cable tension stalls.** The free end of the cable, dragged across other NICs during approach, can pull the wrist with several Newtons of lateral force. The Z-axis impedance controller sees no Z motion and (incorrectly) thinks it's hit the rim.
- **Off-by-1mm rim contacts.** The plug is on the rim but not centered. A naive "give up and try again" strategy adds no information; we need to actively search.
- **Narrow-band snags.** A specific Z range (the "lip" between the rim and the inner hole) where the plug catches but isn't seated.

The recovery layer addresses each with a targeted strategy.

## Phase 2: descent with stall classification

### Why "classification" not just "stall detection"

The naive descent loop is:

```python
while z_actual > z_target:
    cmd_z_down_one_step()
    if |z_actual - z_actual_previous| < dz_threshold:
        consecutive_stalls += 1
        if consecutive_stalls >= 5:
            return RIM_CONTACT
    else:
        consecutive_stalls = 0
```

This declares rim contact whenever Z stops moving for 5 ticks. **But Z stops for two distinct reasons, and the right response differs.**

### Reason 1: real rim contact

The plug body has hit the port's outer rim. Z is now at or just above the rim height (`PHASE2_STALL_MIN_Z = 0.2499 m` in world frame). Pushing harder doesn't help — the plug needs to be repositioned XY before more Z can be applied.

**Correct response:** exit to Phase 3 recovery.

### Reason 2: cable-tension stall in air

The cable's free end is dragging the wrist sideways. The Z controller has good Z tracking but the lateral disturbance creates a torque that bleeds into the position controller's authority, slowing actual Z motion below the threshold. The plug isn't touching anything — there's just no Z progress.

**Correct response:** keep pushing, but cap the commanded-vs-actual Z gap so the controller's tracking-error timeout doesn't fire.

### The classifier

The discriminator is the **actual TCP Z** at the stall:

```python
def classify_stall(z_actual):
    if z_actual < PHASE2_STALL_MIN_Z:    # 0.2499 m
        return "REAL_RIM_CONTACT"
    else:
        return "CABLE_TENSION_STALL"
```

`PHASE2_STALL_MIN_Z` is just above the rim's world Z (~0.245 m, depending on board pose). Any stall at Z above this is in free air — it must be cable tension.

### Cable-tension stall handling

```python
if classify_stall(z_actual) == "CABLE_TENSION_STALL":
    # Cap the gap to prevent controller timeout
    if z_cmd < z_actual - 0.005:  # 5mm gap
        z_cmd = z_actual - 0.005  # tighten to 5mm

    consecutive_stalls = 0  # don't count this as progress toward exit
    continue  # keep descending
```

The cap is critical. Without it, `z_cmd` keeps decreasing each tick while `z_actual` stays put. After several ticks the commanded-vs-actual gap exceeds the controller's safety threshold and the controller faults. Capping the gap keeps the controller happy and lets the cable-tension condition resolve naturally (usually within 5–15 ticks the cable settles and Z motion resumes).

### Stiffness in Phase 2

XY stiffness is locked to **800 N/m** — much stiffer than the servo's 150. Reasoning:

- Cable forces during descent are 2–10 N
- At 150 N/m XY stiffness, 5 N lateral disturbance = 33 mm displacement. Catastrophic.
- At 800 N/m, 5 N = 6 mm. Manageable.

Z stiffness remains at 70 N/m — we want compliance for the actual plug-into-port contact.

## Phase 3: mode-gated spiral recovery

When Phase 2 exits with `REAL_RIM_CONTACT`, we're on the rim but not seated. Phase 3 searches for the hole.

### Why a spiral

The plug is roughly centered on the port (±1 mm). The hole is a smaller target inside the rim. A spiral pattern, moving outward at the right step size and inward periodically, has high probability of crossing the hole's small lateral extent.

Spiral parameters:
- Radius: starts at 0 mm, grows to ~6–8 mm
- Step: ~1 mm per revolution
- Z behavior: push down continuously at low gain; if Z drops past `Z_INSERTED_THRESHOLD`, descent succeeds immediately

### Why mode-gated

The pattern of moves *between* spirals depends on which mode the servo ended in. This is the key insight of the V18 iteration.

### YOLO-mode-exit pattern

If the servo ended in YOLO mode, its convergence was actual-port-relative (live YOLO at 7 px tolerance, ~1 mm in world). The wrist is centered on the *actual* port. Symmetric drift covers any residual error:

```
spiral 1 → RETRACT 2.5mm → spiral 2 → RIGHT 2.5mm → spiral 3 → RETRACT → spiral 4 → RIGHT → spiral 5 → RETRACT → spiral 6
```

RETRACT first because the homing geometry tends to overshoot slightly forward; RETRACT covers that bias.

### REG-mode-exit pattern

If the servo ended in REG mode, its convergence was registry-relative. The registry's target XYZ has systematic error from PnP fusion — empirically biased back-right by ~3 mm (the survey rectangle's geometry concentrates detections on the front-left, biasing the cluster medians).

So before spiraling, we apply a **4 mm forward nudge** (toward the board) to compensate for the back-bias. Then we spiral with RIGHT first to cover the right-bias:

```
[4mm forward nudge first]
spiral 1 → RIGHT 2.5mm → spiral 2 → RETRACT 2.5mm → spiral 3 → RIGHT → spiral 4 → RETRACT → spiral 5 → RIGHT → spiral 6
```

### Why this matters

Before the mode-gated pattern (V13 and earlier), we used the same pattern for both modes — RETRACT first, no nudge. SFP trial-2 success rate was around 60%. After the gated pattern (V14+), it climbed to 90%+ on the trials we can reach.

The intuition: the *right* search direction depends on the *bias* in our current estimate. YOLO mode has different biases than REG mode, so a single pattern can't be optimal for both.

### Stiffness in Phase 3

XY drops to **70 N/m** — even softer than the servo. Reasoning:

- We're now in contact with the rim. Stiff XY would mean rigid pushing against the rim, not slipping into the hole.
- Soft XY lets the plug "find" the hole — small force differences (rim normal vs. hole opening) translate to lateral motion that helps the plug slip in.

Z stiffness stays at 70 N/m.

## Narrow-band rim-snag rescue

A specific failure observed early: the plug's tip enters the hole partway, then catches on the inner lip at Z ≈ 0.234–0.235 m (a ~1 mm Z band). Z doesn't move; pushing harder doesn't help (the plug is bent slightly, applying lateral force on the lip).

### Detection

After every spiral or move, check Z. If `0.2340 < z_actual < 0.2350`, declare narrow-band snag.

### Response

```python
1. Lift to Z = 0.2380 (4-5 mm above the snag band)
2. Move forward (+local_y) by 8 mm
3. Retry the spiral
```

The lift unsticks the plug. The forward move shifts the lateral angle, so the next descent contacts the hole differently. This single check rescues roughly 5–10% of would-otherwise-fail trials.

## Wiggle-while-pushing (last-resort descent)

If 6 spirals all fail to seat the plug, we make one final attempt with a different strategy: **dynamic XY wiggle while continuously pushing Z**.

```python
for t in range(WIGGLE_TICKS):  # ~3 seconds
    wiggle_x = 0.0015 * sin(2*pi*2*t/RATE)  # ±1.5 mm at 2 Hz
    wiggle_y = 0.0015 * cos(2*pi*2*t/RATE)
    cmd_xy(current_xy + (wiggle_x, wiggle_y))
    cmd_z(z_actual - 0.0005)  # 0.5 mm/tick downward

    if z_actual < Z_INSERTED:
        return SUCCESS
```

The intuition is mechanical: a plug stuck on the rim's edge can sometimes be coaxed in by small dynamic XY motion *while* downward force is applied. Static spirals visit each XY position once; the wiggle revisits them at different Z depths as Z slowly progresses.

This is the "last 5%" — the trials where geometric search has failed but dynamic agitation seats the plug.

## Exit conditions for Phase 3

```
SUCCESS         → Z reached insertion target, plug seated
MAX_RECOVERY    → All spirals exhausted, wiggle exhausted; give up cleanly
PHYSICAL_LIMIT  → TCP hit a workspace boundary; can't continue safely
```

For `MAX_RECOVERY` and `PHYSICAL_LIMIT` we lift the wrist back to a safe Z and report failure to the engine. The trial is over.

## Tuning notes

- **`PHASE2_STALL_MIN_Z = 0.2499`** — set just above the *highest* rim Z observed across all eval configs. If your eval set has a different board height, retune.
- **`REG_PRE_SPIRAL_NUDGE_M = 0.004`** — empirical, from observing where REG-mode failures concentrated XY-wise. Lower (2 mm) under-corrects; higher (8 mm) overshoots.
- **`SPIRAL_STEP_NUDGE_M = 0.0025`** — small enough that consecutive spirals overlap (no missed XY positions), large enough to cover the full 8 mm search radius in 4 nudges.
- **Spiral radius and step** — the spiral parameters in code use 6–8 mm peak radius and ~1 mm/turn step. Smaller is more thorough but slower; larger is faster but may skip the hole.

## Cross-references

- The flag `self._last_servo_mode` is set in [visual_servoing.md](visual_servoing.md)
- Stiffness modes overview: [architecture.md](architecture.md)
- Z-coordinate landmarks (rim, insert, safe): see the constant section at the top of [`../policy/MultiViewACTPolicy.py`](../policy/MultiViewACTPolicy.py)
