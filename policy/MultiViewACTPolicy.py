#!/usr/bin/env python3
"""
MultiView ACT UR5 Servoing — AI for Industry Challenge
MultiViewACTPolicy.py — hybrid ACT + multi-view perception + visual servoing

Architectural shift from V13:
  • All YOLO detection happens BEFORE homing.
  • Robot performs a 4-waypoint anti-clockwise square SURVEY around the spawn
    point (front → front-left → back-left → spawn), gathering multi-angle
    observations of the task board.
  • All detected NIC pairs across all waypoints are CLUSTERED in world frame
    to build a registry: one entry per physical NIC with median world-frame
    port positions, midpoint, yaw, and observation count.
  • Mount indices (0..4) are assigned to registry entries via V13's rail-
    spacing geometry + the task's `target_module_name` as the anchor.
  • The TARGET is then looked up in the registry. Its STORED world pose
    drives homing, tilt, and the visual servo (which now FILTERS live YOLO
    detections to only those within 2cm of the registered target world
    position — preventing the servo from latching onto an adjacent NIC).
  • Post-homing, the policy operates almost entirely on registry data.

Phases:
  Phase 0    — Quick check at spawn (1s, just to see if any pair visible)
  Phase 0.3  — Scout (bidirectional) IF Phase 0 returned 0 pairs
  Phase 0.5  — SURVEY (NEW): 4-waypoint anti-clockwise square pattern
               around current TCP, ~12s, multi-camera multi-frame detection
  Phase 0.6  — REGISTRY (NEW): cluster all survey detections → NIC list
  Phase 0.7  — MOUNT-INDEX ASSIGNMENT (NEW): rail-spacing geometry
  Phase 0.8  — TARGET LOOKUP (NEW): find entry matching task.target_module_name
               with F1/F2/F3 fallback chain
  Phase 1    — Home to REGISTERED target (no live YOLO)
  Phase 1.6  — Local-frame tilt + yaw (uses registered yaw)
  Phase 1.7  — Visual servo with WORLD-POSE LOCK (D3):
               live YOLO is still used for pixel refinement, but ONLY
               detections within 2cm of registered target are accepted
  Phase 2    — Forced descent
  Phase 3    — Spiral recovery

Author: Team ATOMIC
"""
import os
import math
import time
from collections import deque
import cv2
import numpy as np
import torch
from aic_control_interfaces.msg import MotionUpdate, TrajectoryGenerationMode
from aic_model.policy import (
    GetObservationCallback,
    MoveRobotCallback,
    Policy,
    SendFeedbackCallback,
)
from aic_task_interfaces.msg import Task
from geometry_msgs.msg import Point, Pose, Quaternion, Wrench, Vector3
from rclpy.time import Time
from sensor_msgs.msg import Image
from tf2_ros import TransformException
# ── Configuration ────────────────────────────────────────────────────────
DEFAULT_YOLO_CHECKPOINT = os.environ.get(
    "AIC_YOLO_CHECKPOINT",
    os.path.expanduser("~/aic_yolo_models/best.pt"))
DEFAULT_ACT_CHECKPOINT = os.environ.get(
    "ATOMIC_ACT_CHECKPOINT",
    os.path.expanduser("~/aic_act_checkpoints/final"))
# Detection — MAIN (strict)
DETECT_CONF_THRESH    = 0.50      # v14.3: lowered 0.75→0.50 — post-tilt
                                  # views and oblique angles drop confidence
                                  # below 0.75 even for valid detections
DETECT_DURATION_S     = 3.0       # how long to collect detections
DETECT_RATE_HZ        = 10        # how often to grab obs and run YOLO
DETECT_MIN_RECORDS    = 6         # need at least N to trust the fusion
DETECT_PAIR_DISTANCE_PX_MAX = 300 # was 200 — handles close-up camera positions
# ── V13 NEW: Multi-NIC disambiguation ─────────────────────────────────────
# Rail geometry (from aic_description/urdf/task_board.urdf.xacro):
#   5 NIC rails along task-board Y axis, 40mm apart
#   Each NIC slides along task-board X (translation ∈ [-0.0215, +0.0234] m)
#   So translation range is ~45mm. Yaw per NIC: ±10°.
RAIL_SPACING_M             = 0.040   # 40mm between adjacent rails
RAIL_TRANSLATION_RANGE_M   = 0.045   # max translation difference between 2 NICs
RAIL_BAND_TOLERANCE_M      = 0.008   # 8mm tolerance on band classification
# Pair validity: world distance between port_0 and port_1 on a real NIC
PAIR_PORT_DIST_MIN_M       = 0.010   # 10mm — below this, not a real pair
PAIR_PORT_DIST_MAX_M       = 0.025   # 25mm — above this, likely cross-NIC false pair
# Hypothesis search: how many top-confidence detections to consider in pair-building
MULTI_NIC_TOP_K_DETECTIONS = 8       # was 4 in v12 — extend for multi-NIC
# ── End V13 multi-NIC constants ───────────────────────────────────────────
# ── V14 NEW: Pre-homing survey ────────────────────────────────────────────
SURVEY_ENABLED            = True
# Survey path options:
#   "square":   ORIGINAL 4-waypoint anti-clockwise (spawn→+Y→-X→-Y→spawn)
#   "forward":  single +Y step + long detect (simplest)
#   "elevated": rise + forward + circle at altitude (most viewpoints)
#   "ushape":   3-waypoint path ending in front of spawn
SURVEY_MODE               = "square"   # back to original per user request
# --- square mode constants (legacy name kept — now rectangle) ---
# V15.3: survey path is now a RECTANGLE (was square at 8cm).
#   Forward (+Y):  SURVEY_RECT_HEIGHT_M = 9 cm
#   Left   (-X):  SURVEY_RECT_WIDTH_M  = 12 cm
#   Back   (-Y):  SURVEY_RECT_HEIGHT_M = 9 cm (returns to spawn Y)
#   Right  (+X):  SURVEY_RECT_WIDTH_M  = 12 cm (returns to spawn)
# The 12cm horizontal extent gives the cameras a wider parallax baseline
# in the dimension where NICs are spaced along the rails.
SURVEY_RECT_HEIGHT_M      = 0.10    # +Y dimension (forward/back leg)
SURVEY_RECT_WIDTH_M       = 0.145    # +X dimension (left/right leg)
SURVEY_SQUARE_SIDE_M      = 0.09    # kept for non-"square" survey modes
                                    # (forward/elevated/ushape) that
                                    # still reference this name. The
                                    # "square" mode itself uses the
                                    # explicit rect constants above.
# --- forward mode constants ---
SURVEY_FORWARD_DIST_M       = 0.12
SURVEY_FORWARD_SETTLE_S     = 1.0
SURVEY_FORWARD_DETECT_S     = 2.0
# --- elevated mode constants ---
SURVEY_ELEVATED_RISE_M      = 0.05
SURVEY_ELEVATED_RISE_SETTLE_S = 1.5
SURVEY_ELEVATED_FORWARD_M   = 0.10
SURVEY_ELEVATED_FORWARD_SETTLE_S = 1.5
SURVEY_CIRCLE_POINTS        = 6
SURVEY_CIRCLE_RADIUS_M      = 0.05
SURVEY_CIRCLE_SETTLE_S      = 1.0
SURVEY_CIRCLE_DETECT_S      = 1.0
# v14.1: slowed down — robot was still moving when detection started, so
# YOLO got motion-blurred frames and many waypoints returned 0 detections.
SURVEY_MOVE_SETTLE_S      = 3.0  # time to settle after each waypoint move
SURVEY_DETECT_DURATION_S  = 2.0  # detection burst duration per waypoint
SURVEY_CLUSTER_RADIUS_M   = 0.014   # v14.3: 20mm → 14mm. Adjacent rails are
                                    # 40mm apart; with PnP XY noise ~4mm, a
                                    # 14mm radius gives ~12mm boundary
                                    # buffer between clusters and prevents
                                    # adjacent NICs from merging.
SURVEY_MIN_OBS_PER_NIC    = 2       # need at least N observations per cluster

# ── V15 NEW: Pre-homing approach via survey square edges ──────────────────
# After Phase 0.8 TARGET FOUND, walk along the survey square edges to the
# CORNER of the square that is closest to the target NIC, THEN let Phase 1
# homing take over. The square corners (spawn, front, front-left, back-left)
# are the same XY positions visited during the survey, so the cable has
# already been routed through them — walking along these edges keeps the
# free cable end on the same trajectory it took during survey and avoids
# the diagonal "spawn → target" move that pulled mount_0 into the
# cable's path when targeting mount_4.
V15_APPROACH_ENABLED      = True
V15_APPROACH_SETTLE_S     = 3.0     # bumped 2.0→2.5 to MATCH
                                    # SURVEY_MOVE_SETTLE_S exactly.
                                    # Each approach waypoint now gets
                                    # the same cable-settle time as
                                    # scouting waypoints, so the cable
                                    # fully stabilizes and "follows
                                    # the plug" before the next move.
V15_APPROACH_FINAL_PAUSE_S = 1.5    # extra hold AFTER reaching the
                                    # final approach corner, BEFORE
                                    # Phase 1 homing starts. Lets the
                                    # cable fully relax into its new
                                    # configuration before the wrist
                                    # commits to the homing trajectory.
V15_APPROACH_SKIP_MARGIN_M = 0.04   # if target XY is within this distance of
                                    # spawn, skip the approach entirely
                                    # (small targets near spawn benefit from
                                    # direct homing instead).

# ── V15 NEW: Narrow-band rim-snag recovery (before spiral) ────────────────
# When Phase 2's forced descent stalls in a narrow Z band just above the
# port rim, the cable's free end has often caught on the rim and is pulling
# the wrist back. A direct spiral search from this position won't help —
# the wrist can't move down at all. Instead we LIFT a few mm and NUDGE
# FORWARD by ~8mm to slip past the snag, then hand off to spiral.
V15_RECOVERY_ENABLED      = True
V15_RECOVERY_STALL_Z_MIN  = 0.2340  # narrow band lower bound
V15_RECOVERY_STALL_Z_MAX  = 0.2350  # narrow band upper bound — between
                                    # these two Z values, the plug is
                                    # almost certainly hung up on the rim.
V15_RECOVERY_LIFT_TARGET_Z = 0.2380  # target Z to lift to (small upward
                                    # nudge to release rim contact)
V15_RECOVERY_LIFT_MAX_M   = 0.03    # safety cap — never lift more than
                                    # this from current Z, regardless of
                                    # target_z computation
V15_RECOVERY_FORWARD_M    = 0.008   # forward (+Y in world frame) nudge
                                    # after lift, to shift the plug from
                                    # "above the rim" to "above the hole"
V15_RECOVERY_SETTLE_S     = 1.0     # settle time between lift and nudge
SURVEY_REGISTRY_LOG_VERBOSE = True
# Visual servo (Phase 1.7) — D3 world-pose lock for v14:
# Accept live YOLO detections ONLY if within this radius of the registered
# target world position. Prevents servo from latching onto an adjacent NIC.
SERVO_WORLD_LOCK_RADIUS_M = 0.025   # 25mm
SERVO_WORLD_LOCK_ENABLED  = True
# Initial quick check at spawn (before deciding to scout vs survey)
SPAWN_QUICK_CHECK_DURATION_S = 1.0
SPAWN_QUICK_CHECK_MIN_PAIRS  = 1
# ── End V14 survey constants ──────────────────────────────────────────────
# Phase 0.5 — Bidirectional Scout
# When Phase 0 returns no detection (board out-of-frame at spawn), the scout
# walks the gripper BACKWARD first (since v10's forward-only scout missed
# T2-style spawn poses), then returns to spawn and goes FORWARD as fallback.
SCOUT_ENABLED        = True
SCOUT_MAX_STEPS      = 8      # was 6; per direction (so 16 max total)
SCOUT_STEP_M         = 0.020  # 2 cm per step
SCOUT_DURATION_S     = 1.5    # short detection burst per step
SCOUT_STABILIZE_S    = 0.4    # pause after each step before detecting
SCOUT_MIN_RECORDS    = 3      # NEW: relaxed (main detection uses 6)
SCOUT_CONF_THRESH    = 0.30   # NEW: relaxed (main detection uses 0.75)
# Slot 3D dims in port_link_entrance frame (Z=0 = opening face, CCW corners)
SFP_W, SFP_H = 0.0134, 0.0084
SC_W,  SC_H  = 0.0102, 0.0102
# Canonical gripper-target relative to port (in PORT_LINK_ENTRANCE frame).
SFP_OFFSET_IN_PORT = np.array([-0.014, 0.025, 0.140])
SC_OFFSET_IN_PORT  = np.array([-0.014, 0.025, 0.140])
# Gripper rotation in port frame at canonical = R_y(π)  ↔  R_x(π) in base at yaw=π
R_Y_PI = np.array([[-1.0,  0.0,  0.0],
                   [ 0.0,  1.0,  0.0],
                   [ 0.0,  0.0, -1.0]], dtype=np.float64)
# Homing — single pose command + settle, V10 style
HOMING_MAX_DIST_M   = 0.30      # safety: refuse if target > 30 cm from current
HOMING_SETTLE_TIME  = 1.5       # was 1.0 — longer settle after big homing move
POST_TILT_SETTLE_S  = 0.7       # NEW v12: settle after Phase 1.6 tilt before Phase 1.7 servo
HOMING_MIN_MOVE_M   = 0.005     # already at target if closer than this
# Settle times around major phases — robot stabilization
PRE_HOMING_PAUSE   = 0.3        # was 0.5
POST_HOMING_PAUSE  = 0.3        # was 1.0
POST_ACT_PAUSE     = 0.3        # was 1.0
POST_FALLBACK_PAUSE = 0.2       # was 0.5
POST_TRIAL_PAUSE   = 0.2        # was 2.0 — replaced by spawn-Z lift on next trial
# ── v12: spawn-Z return between trials ──────────────────────────────
# If previous trial descended (success or partial), next trial starts with
# TCP deep in/near the port. Capture spawn Z on first trial; on later trials
# lift TCP straight back up to spawn Z before Phase 0 detection.
SPAWN_Z_RETURN_THRESHOLD_M = 0.03  # lift if TCP starts >3cm below spawn
SPAWN_Z_LIFT_SETTLE_BASE_S = 0.5   # base settle time after lift
SPAWN_Z_LIFT_SETTLE_PER_M  = 6.0   # extra seconds per meter of lift
# Wiggle (lift+retract+push) — when ACT stalls below DESCENT_TRIGGER_Z
WIGGLE_LIFT_Z        = 0.2390
WIGGLE_RETRACT_M     = 0.007
WIGGLE_SUCCESS_DEPTH = 0.005
WIGGLE_PAUSE_S       = 0.5
# Visual servoing (image-based closed-loop alignment, after Phase 1 homing)
SERVO_ENABLED         = True
SERVO_CAMERA          = "center"
SERVO_PIXEL_TOLERANCE = 7             # YOLO mode convergence threshold
# V15.7: REG-mode convergence threshold (tighter than YOLO since the
# registered-pose projection is deterministic — no per-frame YOLO noise).
# When SERVO_YOLO_FALLBACK_TO_REG fires and servo_mode becomes "reg",
# convergence is judged against this value instead of SERVO_PIXEL_TOLERANCE.
# Pixel→world at typical descent height (Z_cam≈0.13m, fx≈850):
#   1 px ≈ 0.15mm, 3 px ≈ 0.45mm, 5 px ≈ 0.75mm
# Tighter for REG since plug-to-port radial clearance is ~0.5-1.0mm.
SERVO_PIXEL_TOLERANCE_REG = 2
SERVO_MAX_ITERATIONS  = 45
SERVO_CONF_THRESH     = 0.10   # v14.3: 0.30→0.20 — oblique post-tilt view
                                # drops YOLO confidence; need to catch those
SERVO_GAIN            = 0.25
SERVO_MAX_STEP_M      = 0.008
SERVO_SETTLE_S        = 0.30
SERVO_CONVERGE_FRAMES = 3
SERVO_DEBUG_TOPIC     = "/aic_model/servo_debug"
# Two-stage servo
SERVO_STAGE1_REF_PIXEL = (574.0, 480.0) ############################
SERVO_STAGE1_STABILIZE_S = 1.0
PRE_DESCENT_STABILIZE_S  = 0.5
# Plug orientation compensation (separate POST-HOMING step)
SET_PLUG_STRAIGHT_AFTER_HOMING = True
# Debug mode: stop after homing (no descent, no ACT).
DEBUG_STOP_AFTER_PLUG_STRAIGHT = False
# SC pipeline gate. False = SC trials use V9 naked ACT.
SC_PIPELINE_ENABLED = False
# Gripper offset (plug pose in TCP frame) — from challenge sample_config.yaml
GRIPPER_OFFSET_T_SFP = np.array([0.0, 0.015385, 0.04245])
GRIPPER_OFFSET_T_SC  = np.array([0.0, 0.015385, 0.04045])
GRIPPER_OFFSET_RPY   = (0.4432, -0.4838, 1.3303)
# Firmer stiffness/damping for the homing move (V10 values)
HOMING_STIFFNESS = [150.0, 0, 0, 0, 0, 0,
                     0, 150.0, 0, 0, 0, 0,
                     0, 0, 150.0, 0, 0, 0,
                     0, 0, 0, 80.0, 0, 0,
                     0, 0, 0, 0, 80.0, 0,
                     0, 0, 0, 0, 0, 80.0]
HOMING_DAMPING   = [60.0, 0, 0, 0, 0, 0,
                     0, 60.0, 0, 0, 0, 0,
                     0, 0, 60.0, 0, 0, 0,
                     0, 0, 0, 25.0, 0, 0,
                     0, 0, 0, 0, 25.0, 0,
                     0, 0, 0, 0, 0, 25.0]
# V15: Stiffer XY during visual servo to defeat cable tension.
# The cable's free end can hang up and pull the wrist back during
# Phase 1.7 alignment, causing pixel error to plateau (we saw err
# stuck at 50-80px while iter count climbed). Bumping XY stiffness
# 150 → 250 N/m boosts push force from 1.2 N → 2.0 N at the 8mm
# servo step clamp — enough to win against typical cable drag while
# still ~10× below the 20 N/-12 pt F-T safety penalty threshold.
# Z stiffness left at 150 (servo locks Z, doesn't try to push down).
# Damping bumped proportionally to keep the system overdamped
# (damping ratio ≈ 1.37 with effective wrist mass ~3 kg).
SERVO_STIFFNESS  = [150.0, 0, 0, 0, 0, 0,
                     0, 150.0, 0, 0, 0, 0,
                     0, 0, 150.0, 0, 0, 0,
                     0, 0, 0, 80.0, 0, 0,
                     0, 0, 0, 0, 80.0, 0,
                     0, 0, 0, 0, 0, 80.0]
SERVO_DAMPING    = [75.0, 0, 0, 0, 0, 0,
                     0, 75.0, 0, 0, 0, 0,
                     0, 0, 60.0, 0, 0, 0,
                     0, 0, 0, 25.0, 0, 0,
                     0, 0, 0, 0, 25.0, 0,
                     0, 0, 0, 0, 0, 25.0]
# V15: Servo starts in YOLO mode (live detection — works fine when the
# port is visible). After N consecutive YOLO misses, fall back to the
# REGISTERED world pose from the survey and project it through the
# camera each iteration to compute pixel error analytically. This way
# trial 1 (where YOLO works reliably) keeps the noisy-but-accurate
# live feedback, and trial 2 (where YOLO loses the port partway
# through due to cable in view / occlusion) gracefully switches to a
# deterministic projection that never says "no detection".
SERVO_YOLO_FALLBACK_TO_REG = True
SERVO_YOLO_FALLBACK_MISSES = 4    # switch after N consecutive misses
SERVO_POST_CONVERGE_STABILIZE_S = 1.0  # extra settle after convergence
                                       # (lets cable/wrist relax before
                                       # downstream phases)
# V15.10: small forward nudge before spiral search, ONLY if visual servo
# converged using REG mode (not YOLO). When REG mode converges, the wrist
# is positioned at the REGISTERED port location, but the registered pose
# may have a small systematic offset from the true port (PnP fusion error
# from the survey). A 4mm forward nudge shifts the wrist slightly toward
# the board so the spiral search starts from a position more likely to
# find the actual hole. YOLO-mode convergence skips this — YOLO already
# locked onto the live port image, no bias correction needed.
REG_PRE_SPIRAL_NUDGE_M       = 0.004   # 4mm forward (toward board)
REG_PRE_SPIRAL_NUDGE_SETTLE_S = 0.5    # settle after the nudge
# ACT mode
ACT_MODE = "skip"
# ACT (same as V9)
IMG_H = 224
IMG_W = 224
STATE_DIM  = 16
ACTION_DIM = 7
CHUNK_SIZE = 50
ACT_MAX_STEPS         = 650
ACT_STALL_CHECK_START = 230
Z_STALL_THRESHOLD     = 0.0006
Z_STALL_WINDOW        = 5
# Fallback spiral + descent
DESCENT_TRIGGER_Z     = 0.2299
INSERTION_TARGET_Z    = 0.06
INSERTION_STALL_STEPS = 5
INSERTION_STALL_MM    = 0.0002
# V15 (re-introduced): Z floor below which a stall counts as real.
# Above this Z, the wrist is still in the air and any stall is the
# cable pulling it back — we keep grinding down rather than calling
# it inserted and triggering spiral recovery prematurely.
# Tuned from the failing trial 2 logs that stalled at Z=0.2346 (real
# rim contact, below threshold) vs Z=0.2755 (cable tension yanking,
# above threshold).
PHASE2_STALL_MIN_Z    = 0.2499
# v12: ONLY for the descent that runs AFTER the spiral search finds a hole.
# The trial only ends when Z stalls AND Z < INSERTION_SUCCESS_Z. If Z stalls
# above this, we escalate downward pressure to push deeper.
# Has NO effect on the initial Phase 2 descent or wiggle descent.
INSERTION_SUCCESS_Z   = 0.1925
# v12: when post-spiral descent stalls ABOVE INSERTION_SUCCESS_Z, do a small
# back-and-forth XY wiggle while still pushing Z down. Tries to break the
# stall by jiggling the plug. No more pressure escalation.
WIGGLE_DESCENT_AMPLITUDE_M      = 0.001   # 2mm peak each side (4mm range)
WIGGLE_DESCENT_CYCLES           = 6        # how many complete back-forth cycles
WIGGLE_DESCENT_STEPS_PER_HALF   = 8        # steps per half-cycle (50ms each)
WIGGLE_DESCENT_Z_PUSH_PER_STEP  = 0.0003   # mm of Z descent commanded per step
# Same gap cap as forced-descent — prevents controller-reset oscillation
POST_SPIRAL_DESCENT_GAP_CAP_M   = 0.010    # 10mm max gap (commanded vs actual)
SPIRAL_MAX_RADIUS_M   = 0.015   # was 0.008 — wider safety net if homing slightly off
SPIRAL_STEP_NUDGE_M   = 0.0025   # v12: 3mm shift per "right" move (and 3mm per retract)
SPIRAL_N_TOTAL        = 6       # v12: spiral, retract→spiral, right→spiral, retract→spiral,
                                #      right→spiral, retract→spiral  (alternating moves)
# Stiffness/damping
STIFFNESS = [100.0, 0, 0, 0, 0, 0,
              0, 100.0, 0, 0, 0, 0,
              0, 0, 100.0, 0, 0, 0,
              0, 0, 0, 50.0, 0, 0,
              0, 0, 0, 0, 50.0, 0,
              0, 0, 0, 0, 0, 50.0]
DAMPING   = [40.0, 0, 0, 0, 0, 0,
              0, 40.0, 0, 0, 0, 0,
              0, 0, 40.0, 0, 0, 0,
              0, 0, 0, 15.0, 0, 0,
              0, 0, 0, 0, 15.0, 0,
              0, 0, 0, 0, 0, 15.0]
SOFT_STIFFNESS = [70.0, 0, 0, 0, 0, 0,
                   0, 70.0, 0, 0, 0, 0,
                   0, 0, 70.0, 0, 0, 0,
                   0, 0, 0, 20.0, 0, 0,
                   0, 0, 0, 0, 20.0, 0,
                   0, 0, 0, 0, 0, 20.0]
SOFT_DAMPING  = [30.0, 0, 0, 0, 0, 0,
                  0, 30.0, 0, 0, 0, 0,
                  0, 0, 30.0, 0, 0, 0,
                  0, 0, 0, 12.0, 0, 0,
                  0, 0, 0, 0, 12.0, 0,
                  0, 0, 0, 0, 0, 12.0]
# V15: descent XY-lock stiffness.
# During Phase 2 forced descent, we want the wrist's XY to stay
# absolutely fixed where servo converged — the cable should NOT
# be able to pull it sideways. So XY stiffness is 800 N/m (very
# stiff: at 10mm gap, 8 N resists any sideways tug). Z stays at
# 70 N/m for gentle push-down. When descent stalls, V15 recovery
# (lift + nudge forward) kicks in, then spiral takes over using
# the normal SOFT_STIFFNESS so it can actually sweep XY.
DESCENT_LOCK_STIFFNESS = [800.0, 0, 0, 0, 0, 0,
                           0, 800.0, 0, 0, 0, 0,
                           0, 0, 70.0, 0, 0, 0,
                           0, 0, 0, 50.0, 0, 0,
                           0, 0, 0, 0, 50.0, 0,
                           0, 0, 0, 0, 0, 50.0]
# Damping ratio at m≈3kg, k=800: c_crit = 2·√(3·800) ≈ 98 → 130
# is overdamped (1.33). For Z at 70, c_crit ≈ 29 → 30 is critical.
DESCENT_LOCK_DAMPING   = [130.0, 0, 0, 0, 0, 0,
                            0, 130.0, 0, 0, 0, 0,
                            0, 0, 30.0, 0, 0, 0,
                            0, 0, 0, 15.0, 0, 0,
                            0, 0, 0, 0, 15.0, 0,
                            0, 0, 0, 0, 0, 15.0]
CLASS_SFP = 0
CLASS_SC  = 1
def _make_corners_3d(w, h):
    """4 CCW corners in port frame, BL → BR → TR → TL."""
    return np.array([
        [-w/2, -h/2, 0.0],
        [+w/2, -h/2, 0.0],
        [+w/2, +h/2, 0.0],
        [-w/2, +h/2, 0.0],
    ], dtype=np.float64)
SFP_CORNERS_3D = _make_corners_3d(SFP_W, SFP_H)
SC_CORNERS_3D  = _make_corners_3d(SC_W,  SC_H)
# ── Geometry helpers ─────────────────────────────────────────────────────
def _quat_to_rot(qx, qy, qz, qw):
    return np.array([
        [1-2*qy*qy-2*qz*qz, 2*qx*qy-2*qz*qw,   2*qx*qz+2*qy*qw],
        [2*qx*qy+2*qz*qw,   1-2*qx*qx-2*qz*qz, 2*qy*qz-2*qx*qw],
        [2*qx*qz-2*qy*qw,   2*qy*qz+2*qx*qw,   1-2*qx*qx-2*qy*qy],
    ])
def _rot_to_quat(R):
    """Return (qx, qy, qz, qw)."""
    tr = R[0, 0] + R[1, 1] + R[2, 2]
    if tr > 0:
        s = 0.5 / math.sqrt(tr + 1.0)
        return ((R[2, 1] - R[1, 2]) * s,
                (R[0, 2] - R[2, 0]) * s,
                (R[1, 0] - R[0, 1]) * s,
                0.25 / s)
    elif R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
        s = 2.0 * math.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2])
        return (0.25 * s,
                (R[0, 1] + R[1, 0]) / s,
                (R[0, 2] + R[2, 0]) / s,
                (R[2, 1] - R[1, 2]) / s)
    elif R[1, 1] > R[2, 2]:
        s = 2.0 * math.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2])
        return ((R[0, 1] + R[1, 0]) / s,
                0.25 * s,
                (R[1, 2] + R[2, 1]) / s,
                (R[0, 2] - R[2, 0]) / s)
    else:
        s = 2.0 * math.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1])
        return ((R[0, 2] + R[2, 0]) / s,
                (R[1, 2] + R[2, 1]) / s,
                0.25 * s,
                (R[1, 0] - R[0, 1]) / s)
def _transform_to_T(t):
    """ROS Transform → 4x4."""
    T = np.eye(4)
    T[:3, :3] = _quat_to_rot(
        t.rotation.x, t.rotation.y, t.rotation.z, t.rotation.w)
    T[:3, 3] = [t.translation.x, t.translation.y, t.translation.z]
    return T
def _rot_to_yaw(R):
    return math.atan2(R[1, 0], R[0, 0])
def _yaw_to_rot(yaw):
    c, s = math.cos(yaw), math.sin(yaw)
    return np.array([[c, -s, 0.0],
                     [s,  c, 0.0],
                     [0.0, 0.0, 1.0]])
def _rpy_to_R(roll, pitch, yaw):
    cr, sr = math.cos(roll), math.sin(roll)
    cp, sp = math.cos(pitch), math.sin(pitch)
    cy, sy = math.cos(yaw), math.sin(yaw)
    Rx = np.array([[1.0, 0.0, 0.0],
                   [0.0,  cr, -sr],
                   [0.0,  sr,  cr]])
    Ry = np.array([[ cp, 0.0,  sp],
                   [0.0, 1.0, 0.0],
                   [-sp, 0.0,  cp]])
    Rz = np.array([[ cy, -sy, 0.0],
                   [ sy,  cy, 0.0],
                   [0.0, 0.0, 1.0]])
    return Rz @ Ry @ Rx
R_PLUG_IN_TCP = _rpy_to_R(*GRIPPER_OFFSET_RPY)
def _snap_yaw_near_pi(yaw):
    yaw = (yaw + math.pi) % (2 * math.pi) - math.pi
    if abs(yaw) < math.pi / 2:
        yaw = yaw + math.pi if yaw < 0 else yaw - math.pi
    return yaw
def _circular_median(angles, period=math.pi):
    if not angles:
        return 0.0
    scale = 2 * math.pi / period
    xs = np.array([math.cos(a * scale) for a in angles])
    ys = np.array([math.sin(a * scale) for a in angles])
    mx, my = float(np.median(xs)), float(np.median(ys))
    return math.atan2(my, mx) / scale
def _imgmsg_to_bgr(msg):
    if msg.encoding == "rgb8":
        arr = np.frombuffer(msg.data, dtype=np.uint8).reshape(
            msg.height, msg.width, 3)
        return cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)
    elif msg.encoding == "bgr8":
        return np.frombuffer(msg.data, dtype=np.uint8).reshape(
            msg.height, msg.width, 3).copy()
    raise ValueError(f"Unsupported encoding: {msg.encoding}")
def _solve_pnp_best_perm(corners_3d, corners_2d, K, dist=None):
    if dist is None:
        dist = np.zeros(5, dtype=np.float64)
    best = None
    for perm in range(4):
        rotated = np.roll(corners_2d, -perm, axis=0)
        try:
            ok, rvec, tvec = cv2.solvePnP(
                corners_3d, rotated, K, dist,
                flags=cv2.SOLVEPNP_ITERATIVE)
            if not ok:
                continue
            proj, _ = cv2.projectPoints(corners_3d, rvec, tvec, K, dist)
            err = float(np.linalg.norm(
                proj.reshape(-1, 2) - rotated, axis=1).mean())
            if best is None or err < best[2]:
                best = (rvec, tvec, err, perm)
        except cv2.error:
            continue
    return best
def _parse_port_meta(port_name: str):
    """Return (port_type, port_index)."""
    parts = port_name.lower().split("_")
    port_type = parts[0]
    port_index = 0
    for p in reversed(parts):
        if p.isdigit():
            port_index = int(p)
            break
    return port_type, port_index
# ── V13 NEW: Multi-NIC helpers ────────────────────────────────────────────
def _parse_module_idx(target_module_name):
    """Extract mount index from 'nic_card_mount_X'. Returns int or None."""
    if not target_module_name:
        return None
    try:
        return int(str(target_module_name).split("_")[-1])
    except (ValueError, AttributeError):
        return None
# Pre-compute rail-spacing bands: (rail_diff, d_min, d_max)
# d_min = rail_diff * 40mm (no translation difference)
# d_max = sqrt((rail_diff*40mm)^2 + (max translation diff)^2)
_RAIL_BANDS = [
    (rd,
     rd * RAIL_SPACING_M,
     math.sqrt((rd * RAIL_SPACING_M)**2 + RAIL_TRANSLATION_RANGE_M**2))
    for rd in (1, 2, 3, 4)
]
def _band_classify(world_dist, tol=RAIL_BAND_TOLERANCE_M):
    """Classify a world distance into a rail-index difference (1, 2, 3, 4).
    Returns int rail_diff or None if no band matches.
    """
    for rail_diff, d_min, d_max in _RAIL_BANDS:
        if (d_min - tol) <= world_dist <= (d_max + tol):
            return rail_diff
    return None
def _band_range(rail_diff):
    """Return (d_min, d_max) for a given rail-index difference, or None."""
    for r, d_min, d_max in _RAIL_BANDS:
        if r == rail_diff:
            return (d_min, d_max)
    return None
def _validate_pair_world_distance(port_a_world, port_b_world):
    """Check if 2 port positions (numpy 3-vectors) represent a real NIC pair.
    A real SFP NIC has port_0 and port_1 spaced ~14mm apart.
    Returns True if distance in [PAIR_PORT_DIST_MIN_M, PAIR_PORT_DIST_MAX_M].
    """
    d = float(np.linalg.norm(np.asarray(port_a_world) - np.asarray(port_b_world)))
    return PAIR_PORT_DIST_MIN_M <= d <= PAIR_PORT_DIST_MAX_M
# ── End V13 multi-NIC helpers ─────────────────────────────────────────────
def _img_msg_to_act_tensor(img_msg, device, mean, std, h=IMG_H, w=IMG_W):
    raw = np.frombuffer(img_msg.data, dtype=np.uint8).reshape(
        img_msg.height, img_msg.width, 3)
    raw = cv2.resize(raw, (w, h), interpolation=cv2.INTER_AREA)
    t = (torch.from_numpy(raw).permute(2, 0, 1)
         .float().div(255.0).unsqueeze(0).to(device))
    return (t - mean) / std
def _spiral_offsets(max_radius_m, n_turns=4, steps_per_turn=24):
    offsets = [(0.0, 0.0)]
    total = n_turns * steps_per_turn
    for i in range(1, total + 1):
        r = max_radius_m * (i / total)
        a = 2 * np.pi * i / steps_per_turn
        offsets.append((r * math.cos(a), r * math.sin(a)))
    return offsets
# ── Policy class ─────────────────────────────────────────────────────────
class MultiViewACTPolicy(Policy):
    def __init__(self, parent_node):
        super().__init__(parent_node)
        # v12: spawn-Z tracking across trials. Captured on first trial,
        # used at start of subsequent trials to lift TCP back to spawn altitude
        # if the previous trial left it descended.
        self.spawn_z = None
        # v14: registered target world position (from Phase 0.5 survey),
        # used by Phase 1.7 visual servo's D3 world-pose lock.
        # Reset to None at start of each trial.
        self._registered_target_world = None
        # V15.10: which mode the visual servo last CONVERGED in.
        # Set to "yolo" or "reg" at end of _visual_servo_align if it
        # returns True. None if it didn't converge. Used by Phase 3
        # recovery to decide whether to apply the small forward nudge
        # before spiral search (only triggers on REG-mode convergence).
        self._last_servo_mode = None
        # Heavy imports here so node startup discovery doesn't time out
        from lerobot.policies.act.modeling_act import ACTPolicy
        from lerobot.policies.act.configuration_act import ACTConfig
        from lerobot.configs.policies import FeatureType, PolicyFeature
        from ultralytics import YOLO
        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu")
        # ── Load ACT (same as V9) ──────────────────────────────────────
        self.get_logger().info(f"Loading ACT from: {DEFAULT_ACT_CHECKPOINT}")
        ckpt_dir = DEFAULT_ACT_CHECKPOINT
        stats = np.load(os.path.join(ckpt_dir, "norm_stats.npy"),
                        allow_pickle=True).item()
        self.state_mean  = torch.from_numpy(stats["state_mean"]).float().to(self.device)
        self.state_std   = torch.from_numpy(stats["state_std"]).float().to(self.device)
        self.action_mean = torch.from_numpy(stats["action_mean"]).float().to(self.device)
        self.action_std  = torch.from_numpy(stats["action_std"]).float().to(self.device)
        self.img_mean = torch.tensor(
            [0.485, 0.456, 0.406]).view(1, 3, 1, 1).to(self.device)
        self.img_std  = torch.tensor(
            [0.229, 0.224, 0.225]).view(1, 3, 1, 1).to(self.device)
        config = ACTConfig(
            n_obs_steps=1, chunk_size=CHUNK_SIZE, n_action_steps=CHUNK_SIZE,
            dim_model=512, n_heads=8, dim_feedforward=3200,
            n_encoder_layers=4, n_decoder_layers=1, n_vae_encoder_layers=4,
            use_vae=True, kl_weight=10.0,
            vision_backbone="resnet18", pretrained_backbone_weights=None,
            input_features={
                "observation.images.left":   PolicyFeature(type=FeatureType.VISUAL, shape=(3, IMG_H, IMG_W)),
                "observation.images.center": PolicyFeature(type=FeatureType.VISUAL, shape=(3, IMG_H, IMG_W)),
                "observation.images.right":  PolicyFeature(type=FeatureType.VISUAL, shape=(3, IMG_H, IMG_W)),
                "observation.state":         PolicyFeature(type=FeatureType.STATE,  shape=(STATE_DIM,)),
            },
            output_features={
                "action": PolicyFeature(type=FeatureType.ACTION, shape=(ACTION_DIM,)),
            },
        )
        self.act_policy = ACTPolicy(config)
        self.act_policy.load_state_dict(
            torch.load(os.path.join(ckpt_dir, "policy.pt"),
                       map_location=self.device))
        self.act_policy.eval()
        self.act_policy.to(self.device)
        n_params = sum(p.numel() for p in self.act_policy.parameters())
        self.get_logger().info(
            f"ACT loaded  params={n_params/1e6:.1f}M  device={self.device}")
        # ── Load YOLO ──────────────────────────────────────────────────
        self.get_logger().info(f"Loading YOLO from: {DEFAULT_YOLO_CHECKPOINT}")
        self.yolo = YOLO(DEFAULT_YOLO_CHECKPOINT)
        self.get_logger().info(
            f"YOLO loaded  task={self.yolo.task}  classes={self.yolo.names}")
        # Cache for per-camera intrinsics
        self._K_cache = {}
        # Live servo debug topic
        self._servo_debug_pub = parent_node.create_publisher(
            Image, SERVO_DEBUG_TOPIC, 10)
        self.get_logger().info(
            f"Servo debug publisher: {SERVO_DEBUG_TOPIC}")
        # Banner — confirm V11 features
        self.get_logger().info(
            "▰▰▰ MultiViewACTPolicy — multi-view perception + visual servoing ▰▰▰")
        self.get_logger().info(
            f"  DETECT_PAIR_DISTANCE_PX_MAX = {DETECT_PAIR_DISTANCE_PX_MAX} (was 200)")
        self.get_logger().info(
            f"  SCOUT_MAX_STEPS per direction = {SCOUT_MAX_STEPS} (was 6)")
        self.get_logger().info(
            f"  SCOUT_MIN_RECORDS = {SCOUT_MIN_RECORDS} (main detection = {DETECT_MIN_RECORDS})")
        self.get_logger().info(
            f"  SCOUT_CONF_THRESH = {SCOUT_CONF_THRESH} (main detection = {DETECT_CONF_THRESH})")
        self.get_logger().info(
            f"  SPIRAL_MAX_RADIUS_M = {SPIRAL_MAX_RADIUS_M} (was 0.008)")
        # v12: stabilization + spawn-Z + stepping spiral
        self.get_logger().info(
            f"  HOMING_SETTLE_TIME   = {HOMING_SETTLE_TIME}s (was 1.0s — more time post-homing)")
        self.get_logger().info(
            f"  POST_TILT_SETTLE_S   = {POST_TILT_SETTLE_S}s (NEW — settle after Phase 1.6 tilt)")
        self.get_logger().info(
            f"  POST_TRIAL_PAUSE     = {POST_TRIAL_PAUSE}s (was 2.0s — replaced by spawn-Z lift)")
        self.get_logger().info(
            f"  SPAWN_Z_RETURN_THRESHOLD_M = {SPAWN_Z_RETURN_THRESHOLD_M}  "
            f"(lift TCP if start Z >{SPAWN_Z_RETURN_THRESHOLD_M*1000:.0f}mm below spawn)")
        self.get_logger().info(
            f"  SPIRAL_N_TOTAL       = {SPIRAL_N_TOTAL}  "
            f"(spiral → retract → spiral → right → spiral → retract → ... interleaved)")
        self.get_logger().info(
            f"  SPIRAL_STEP_NUDGE_M  = {SPIRAL_STEP_NUDGE_M}  "
            f"({SPIRAL_STEP_NUDGE_M*1000:.0f}mm per retract and per right move)")
        # v12: descent success threshold + pressure escalation
        self.get_logger().info(
            f"  INSERTION_SUCCESS_Z  = {INSERTION_SUCCESS_Z}  "
            f"(POST-SPIRAL descent only: stall BELOW = ✓ inserted; "
            f"stall ABOVE = back-front wiggle while pushing down)")
        self.get_logger().info(
            f"  WIGGLE_DESCENT_AMPLITUDE_M = {WIGGLE_DESCENT_AMPLITUDE_M}  "
            f"({WIGGLE_DESCENT_CYCLES} cycles "
            f"× {WIGGLE_DESCENT_STEPS_PER_HALF * 2} steps/cycle, "
            f"Z drops {WIGGLE_DESCENT_Z_PUSH_PER_STEP*1000:.1f}mm/step)")
        if SET_PLUG_STRAIGHT_AFTER_HOMING:
            self.get_logger().info("  Plug-straight: ENABLED")
        if DEBUG_STOP_AFTER_PLUG_STRAIGHT:
            self.get_logger().info("  DEBUG MODE — stopping after plug-straight")
    # ── TF / TCP helpers ─────────────────────────────────────────────────
    def _lookup_T(self, target, source, stamp=None):
        try:
            time = stamp if stamp is not None else Time()
            tf = self._parent_node._tf_buffer.lookup_transform(
                target, source, time)
            return _transform_to_T(tf.transform)
        except TransformException:
            try:
                tf = self._parent_node._tf_buffer.lookup_transform(
                    target, source, Time())
                return _transform_to_T(tf.transform)
            except TransformException:
                return None
    def _get_tcp_pose(self):
        T = self._lookup_T("base_link", "gripper/tcp")
        if T is None:
            return None
        qx, qy, qz, qw = _rot_to_quat(T[:3, :3])
        return np.array([T[0, 3], T[1, 3], T[2, 3], qx, qy, qz, qw],
                        dtype=np.float32)
    def _get_tcp_z(self):
        T = self._lookup_T("base_link", "gripper/tcp")
        return None if T is None else T[2, 3]
    def _send_pose(self, move_robot, x, y, z, qx, qy, qz, qw,
                   stiffness=None, damping=None,
                   feedforward_force_z=0.0):
        """Send a Cartesian pose target. If feedforward_force_z is nonzero,
        add a downward force (negative Z in base frame) at the TCP.
        feedforward_force_z is in Newtons; pass a POSITIVE value to push
        DOWN (we negate internally).
        """
        msg = MotionUpdate()
        msg.header.frame_id = "base_link"
        msg.pose = Pose(
            position    = Point(x=float(x), y=float(y), z=float(z)),
            orientation = Quaternion(x=float(qx), y=float(qy),
                                     z=float(qz), w=float(qw)),
        )
        msg.target_stiffness = stiffness if stiffness else STIFFNESS
        msg.target_damping   = damping   if damping   else DAMPING
        if feedforward_force_z != 0.0:
            # Positive arg = push DOWN. Negate to set world-Z force.
            msg.feedforward_wrench_at_tip = Wrench(
                force=Vector3(x=0.0, y=0.0,
                              z=-float(feedforward_force_z)),
                torque=Vector3(x=0.0, y=0.0, z=0.0))
        msg.trajectory_generation_mode.mode = TrajectoryGenerationMode.MODE_POSITION
        move_robot(motion_update=msg)
    # ── Phase 0: Detection ───────────────────────────────────────────────
    def _camera_intrinsics(self, obs_msg, cam_name):
        if cam_name in self._K_cache:
            return self._K_cache[cam_name]
        info_attr = f"{cam_name}_camera_info"
        if not hasattr(obs_msg, info_attr):
            return None
        info = getattr(obs_msg, info_attr)
        if info.k is None or len(info.k) < 9:
            return None
        K = np.array(info.k, dtype=np.float64).reshape(3, 3)
        if K[0, 0] == 0.0:
            return None
        D_raw = info.d
        D = (np.array(D_raw, dtype=np.float64)
             if D_raw is not None and len(D_raw) > 0
             else np.zeros(5))
        self._K_cache[cam_name] = (K, D)
        return (K, D)
    def _yolo_detect(self, image_bgr, conf_threshold=None):
        conf = conf_threshold if conf_threshold is not None else DETECT_CONF_THRESH
        results = self.yolo.predict(
            image_bgr, conf=conf, verbose=False, imgsz=640)
        if not results:
            return []
        result = results[0]
        if result.obb is None or len(result.obb) == 0:
            return []
        corners_arr = result.obb.xyxyxyxy.cpu().numpy()
        cls_arr     = result.obb.cls.cpu().numpy().astype(int)
        conf_arr    = result.obb.conf.cpu().numpy()
        out = []
        for i in range(len(corners_arr)):
            out.append({
                "cls":     int(cls_arr[i]),
                "conf":    float(conf_arr[i]),
                "corners": corners_arr[i].astype(np.float64),
                "center":  corners_arr[i].mean(axis=0),
            })
        # v14.5: deduplicate. YOLO's OBB model frequently fires 2-3
        # overlapping boxes on the same physical port; without this, the
        # pair-forming logic creates phantom "pairs" between duplicate
        # detections (seen in trial 2: 5 pairs detected from 3 NICs).
        # Adjacent SFP ports on one NIC are ~45-50px apart (~14mm @ 30cm
        # cam distance); ports on adjacent NICs are 85+px apart. A 20px
        # threshold cleanly merges duplicates without merging real ports.
        out = self._deduplicate_detections(out, center_threshold_px=20.0)
        return out
    def _deduplicate_detections(self, dets, center_threshold_px=20.0):
        """v14.5: NMS-style dedup by center-distance + class.
        For each class independently, sort by confidence DESC and greedily
        keep detections whose centers are >= center_threshold_px from any
        already-kept detection of the same class.
        """
        if len(dets) <= 1:
            return dets
        by_class = {}
        for d in dets:
            by_class.setdefault(d["cls"], []).append(d)
        kept_total = []
        n_removed = 0
        for cls_id, group in by_class.items():
            group_sorted = sorted(group, key=lambda d: -d["conf"])
            kept_in_class = []
            for d in group_sorted:
                is_dup = False
                for k in kept_in_class:
                    dist = float(np.linalg.norm(
                        np.asarray(d["center"]) - np.asarray(k["center"])))
                    if dist < center_threshold_px:
                        is_dup = True
                        break
                if not is_dup:
                    kept_in_class.append(d)
                else:
                    n_removed += 1
            kept_total.extend(kept_in_class)
        return kept_total
    def _select_target_detection(self, detections, port_type,
                                 target_port_index,
                                 # ── V13 NEW optional params for multi-NIC ──
                                 target_module_name=None,
                                 current_tcp_pos=None,
                                 K=None, D=None, T_base_cam=None):
        """Pick the YOLO detection corresponding to the task target.
        For SFP: requires BOTH ports visible, sort by image-x:
          target_port_index=0 (port_0) → rightmost (higher x)
          target_port_index=1 (port_1) → leftmost  (lower  x)
        For SC: return highest-confidence SC detection.

        V13: If target_module_name, K, D, and T_base_cam are all provided AND
        multiple valid SFP pairs are found, uses world-frame geometry +
        rail-spacing bands to deterministically pick the pair belonging to
        the target NIC mount. Otherwise falls back to V12 single-pair logic.
        """
        cls_id = CLASS_SFP if port_type == "sfp" else CLASS_SC
        same_class = [d for d in detections if d["cls"] == cls_id]
        if not same_class:
            return None
        if port_type == "sc":
            return max(same_class, key=lambda d: d["conf"])
        # SFP: need a pair to identify port_0 vs port_1
        if len(same_class) < 2:
            return None
        # ── Find ALL valid pairs by image-pixel distance ─────────────────
        same_class.sort(key=lambda d: -d["conf"])
        top_k = min(MULTI_NIC_TOP_K_DETECTIONS, len(same_class))
        candidate_pairs = []
        for i in range(top_k):
            for j in range(i + 1, top_k):
                d_ij = math.hypot(
                    *(same_class[i]["center"] - same_class[j]["center"]))
                if d_ij < DETECT_PAIR_DISTANCE_PX_MAX:
                    candidate_pairs.append((same_class[i], same_class[j], d_ij))
        if not candidate_pairs:
            return None
        # ── Check if we can do multi-NIC disambiguation ──────────────────
        can_disambig = (target_module_name is not None
                        and K is not None and D is not None
                        and T_base_cam is not None)
        target_module_idx = (_parse_module_idx(target_module_name)
                             if can_disambig else None)
        can_disambig = can_disambig and (target_module_idx is not None)
        # ── Legacy single-NIC path (v12 behavior) ────────────────────────
        if not can_disambig:
            # Pick the smallest-pixel-distance pair (v12 behavior)
            candidate_pairs.sort(key=lambda x: x[2])
            a, b, _ = candidate_pairs[0]
            return self._apply_left_right_rule(a, b, target_port_index)
        # ── V13 multi-NIC path ────────────────────────────────────────────
        # Compute world data for each candidate pair, validate by port-spacing
        valid_pair_data = []
        for a, b, _ in candidate_pairs:
            pair_data = self._compute_pair_world_data(
                a, b, K, D, T_base_cam, SFP_CORNERS_3D)
            if pair_data is None:
                continue
            if not _validate_pair_world_distance(
                    pair_data["port_a_world"], pair_data["port_b_world"]):
                continue
            valid_pair_data.append(pair_data)
        if not valid_pair_data:
            return None
        if len(valid_pair_data) == 1:
            pd = valid_pair_data[0]
            return self._apply_left_right_rule(
                pd["det_a"], pd["det_b"], target_port_index)
        # Multiple valid pairs → disambiguate by rail-spacing geometry
        chosen_idx = self._disambiguate_multi_nic(
            valid_pair_data, target_module_idx, current_tcp_pos)
        if chosen_idx is None:
            # Couldn't decide — fall back to closest-to-TCP heuristic
            if current_tcp_pos is not None:
                tcp_dists = [float(np.linalg.norm(
                    p["midpoint_world"] - np.asarray(current_tcp_pos)))
                    for p in valid_pair_data]
                chosen_idx = int(np.argmin(tcp_dists))
            else:
                chosen_idx = 0
        chosen = valid_pair_data[chosen_idx]
        return self._apply_left_right_rule(
            chosen["det_a"], chosen["det_b"], target_port_index)
    # ── V13 helpers for multi-NIC disambiguation ──────────────────────────
    def _apply_left_right_rule(self, det_a, det_b, target_port_index):
        """Sort pair by image x: leftmost = port_1, rightmost = port_0."""
        if det_a["center"][0] < det_b["center"][0]:
            leftmost, rightmost = det_a, det_b
        else:
            leftmost, rightmost = det_b, det_a
        return rightmost if target_port_index == 0 else leftmost
    def _compute_pair_world_data(self, det_a, det_b, K, D, T_base_cam,
                                 corners_3d):
        """Run PnP on both ports of a candidate pair, transform to base_link.
        Returns dict with port_a_world, port_b_world, midpoint_world,
        det_a, det_b — or None if PnP fails for either port.
        """
        pnp_a = _solve_pnp_best_perm(corners_3d, det_a["corners"], K, D)
        pnp_b = _solve_pnp_best_perm(corners_3d, det_b["corners"], K, D)
        if pnp_a is None or pnp_b is None:
            return None
        rvec_a, tvec_a, _, _ = pnp_a
        rvec_b, tvec_b, _, _ = pnp_b
        R_a, _ = cv2.Rodrigues(rvec_a)
        T_cam_a = np.eye(4); T_cam_a[:3, :3] = R_a
        T_cam_a[:3, 3] = tvec_a.flatten()
        R_b, _ = cv2.Rodrigues(rvec_b)
        T_cam_b = np.eye(4); T_cam_b[:3, :3] = R_b
        T_cam_b[:3, 3] = tvec_b.flatten()
        T_base_a = T_base_cam @ T_cam_a
        T_base_b = T_base_cam @ T_cam_b
        pos_a = T_base_a[:3, 3]
        pos_b = T_base_b[:3, 3]
        return {
            "det_a":          det_a,
            "det_b":          det_b,
            "port_a_world":   pos_a,
            "port_b_world":   pos_b,
            "midpoint_world": 0.5 * (pos_a + pos_b),
        }
    def _disambiguate_multi_nic(self, pair_data_list, target_module_idx,
                                current_tcp_pos):
        """Enumerate target hypotheses across detected pairs, find which
        pair is consistent with target_module_idx + rail-spacing geometry.
        Returns: index in pair_data_list, or None if undecidable.
        """
        n = len(pair_data_list)
        if n == 0:
            return None
        if n == 1:
            return 0
        # Pairwise world distances between pair midpoints
        midpoints = [p["midpoint_world"] for p in pair_data_list]
        dist_matrix = np.zeros((n, n))
        for i in range(n):
            for j in range(i + 1, n):
                d = float(np.linalg.norm(midpoints[i] - midpoints[j]))
                dist_matrix[i, j] = d
                dist_matrix[j, i] = d
        # Log raw distances for debugging
        self.get_logger().info(
            f"Multi-NIC: {n} valid pairs detected. "
            f"target=mount_{target_module_idx}. "
            f"Pairwise distances: " + ", ".join(
                f"({i},{j})={dist_matrix[i,j]*1000:.1f}mm"
                for i in range(n) for j in range(i+1, n)))
        # Enumerate "this pair is the target" hypotheses
        consistent_candidates = []
        for hypo_target_idx in range(n):
            assignment = self._try_full_assignment(
                hypo_target_idx, target_module_idx, n, dist_matrix)
            if assignment is not None:
                consistent_candidates.append((hypo_target_idx, assignment))
                self.get_logger().info(
                    f"  hypothesis: pair[{hypo_target_idx}]=mount_{target_module_idx} "
                    f"→ CONSISTENT, full assignment: {assignment}")
            else:
                self.get_logger().info(
                    f"  hypothesis: pair[{hypo_target_idx}]=mount_{target_module_idx} "
                    f"→ inconsistent")
        if len(consistent_candidates) == 0:
            self.get_logger().warn(
                "Multi-NIC: NO consistent hypothesis. Cannot disambiguate.")
            return None
        if len(consistent_candidates) == 1:
            chosen = consistent_candidates[0][0]
            self.get_logger().info(
                f"Multi-NIC: UNIQUE assignment → pair[{chosen}] is target.")
            return chosen
        # Multiple consistent → tiebreak by TCP distance (closer wins)
        if current_tcp_pos is not None:
            tcp_pos = np.asarray(current_tcp_pos)
            tcp_dists = [(idx, float(np.linalg.norm(
                pair_data_list[idx]["midpoint_world"] - tcp_pos)))
                for idx, _ in consistent_candidates]
            tcp_dists.sort(key=lambda x: x[1])
            chosen = tcp_dists[0][0]
            self.get_logger().info(
                f"Multi-NIC: {len(consistent_candidates)} consistent hypotheses, "
                f"tiebreak by TCP distance. TCP-dists: "
                f"{[(i, f'{d*1000:.0f}mm') for i, d in tcp_dists]} "
                f"→ pair[{chosen}] is target.")
            return chosen
        # No TCP info — pick first consistent
        chosen = consistent_candidates[0][0]
        self.get_logger().info(
            f"Multi-NIC: {len(consistent_candidates)} consistent hypotheses, "
            f"no TCP for tiebreak. Defaulting to pair[{chosen}].")
        return chosen
    def _try_full_assignment(self, hypo_target_pair_idx, target_module_idx,
                              n_pairs, dist_matrix):
        """Given a hypothesis (one specific pair_idx is the target mount),
        try to assign all OTHER pairs to valid mount indices consistent
        with all pairwise distances. Uses backtracking.
        v14.4: returns the FIRST consistent assignment (for backward compat
        with the live servo's _select_target_detection). For the registry
        assignment, _try_full_assignments_all returns ALL solutions to allow
        downstream yaw filtering to disambiguate.
        Returns dict {pair_idx: mount_idx} if a full consistent assignment
        exists, else None.
        """
        all_sols = self._try_full_assignments_all(
            hypo_target_pair_idx, target_module_idx, n_pairs, dist_matrix)
        return all_sols[0] if all_sols else None
    def _try_full_assignments_all(self, hypo_target_pair_idx, target_module_idx,
                                    n_pairs, dist_matrix):
        """v14.4: enumerate ALL consistent assignments. Critical for the
        |Δi|=1 case where target=mount_M could pair with mount_{M-1} or
        mount_{M+1} — both are band-consistent, and only the yaw-direction
        filter can resolve which is physically present in the scene.
        Returns list of assignment dicts.
        """
        assignment = {hypo_target_pair_idx: target_module_idx}
        remaining = [i for i in range(n_pairs) if i != hypo_target_pair_idx]
        results = []
        self._assign_recursive_all(assignment, remaining, dist_matrix, results)
        return results
    def _assign_recursive(self, assignment, remaining, dist_matrix):
        """Recursive backtracking. Returns FIRST full assignment dict or None.
        Kept for backward compatibility with live servo."""
        if not remaining:
            return assignment
        pair_idx = remaining[0]
        rest = remaining[1:]
        used_mounts = set(assignment.values())
        candidate_mounts = [m for m in (0, 1, 2, 3, 4) if m not in used_mounts]
        for try_mount in candidate_mounts:
            ok = True
            for prev_pair_idx, prev_mount in assignment.items():
                rail_diff = abs(try_mount - prev_mount)
                if rail_diff == 0:
                    ok = False; break
                band = _band_range(rail_diff)
                if band is None:
                    ok = False; break
                d_actual = dist_matrix[pair_idx, prev_pair_idx]
                d_min, d_max = band
                tol = RAIL_BAND_TOLERANCE_M
                if not (d_min - tol <= d_actual <= d_max + tol):
                    ok = False; break
            if not ok:
                continue
            new_assignment = dict(assignment)
            new_assignment[pair_idx] = try_mount
            result = self._assign_recursive(new_assignment, rest, dist_matrix)
            if result is not None:
                return result
        return None
    def _assign_recursive_all(self, assignment, remaining, dist_matrix, out):
        """v14.4: collect ALL valid full assignments into `out`."""
        if not remaining:
            out.append(dict(assignment))
            return
        pair_idx = remaining[0]
        rest = remaining[1:]
        used_mounts = set(assignment.values())
        candidate_mounts = [m for m in (0, 1, 2, 3, 4) if m not in used_mounts]
        for try_mount in candidate_mounts:
            ok = True
            for prev_pair_idx, prev_mount in assignment.items():
                rail_diff = abs(try_mount - prev_mount)
                if rail_diff == 0:
                    ok = False; break
                band = _band_range(rail_diff)
                if band is None:
                    ok = False; break
                d_actual = dist_matrix[pair_idx, prev_pair_idx]
                d_min, d_max = band
                tol = RAIL_BAND_TOLERANCE_M
                if not (d_min - tol <= d_actual <= d_max + tol):
                    ok = False; break
            if not ok:
                continue
            new_assignment = dict(assignment)
            new_assignment[pair_idx] = try_mount
            self._assign_recursive_all(new_assignment, rest, dist_matrix, out)
    # ── End V13 multi-NIC helpers ─────────────────────────────────────────
    def _detect_port_pose(self, task, get_observation, send_feedback,
                          duration=None, quiet=False,
                          min_records=None, conf_threshold=None):
        """Phase 0 — collect detections, fuse, return port pose.
        Args:
          duration:        seconds (defaults to DETECT_DURATION_S)
          quiet:           suppress "Phase 0" log header
          min_records:     records needed to succeed (default DETECT_MIN_RECORDS=6)
          conf_threshold:  YOLO confidence (default DETECT_CONF_THRESH=0.75)
        Scout uses min_records=3, conf_threshold=0.60 (relaxed).
        Main detection uses defaults (strict).
        """
        port_type, target_port_index = _parse_port_meta(task.port_name)
        cls_id = CLASS_SFP if port_type == "sfp" else CLASS_SC
        corners_3d = SFP_CORNERS_3D if port_type == "sfp" else SC_CORNERS_3D
        det_duration = DETECT_DURATION_S if duration is None else duration
        min_req      = DETECT_MIN_RECORDS if min_records is None else min_records
        conf_req     = DETECT_CONF_THRESH if conf_threshold is None else conf_threshold
        if not quiet:
            self.get_logger().info(
                f"Phase 0: detecting {port_type} port_{target_port_index}, "
                f"collecting for {det_duration:.1f}s "
                f"(min_records={min_req}, conf={conf_req:.2f})")
            send_feedback(f"Detecting {port_type} port {target_port_index}")
        records = []
        cam_names = ("left", "center", "right")
        period_s = 1.0 / DETECT_RATE_HZ
        t_start = time.time()
        n_attempts = 0
        n_no_obs = 0
        n_no_target = 0
        while time.time() - t_start < det_duration:
            obs_msg = get_observation()
            if obs_msg is None:
                n_no_obs += 1
                self.sleep_for(period_s)
                continue
            for cam in cam_names:
                n_attempts += 1
                img_attr = f"{cam}_image"
                if not hasattr(obs_msg, img_attr):
                    continue
                try:
                    bgr = _imgmsg_to_bgr(getattr(obs_msg, img_attr))
                except Exception:
                    continue
                Kd = self._camera_intrinsics(obs_msg, cam)
                if Kd is None:
                    continue
                K, D = Kd
                # V13: look up T_base_cam BEFORE selection (multi-NIC needs it for PnP)
                T_base_cam = self._lookup_T(
                    "base_link", f"{cam}_camera/optical")
                if T_base_cam is None:
                    continue
                # V13: current TCP position for multi-NIC tiebreaker
                tcp_pose = self._get_tcp_pose()
                current_tcp_pos = tcp_pose[:3] if tcp_pose is not None else None
                # V13: target_module_name for multi-NIC disambiguation
                target_module_name = getattr(task, "target_module_name", None)
                detections = self._yolo_detect(bgr, conf_threshold=conf_req)
                target = self._select_target_detection(
                    detections, port_type, target_port_index,
                    target_module_name=target_module_name,
                    current_tcp_pos=current_tcp_pos,
                    K=K, D=D, T_base_cam=T_base_cam)
                if target is None:
                    n_no_target += 1
                    continue
                pnp = _solve_pnp_best_perm(
                    corners_3d, target["corners"], K, D)
                if pnp is None:
                    continue
                rvec, tvec, reproj_err, _ = pnp
                R_cp, _ = cv2.Rodrigues(rvec)
                T_cam_port = np.eye(4)
                T_cam_port[:3, :3] = R_cp
                T_cam_port[:3, 3]  = tvec.flatten()
                T_base_port = T_base_cam @ T_cam_port
                records.append({
                    "trans": T_base_port[:3, 3].copy(),
                    "yaw":   _snap_yaw_near_pi(_rot_to_yaw(T_base_port[:3, :3])),
                    "conf":  target["conf"],
                    "reproj": reproj_err,
                    "cam":   cam,
                })
            self.sleep_for(period_s)
        elapsed = time.time() - t_start
        if not quiet:
            self.get_logger().info(
                f"Phase 0 done in {elapsed:.1f}s — attempts={n_attempts} "
                f"no_target={n_no_target} no_obs={n_no_obs} records={len(records)}")
        if len(records) < min_req:
            if not quiet:
                self.get_logger().warn(
                    f"Phase 0: only {len(records)} records (<{min_req}) — "
                    f"detection FAILED, falling back to V9 ACT-from-home")
                send_feedback(f"Detection failed ({len(records)} records)")
            return None
        # ── Fusion ────────────────────────────────────────────────────
        trans_arr = np.array([r["trans"] for r in records])
        yaws      = [r["yaw"] for r in records]
        confs     = [r["conf"] for r in records]
        fused_trans = np.median(trans_arr, axis=0)
        fused_yaw = _circular_median(yaws, period=2 * math.pi)
        T_base_port = np.eye(4)
        T_base_port[:3, :3] = _yaw_to_rot(fused_yaw)
        T_base_port[:3, 3]  = fused_trans
        med_conf = float(np.median(confs))
        self.get_logger().info(
            f"Phase 0 FUSED: pos=({fused_trans[0]:+.4f}, {fused_trans[1]:+.4f}, "
            f"{fused_trans[2]:+.4f})  yaw={math.degrees(fused_yaw):+.2f}°  "
            f"n_records={len(records)}  med_conf={med_conf:.3f}")
        return {
            "T_base_port": T_base_port,
            "n_records":   len(records),
            "confidence":  med_conf,
        }
    # ── V14: Pre-homing survey + registry ────────────────────────────────
    def _detect_pair_burst(self, task, get_observation, duration_s):
        """Run YOLO + PnP for `duration_s` at current TCP pose.
        Returns: list of per-pair detection dicts with:
            port_0_world, port_1_world, midpoint_world,
            yaw, conf, cam, t_stamp
        Where port_0 = rightmost in image (per the convention used elsewhere)
        and port_1 = leftmost in image.
        """
        out = []
        t_start = time.time()
        period_s = 1.0 / DETECT_RATE_HZ
        cam_names = ("left", "center", "right")
        while time.time() - t_start < duration_s:
            obs_msg = get_observation()
            if obs_msg is None:
                self.sleep_for(period_s)
                continue
            for cam in cam_names:
                img_attr = f"{cam}_image"
                if not hasattr(obs_msg, img_attr):
                    continue
                try:
                    bgr = _imgmsg_to_bgr(getattr(obs_msg, img_attr))
                except Exception:
                    continue
                Kd = self._camera_intrinsics(obs_msg, cam)
                if Kd is None:
                    continue
                K, D = Kd
                T_base_cam = self._lookup_T(
                    "base_link", f"{cam}_camera/optical")
                if T_base_cam is None:
                    continue
                yolo_dets = self._yolo_detect(
                    bgr, conf_threshold=DETECT_CONF_THRESH)
                sfp_dets = [d for d in yolo_dets if d["cls"] == CLASS_SFP]
                if len(sfp_dets) < 2:
                    continue
                sfp_dets.sort(key=lambda d: -d["conf"])
                top_k = min(MULTI_NIC_TOP_K_DETECTIONS, len(sfp_dets))
                for i in range(top_k):
                    for j in range(i + 1, top_k):
                        d_ij = math.hypot(
                            *(sfp_dets[i]["center"] - sfp_dets[j]["center"]))
                        if d_ij >= DETECT_PAIR_DISTANCE_PX_MAX:
                            continue
                        pair = self._compute_pair_world_data(
                            sfp_dets[i], sfp_dets[j],
                            K, D, T_base_cam, SFP_CORNERS_3D)
                        if pair is None:
                            continue
                        if not _validate_pair_world_distance(
                                pair["port_a_world"], pair["port_b_world"]):
                            continue
                        # Identify port_0 (rightmost in image) vs port_1 (leftmost)
                        if pair["det_a"]["center"][0] >= pair["det_b"]["center"][0]:
                            port_0_world = pair["port_a_world"]
                            port_1_world = pair["port_b_world"]
                            det_p0, det_p1 = pair["det_a"], pair["det_b"]
                        else:
                            port_0_world = pair["port_b_world"]
                            port_1_world = pair["port_a_world"]
                            det_p0, det_p1 = pair["det_b"], pair["det_a"]
                        # Extract yaw from port_0's PnP rotation (R_base_port)
                        # Re-PnP to grab full pose for yaw
                        pnp_p0 = _solve_pnp_best_perm(
                            SFP_CORNERS_3D, det_p0["corners"], K, D)
                        yaw_world = None
                        if pnp_p0 is not None:
                            rvec_p0, tvec_p0, _, _ = pnp_p0
                            R_cp, _ = cv2.Rodrigues(rvec_p0)
                            T_cp = np.eye(4)
                            T_cp[:3, :3] = R_cp
                            T_cp[:3, 3]  = tvec_p0.flatten()
                            T_bp = T_base_cam @ T_cp
                            yaw_world = _snap_yaw_near_pi(
                                _rot_to_yaw(T_bp[:3, :3]))
                        out.append({
                            "port_0_world":   port_0_world,
                            "port_1_world":   port_1_world,
                            "midpoint_world": pair["midpoint_world"],
                            "yaw":            yaw_world,
                            "conf":           0.5 * (det_p0["conf"] + det_p1["conf"]),
                            "cam":            cam,
                            "t_stamp":        time.time() - t_start,
                        })
            self.sleep_for(period_s)
        return out
    def _survey_scene_square(self, task, get_observation, move_robot,
                              send_feedback):
        """V14.6: dispatch to square / forward / elevated / ushape mode.
        Kept the legacy method name `_survey_scene_square` so callers
        don't need to change. The actual path depends on SURVEY_MODE.
        """
        if SURVEY_MODE == "square":
            return self._survey_scene_4wp_square(
                task, get_observation, move_robot, send_feedback)
        elif SURVEY_MODE == "forward":
            return self._survey_scene_forward(
                task, get_observation, move_robot, send_feedback)
        elif SURVEY_MODE == "elevated":
            return self._survey_scene_elevated(
                task, get_observation, move_robot, send_feedback)
        else:  # "ushape"
            return self._survey_scene_ushape(
                task, get_observation, move_robot, send_feedback)

    def _survey_scene_4wp_square(self, task, get_observation, move_robot,
                                  send_feedback):
        """V15.8: 8-waypoint anti-clockwise RECTANGLE survey with midpoints.
        Path now visits BOTH corners AND the midpoint of each edge:
          spawn → front-mid (+Y H/2) → front (+Y H)
                → front-left-mid (-X W/2) → front-left (-X W)
                → back-left-mid (-Y H/2) → back-left (-Y H)
                → spawn-mid (+X W/2) → spawn-return (+X W, back to spawn)
        Same anti-clockwise traversal as before, just with intermediate
        observation stops on each leg. Doubles the world-frame detection
        coverage for the registry clustering — better at separating
        adjacent NICs and producing accurate per-port world poses.
        Time cost: ~2× the old 4-waypoint survey (8 wps × ~5s each).
        """
        spawn = self._get_tcp_pose()
        if spawn is None:
            self.get_logger().warn(
                "Survey (rectangle): cannot read TCP — aborting")
            return []
        cx, cy, cz = spawn[0], spawn[1], spawn[2]
        qx, qy, qz, qw = spawn[3], spawn[4], spawn[5], spawn[6]
        H = SURVEY_RECT_HEIGHT_M
        W = SURVEY_RECT_WIDTH_M
        Hm = H / 2.0   # +Y half-step (midpoint of vertical legs)
        Wm = W / 2.0   # -X half-step (midpoint of horizontal legs)
        # Anti-clockwise rectangle with midpoints inserted on every leg:
        #   spawn (cx, cy)
        #     → front-mid       (cx,       cy + Hm)
        #     → front           (cx,       cy + H)
        #     → front-left-mid  (cx - Wm,  cy + H)
        #     → front-left      (cx - W,   cy + H)
        #     → back-left-mid   (cx - W,   cy + Hm)
        #     → back-left       (cx - W,   cy)
        #     → spawn-mid       (cx - Wm,  cy)
        #     → spawn-return    (cx,       cy)
        waypoints = [
            ("front-mid",        cx,        cy + Hm,  cz),
            ("front",            cx,        cy + H,   cz),
            ("front-left-mid",   cx - Wm,   cy + H,   cz),
            ("front-left",       cx - W,    cy + H,   cz),
            ("back-left-mid",    cx - W,    cy + Hm,  cz),
            ("back-left",        cx - W,    cy,       cz),
            ("spawn-mid",        cx - Wm,   cy,       cz),
            ("spawn-return",     cx,        cy,       cz),
        ]
        n_wps = len(waypoints)
        self.get_logger().info(
            f"Phase 0.5 SURVEY: {n_wps}-waypoint anti-clockwise RECTANGLE "
            f"(corners + midpoints), "
            f"H(+Y)={H*1000:.0f}mm × W(-X)={W*1000:.0f}mm, "
            f"ends back at spawn, "
            f"~{(SURVEY_MOVE_SETTLE_S + SURVEY_DETECT_DURATION_S) * n_wps:.0f}s total. "
            f"All 3 cameras (left+center+right) feed YOLO + PnP.")
        send_feedback(f"Survey: rectangle+midpoints ({n_wps} waypoints)")
        all_detections = []
        for label, tx, ty, tz in waypoints:
            self.get_logger().info(
                f"  Survey waypoint '{label}': moving to "
                f"({tx:+.3f}, {ty:+.3f}, {tz:+.3f})")
            self._send_pose(move_robot, tx, ty, tz, qx, qy, qz, qw)
            self.sleep_for(SURVEY_MOVE_SETTLE_S)
            burst = self._detect_pair_burst(
                task, get_observation, SURVEY_DETECT_DURATION_S)
            for d_rec in burst:
                d_rec["waypoint"] = label
                all_detections.append(d_rec)
            self.get_logger().info(
                f"  Survey '{label}': {len(burst)} pair detections "
                f"(cumulative: {len(all_detections)})")
        return all_detections

    def _v15_pre_homing_approach(self, target_world_xy, move_robot,
                                 send_feedback):
        """V15: walk along the survey square edges to a corner near the
        target, THEN let Phase 1 homing take over.
        Reasoning: the survey already visited these 4 corner positions
        (spawn / front / front-left / back-left). The free cable end
        was DRAGGED through this path during the survey. Walking back
        through the same path to approach the target re-uses that
        already-settled cable trajectory and avoids the diagonal jump
        from spawn → target that pulled mount_0 into the cable's path
        when homing to mount_4.
        Logic:
          1. Compute the 4 square corners (same as survey waypoints).
          2. Pick the corner closest to the target NIC XY.
          3. If that corner is spawn (target is right next to spawn),
             skip — direct homing is fine.
          4. Otherwise walk in survey-order (spawn → front →
             front-left → back-left) until we reach the chosen corner.
             Stop there. Phase 1 homing handles the rest.
        Target_world_xy is the XY position of the target port (from
        Phase 0.8 TARGET FOUND).
        """
        if not V15_APPROACH_ENABLED:
            return
        tcp = self._get_tcp_pose()
        if tcp is None:
            self.get_logger().warn(
                "V15 approach: cannot read TCP — skipping")
            return
        cx, cy, cz = tcp[0], tcp[1], tcp[2]
        qx, qy, qz, qw = tcp[3], tcp[4], tcp[5], tcp[6]
        H = SURVEY_RECT_HEIGHT_M
        W = SURVEY_RECT_WIDTH_M
        tx, ty = float(target_world_xy[0]), float(target_world_xy[1])
        # V15.1: WALK ORDER is LEFT-FIRST (spawn → back-left →
        # front-left → front). V15.3: rectangle dims = H × W, must
        # match _survey_scene_4wp_square.
        corners = [
            ("spawn",       cx,       cy      ),   # idx 0 — start
            ("back-left",   cx - W,   cy      ),   # idx 1 — LEFT first
            ("front-left",  cx - W,   cy + H  ),   # idx 2 — then UP
            ("front",       cx,       cy + H  ),   # idx 3 — last
        ]
        # Compute distance from each corner to the target XY.
        # V15.3: the 'front' corner is on the RIGHT side of the
        # rectangle (same X as spawn). Excluding it from candidate
        # selection guarantees the approach always ends at a LEFT-side
        # corner — keeps the cable on the left, away from neighboring
        # NICs that typically sit at the back of the board.
        LEFT_SIDE_INDICES = {0, 1, 2}   # spawn / back-left / front-left
        best_idx = 0
        best_dist = float("inf")
        for i, (name, xx, yy) in enumerate(corners):
            dist = math.sqrt((xx - tx) ** 2 + (yy - ty) ** 2)
            marker = "" if i in LEFT_SIDE_INDICES else "  [right side — excluded from selection]"
            self.get_logger().info(
                f"V15 approach: corner '{name}' = ({xx:+.3f}, {yy:+.3f}) "
                f"→ target dist = {dist*100:.1f}cm{marker}")
            if i in LEFT_SIDE_INDICES and dist < best_dist:
                best_dist = dist
                best_idx = i
        best_name = corners[best_idx][0]
        # Skip if best corner is spawn — direct homing is best
        if best_idx == 0:
            self.get_logger().info(
                f"V15 approach: target closest to spawn corner "
                f"({best_dist*100:.1f}cm) — skipping detour, "
                f"direct homing")
            return
        # Skip if target is very close to spawn even if closest corner
        # isn't spawn (e.g., target between spawn and front)
        spawn_dist = math.sqrt((cx - tx) ** 2 + (cy - ty) ** 2)
        if spawn_dist < V15_APPROACH_SKIP_MARGIN_M:
            self.get_logger().info(
                f"V15 approach: target within "
                f"{V15_APPROACH_SKIP_MARGIN_M*1000:.0f}mm of spawn "
                f"({spawn_dist*100:.1f}cm) — skipping detour, "
                f"direct homing")
            return
        # V15.1: walk LEFT-first order from spawn to the chosen corner
        self.get_logger().info(
            f"V15 PRE-HOMING APPROACH: target XY=({tx:+.3f}, {ty:+.3f}), "
            f"closest corner = '{best_name}' "
            f"({best_dist*100:.1f}cm), walking LEFT-FIRST: spawn → "
            f"{' → '.join(corners[i][0] for i in range(1, best_idx + 1))} "
            f"({best_idx} waypoint{'s' if best_idx > 1 else ''})")
        send_feedback(f"V15 approach: walking left-first to '{best_name}'")
        # Move through corners 1..best_idx (skip corner 0 = spawn,
        # we're already there)
        for i in range(1, best_idx + 1):
            name, xx, yy = corners[i]
            self.get_logger().info(
                f"  V15 approach: → '{name}' at "
                f"({xx:+.3f}, {yy:+.3f}, {cz:+.3f})")
            self._send_pose(move_robot, xx, yy, cz, qx, qy, qz, qw)
            self.sleep_for(V15_APPROACH_SETTLE_S)
            actual_tcp = self._get_tcp_pose()
            if actual_tcp is not None:
                self.get_logger().info(
                    f"  V15 approach: arrived at '{name}' "
                    f"({actual_tcp[0]:+.3f}, {actual_tcp[1]:+.3f}, "
                    f"{actual_tcp[2]:+.3f})  — stabilizing "
                    f"{V15_APPROACH_SETTLE_S}s before next move")
        # V15.2: final pause AFTER reaching the chosen corner, BEFORE
        # Phase 1 homing starts. Cable's free end has been dragged
        # through 1-2 waypoints, gives it time to fully settle into
        # its new equilibrium before homing commits to the trajectory.
        self.get_logger().info(
            f"V15 PRE-HOMING APPROACH: complete at '{best_name}'. "
            f"Holding for {V15_APPROACH_FINAL_PAUSE_S}s final "
            f"stabilization before Phase 1 homing.")
        self.sleep_for(V15_APPROACH_FINAL_PAUSE_S)
        final_tcp = self._get_tcp_pose()
        if final_tcp is not None:
            self.get_logger().info(
                f"V15 approach: final stabilized TCP="
                f"({final_tcp[0]:+.3f}, {final_tcp[1]:+.3f}, "
                f"{final_tcp[2]:+.3f}) — Phase 1 will home from here.")

    def _survey_scene_forward(self, task, get_observation, move_robot,
                              send_feedback):
        """V14.6 SIMPLE: single forward step + 2-second detection.
        From spawn:
          1. Move forward (+Y) by SURVEY_FORWARD_DIST_M (12cm) at spawn Z
          2. Settle briefly (avoid motion-blurred frames in detection)
          3. Detect for SURVEY_FORWARD_DETECT_S using ALL 3 cameras
             (left + center + right, fused into single world-frame pool)
          4. NO descend, NO return — Phase 1 homing goes directly from
             this forward position to the target NIC.
        At 20Hz observation rate × 3 cameras × 2s detect, we run YOLO
        up to ~120 times in this single position, yielding many
        time-synchronized world-frame port detections to cluster.
        """
        spawn = self._get_tcp_pose()
        if spawn is None:
            self.get_logger().warn(
                "Survey (forward): cannot read TCP — aborting")
            return []
        cx, cy, cz = spawn[0], spawn[1], spawn[2]
        qx, qy, qz, qw = spawn[3], spawn[4], spawn[5], spawn[6]
        fwd_x = cx
        fwd_y = cy + SURVEY_FORWARD_DIST_M
        fwd_z = cz  # same altitude — no lift
        total_time = (SURVEY_FORWARD_SETTLE_S + SURVEY_FORWARD_DETECT_S)
        self.get_logger().info(
            f"Phase 0.5 SURVEY: FORWARD single-step, "
            f"step={SURVEY_FORWARD_DIST_M*1000:.0f}mm +Y at spawn altitude, "
            f"settle={SURVEY_FORWARD_SETTLE_S:.1f}s + "
            f"detect={SURVEY_FORWARD_DETECT_S:.1f}s, "
            f"~{total_time:.0f}s total. "
            f"All 3 cameras (left+center+right) feed YOLO + PnP.")
        send_feedback("Survey: forward step + detect")
        # ── Step 1: move forward ──
        self.get_logger().info(
            f"  [1/2] Moving forward: "
            f"({cx:+.3f}, {cy:+.3f}, {cz:+.3f}) → "
            f"({fwd_x:+.3f}, {fwd_y:+.3f}, {fwd_z:+.3f})")
        self._send_pose(move_robot, fwd_x, fwd_y, fwd_z, qx, qy, qz, qw)
        self.sleep_for(SURVEY_FORWARD_SETTLE_S)
        actual_tcp = self._get_tcp_pose()
        if actual_tcp is not None:
            self.get_logger().info(
                f"  Settled at: ({actual_tcp[0]:+.3f}, {actual_tcp[1]:+.3f}, "
                f"{actual_tcp[2]:+.3f})")
        # ── Step 2: detect (all 3 cameras inside _detect_pair_burst) ──
        self.get_logger().info(
            f"  [2/2] Detecting for {SURVEY_FORWARD_DETECT_S:.1f}s "
            f"using left+center+right cameras")
        burst = self._detect_pair_burst(
            task, get_observation, SURVEY_FORWARD_DETECT_S)
        for d_rec in burst:
            d_rec["waypoint"] = "forward"
        # Log camera breakdown
        from collections import Counter
        cam_counts = Counter(d.get("cam", "?") for d in burst)
        self.get_logger().info(
            f"Phase 0.5 SURVEY: complete — {len(burst)} pair detections. "
            f"By camera: {dict(cam_counts)}")
        return burst

    def _survey_scene_elevated(self, task, get_observation, move_robot,
                               send_feedback):
        """V14.6: ELEVATED bird's-eye survey with forward offset.
        Flow:
          1. Rise straight up by SURVEY_ELEVATED_RISE_M (15cm)
          2. Move FORWARD by SURVEY_ELEVATED_FORWARD_M (+Y, 10cm) so the
             camera FoV centers over the board area, not directly above
             spawn (where it would look past the NICs due to camera tilt)
          3. Do a SURVEY_CIRCLE_POINTS-point circle of radius
             SURVEY_CIRCLE_RADIUS_M at the elevated+forward position
          4. STAY at altitude — Phase 1 homing will descend directly to
             the target NIC pose (no return to spawn)
        Rationale: at altitude, camera FoV covers more ground; forward
        offset compensates for camera tilt; skipping the descend-then-home
        avoids unnecessary up-down-up motion.
        """
        spawn = self._get_tcp_pose()
        if spawn is None:
            self.get_logger().warn(
                "Survey (elevated): cannot read TCP — aborting")
            return []
        cx, cy, cz = spawn[0], spawn[1], spawn[2]
        qx, qy, qz, qw = spawn[3], spawn[4], spawn[5], spawn[6]
        rise_z = cz + SURVEY_ELEVATED_RISE_M
        center_x = cx
        center_y = cy + SURVEY_ELEVATED_FORWARD_M  # forward offset
        # Total time estimate
        circle_time = SURVEY_CIRCLE_POINTS * (
            SURVEY_CIRCLE_SETTLE_S + SURVEY_CIRCLE_DETECT_S)
        total_time = (SURVEY_ELEVATED_RISE_SETTLE_S
                      + SURVEY_ELEVATED_FORWARD_SETTLE_S
                      + circle_time)
        self.get_logger().info(
            f"Phase 0.5 SURVEY: ELEVATED bird's-eye + forward offset, "
            f"rise={SURVEY_ELEVATED_RISE_M*1000:.0f}mm to Z={rise_z:.3f}, "
            f"forward={SURVEY_ELEVATED_FORWARD_M*1000:.0f}mm in +Y, "
            f"circle={SURVEY_CIRCLE_POINTS} pts × "
            f"radius={SURVEY_CIRCLE_RADIUS_M*1000:.0f}mm × "
            f"{SURVEY_CIRCLE_DETECT_S:.1f}s detect, "
            f"~{total_time:.0f}s total (no descend — homing handles it)")
        send_feedback("Survey: elevated bird's-eye view")
        all_detections = []
        # ── Step 1: rise straight up ──
        self.get_logger().info(
            f"  [1/3] Rising to elevated Z={rise_z:.3f} "
            f"(spawn Z={cz:.3f}, rise={SURVEY_ELEVATED_RISE_M*1000:.0f}mm)")
        self._send_pose(move_robot, cx, cy, rise_z, qx, qy, qz, qw)
        self.sleep_for(SURVEY_ELEVATED_RISE_SETTLE_S)
        # ── Step 2: move forward at altitude ──
        self.get_logger().info(
            f"  [2/3] Moving forward at altitude to "
            f"({center_x:+.3f}, {center_y:+.3f}, {rise_z:.3f}) "
            f"(+Y by {SURVEY_ELEVATED_FORWARD_M*1000:.0f}mm)")
        self._send_pose(move_robot, center_x, center_y, rise_z,
                        qx, qy, qz, qw)
        self.sleep_for(SURVEY_ELEVATED_FORWARD_SETTLE_S)
        actual_tcp = self._get_tcp_pose()
        if actual_tcp is not None:
            self.get_logger().info(
                f"  Forward arrival: TCP=({actual_tcp[0]:+.3f}, "
                f"{actual_tcp[1]:+.3f}, {actual_tcp[2]:+.3f})")
        # ── Step 3: circle around the forward+elevated center ──
        self.get_logger().info(
            f"  [3/3] Circling at altitude around "
            f"({center_x:+.3f}, {center_y:+.3f}): "
            f"{SURVEY_CIRCLE_POINTS} pts, "
            f"radius={SURVEY_CIRCLE_RADIUS_M*1000:.0f}mm, "
            f"detect={SURVEY_CIRCLE_DETECT_S:.1f}s/pt")
        for i in range(SURVEY_CIRCLE_POINTS):
            angle = 2.0 * math.pi * i / SURVEY_CIRCLE_POINTS
            tx = center_x + SURVEY_CIRCLE_RADIUS_M * math.cos(angle)
            ty = center_y + SURVEY_CIRCLE_RADIUS_M * math.sin(angle)
            label = f"elevated-{i:02d}"
            self.get_logger().info(
                f"    Circle pt {i+1}/{SURVEY_CIRCLE_POINTS} '{label}': "
                f"angle={math.degrees(angle):6.1f}°, "
                f"pos=({tx:+.3f}, {ty:+.3f}, {rise_z:+.3f})")
            self._send_pose(move_robot, tx, ty, rise_z, qx, qy, qz, qw)
            self.sleep_for(SURVEY_CIRCLE_SETTLE_S)
            burst = self._detect_pair_burst(
                task, get_observation, SURVEY_CIRCLE_DETECT_S)
            for d_rec in burst:
                d_rec["waypoint"] = label
                all_detections.append(d_rec)
            self.get_logger().info(
                f"    Circle '{label}': {len(burst)} pair detections "
                f"(cumulative: {len(all_detections)})")
        # ── NO DESCEND ──
        # Phase 1 homing will move from the circle's last position
        # directly to the target NIC pose (lateral + Z descent in one move).
        final_tcp = self._get_tcp_pose()
        if final_tcp is not None:
            self.get_logger().info(
                f"Phase 0.5 SURVEY: complete — {len(all_detections)} total "
                f"detections from {SURVEY_CIRCLE_POINTS} elevated viewpoints. "
                f"Final TCP: ({final_tcp[0]:+.3f}, {final_tcp[1]:+.3f}, "
                f"{final_tcp[2]:+.3f}) — Phase 1 homing will descend from here.")
        return all_detections

    def _survey_scene_ushape(self, task, get_observation, move_robot,
                              send_feedback):
        """V14.6: 3-waypoint U-shape survey (legacy path).
        Path: spawn → -X (left) → +Y (front-left) → +X (front, END).
        Available by setting SURVEY_MODE = 'ushape'.
        """
        spawn = self._get_tcp_pose()
        if spawn is None:
            self.get_logger().warn(
                "Survey (ushape): cannot read TCP — aborting")
            return []
        cx, cy, cz = spawn[0], spawn[1], spawn[2]
        qx, qy, qz, qw = spawn[3], spawn[4], spawn[5], spawn[6]
        d = SURVEY_SQUARE_SIDE_M
        waypoints = [
            ("left",          cx - d,   cy,       cz),
            ("front-left",    cx - d,   cy + d,   cz),
            ("front",         cx,       cy + d,   cz),
        ]
        n_wps = len(waypoints)
        self.get_logger().info(
            f"Phase 0.5 SURVEY: U-shape (spawn→left→front-left→front), "
            f"step={d*1000:.0f}mm, {n_wps} waypoints, ends at "
            f"(spawn_x, spawn_y+{d*1000:.0f}mm), "
            f"~{(SURVEY_MOVE_SETTLE_S + SURVEY_DETECT_DURATION_S) * n_wps:.0f}s total")
        send_feedback(f"Survey: scanning board ({n_wps} viewpoints)")
        all_detections = []
        for label, tx, ty, tz in waypoints:
            self.get_logger().info(
                f"  Survey waypoint '{label}': moving to "
                f"({tx:+.3f}, {ty:+.3f}, {tz:+.3f})")
            self._send_pose(move_robot, tx, ty, tz, qx, qy, qz, qw)
            self.sleep_for(SURVEY_MOVE_SETTLE_S)
            burst = self._detect_pair_burst(
                task, get_observation, SURVEY_DETECT_DURATION_S)
            for d_rec in burst:
                d_rec["waypoint"] = label
                all_detections.append(d_rec)
            self.get_logger().info(
                f"  Survey '{label}': {len(burst)} pair detections "
                f"(cumulative: {len(all_detections)})")
        return all_detections
    def _build_nic_registry(self, all_detections):
        """Cluster survey detections into per-NIC entries.
        v14.2: XY-only clustering. PnP depth (Z) is noisy when the camera
        looks down at variable angles, so the SAME physical NIC can produce
        detections with very different Z values from different waypoints.
        We project to the board plane (XY) for clustering — Z is recovered
        as the median across the cluster.
        """
        if not all_detections:
            return []
        clusters = []
        for det in all_detections:
            mp_xy = det["midpoint_world"][:2]  # ignore Z
            assigned = False
            for cl in clusters:
                centroid_xy = np.mean(
                    [d["midpoint_world"][:2] for d in cl], axis=0)
                if float(np.linalg.norm(mp_xy - centroid_xy)) < SURVEY_CLUSTER_RADIUS_M:
                    cl.append(det)
                    assigned = True
                    break
            if not assigned:
                clusters.append([det])
        # Filter clusters with too few observations (likely YOLO noise)
        clusters = [c for c in clusters
                    if len(c) >= SURVEY_MIN_OBS_PER_NIC]
        registry = []
        for cl in clusters:
            port_0s = np.array([d["port_0_world"] for d in cl])
            port_1s = np.array([d["port_1_world"] for d in cl])
            mids    = np.array([d["midpoint_world"] for d in cl])
            yaws    = [d["yaw"] for d in cl if d["yaw"] is not None]
            confs   = [d["conf"] for d in cl]
            entry = {
                "port_0_world":   np.median(port_0s, axis=0),
                "port_1_world":   np.median(port_1s, axis=0),
                "midpoint_world": np.median(mids,    axis=0),
                "yaw_world":      (_circular_median(yaws, period=2*math.pi)
                                   if yaws else None),
                "n_obs":          len(cl),
                "conf_median":    float(np.median(confs)),
                "mount_idx":      None,  # assigned later
            }
            registry.append(entry)
        if SURVEY_REGISTRY_LOG_VERBOSE:
            self.get_logger().info(
                f"Phase 0.6 REGISTRY: {len(registry)} clusters (from "
                f"{len(all_detections)} raw detections, "
                f"min_obs={SURVEY_MIN_OBS_PER_NIC})")
            for i, e in enumerate(registry):
                mp = e["midpoint_world"]
                yaw_str = (f"{math.degrees(e['yaw_world']):+.1f}°"
                           if e["yaw_world"] is not None else "—")
                self.get_logger().info(
                    f"  registry[{i}]  midpoint=({mp[0]:+.4f}, "
                    f"{mp[1]:+.4f}, {mp[2]:+.4f})  yaw={yaw_str}  "
                    f"n_obs={e['n_obs']}  conf={e['conf_median']:.2f}")
        return registry
    def _assign_mount_indices_to_registry(self, registry, target_module_idx):
        """Assign mount_idx (0..4) to each registry entry using rail-spacing
        geometry. Reuses V13's hypothesis enumeration.
        Mutates registry in place.
        """
        n = len(registry)
        if n == 0:
            return registry
        if n == 1:
            # Only one NIC observed — it MUST be the target
            registry[0]["mount_idx"] = (target_module_idx
                                        if target_module_idx is not None
                                        else 0)
            self.get_logger().info(
                f"Phase 0.7 ASSIGN: single registry entry → "
                f"mount_{registry[0]['mount_idx']}")
            return registry
        if target_module_idx is None:
            self.get_logger().warn(
                "Phase 0.7 ASSIGN: target_module_idx is None, "
                "cannot assign indices")
            return registry
        # Build distance matrix between registry midpoints
        # v14.2: XY-only distance (matches clustering, robust to Z noise)
        dist_matrix = np.zeros((n, n))
        for i in range(n):
            for j in range(i + 1, n):
                d = float(np.linalg.norm(
                    registry[i]["midpoint_world"][:2]
                    - registry[j]["midpoint_world"][:2]))
                dist_matrix[i, j] = d
                dist_matrix[j, i] = d
        # Try each registry index as the target — find ALL consistent assignments
        # v14.4: enumerate ALL valid mount-index combos per hypothesis, not
        # just the first one. Necessary for trial 1 case (rails 3+4 spawned,
        # target=mount_3): both {reg[0]=3, reg[1]=2} and {reg[0]=3, reg[1]=4}
        # are band-consistent. Only the yaw filter can disambiguate.
        consistent_solutions = []
        for hypo_target_idx in range(n):
            all_assignments = self._try_full_assignments_all(
                hypo_target_idx, target_module_idx, n, dist_matrix)
            for assignment in all_assignments:
                consistent_solutions.append((hypo_target_idx, assignment))
        if not consistent_solutions:
            self.get_logger().warn(
                f"Phase 0.7 ASSIGN: NO consistent mount-index assignment "
                f"for target=mount_{target_module_idx} with {n} entries")
            return registry
        self.get_logger().info(
            f"Phase 0.7 ASSIGN: enumerated {len(consistent_solutions)} "
            f"consistent solution(s) before yaw filter")
        # v14.2: yaw-direction filter to disambiguate multiple solutions.
        # "Increasing mount_idx" direction in world = (sin(yaw), -cos(yaw), 0)
        # derived empirically from board geometry & port frame convention.
        # Eliminates solutions where mount_idx ordering disagrees with the
        # positional ordering along this direction.
        if len(consistent_solutions) > 1:
            yaws = [e["yaw_world"] for e in registry
                    if e["yaw_world"] is not None]
            if yaws:
                med_yaw = _circular_median(yaws, period=2 * math.pi)
                direction = np.array(
                    [math.sin(med_yaw), -math.cos(med_yaw), 0.0])
                self.get_logger().info(
                    f"Phase 0.7 ASSIGN: {len(consistent_solutions)} consistent "
                    f"solutions, filtering by yaw-direction (med_yaw="
                    f"{math.degrees(med_yaw):+.1f}°, dir=({direction[0]:+.3f}, "
                    f"{direction[1]:+.3f}))")
                positionally_valid = []
                for hypo_idx, sol in consistent_solutions:
                    # Sort registry indices in this solution by mount_idx,
                    # then verify the corresponding projections are monotonic.
                    items = sorted(sol.items(), key=lambda x: x[1])
                    projs = [float(np.dot(
                                 registry[reg_idx]["midpoint_world"],
                                 direction))
                             for reg_idx, _ in items]
                    monotonic = all(
                        projs[k] < projs[k + 1]
                        for k in range(len(projs) - 1))
                    self.get_logger().info(
                        f"  hypothesis {hypo_idx}: "
                        + ", ".join(f"reg[{i}]→mount_{m} (proj={ranks_proj:+.4f})"
                                    for (i, m), ranks_proj
                                    in zip(items, projs))
                        + f"  → {'VALID' if monotonic else 'rejected (non-monotonic)'}")
                    if monotonic:
                        positionally_valid.append((hypo_idx, sol))
                if positionally_valid:
                    consistent_solutions = positionally_valid
                else:
                    self.get_logger().warn(
                        "Phase 0.7 ASSIGN: no positionally-valid solution "
                        "(yaw-direction may be wrong) — falling back to TCP "
                        "distance for tiebreak")
        # If STILL multiple consistent (e.g., yaw filter passed all), pick
        # closest to current TCP as final tiebreak.
        # v14.6 NOTE: in the 3-NIC trial 2 case (rails 0+1+3, target=mount_1)
        # the yaw filter leaves 2 hypotheses that are geometrically
        # indistinguishable from rail-spacing data alone (one is just a
        # 1-rail "shift" of the other). TCP-distance picks the wrong one
        # in trial 2 but the right one in trial 1. We accept this as
        # an inherent limitation — fixing would require either active
        # probing or a board-edge feature anchor (not in our YOLO model).
        if len(consistent_solutions) > 1:
            tcp = self._get_tcp_pose()
            tcp_pos = tcp[:3] if tcp is not None else None
            if tcp_pos is not None:
                scores = [(idx, sol,
                           float(np.linalg.norm(
                               registry[idx]["midpoint_world"]
                               - np.asarray(tcp_pos))))
                          for idx, sol in consistent_solutions]
                scores.sort(key=lambda x: x[2])
                _, chosen_solution, _ = scores[0]
            else:
                _, chosen_solution = consistent_solutions[0]
        else:
            _, chosen_solution = consistent_solutions[0]
        # Apply the assignment
        for reg_idx, mount_idx in chosen_solution.items():
            registry[reg_idx]["mount_idx"] = mount_idx
        self.get_logger().info(
            f"Phase 0.7 ASSIGN: {len(consistent_solutions)} consistent "
            f"hypothesis(es). Assignment: " +
            ", ".join(f"reg[{i}]→mount_{m}"
                      for i, m in chosen_solution.items()))
        return registry
    def _find_target_in_registry(self, registry, target_module_idx):
        """Find the registry entry whose mount_idx matches the target.
        Returns the entry dict, or None if not found.
        """
        if target_module_idx is None:
            return None
        for entry in registry:
            if entry.get("mount_idx") == target_module_idx:
                return entry
        return None
    def _run_pre_homing_survey_and_find_target(
            self, task, get_observation, move_robot, send_feedback,
            target_module_idx, target_port_index):
        """Complete pre-homing pipeline: survey → cluster → assign → lookup.
        Returns: (target_entry, target_port_world, target_yaw_world) on success
                  or (None, None, None) on failure.
        """
        survey_dets = self._survey_scene_square(
            task, get_observation, move_robot, send_feedback)
        if not survey_dets:
            self.get_logger().warn(
                "Phase 0.5 SURVEY returned 0 detections — fail")
            return None, None, None
        registry = self._build_nic_registry(survey_dets)
        if not registry:
            self.get_logger().warn(
                "Phase 0.6 REGISTRY is empty after clustering — fail")
            return None, None, None
        registry = self._assign_mount_indices_to_registry(
            registry, target_module_idx)
        target_entry = self._find_target_in_registry(
            registry, target_module_idx)
        if target_entry is None:
            self.get_logger().warn(
                f"Phase 0.8 TARGET LOOKUP: mount_{target_module_idx} "
                f"NOT in registry. Falling back to closest-to-TCP.")
            # F3 fallback: closest registry entry to TCP
            tcp = self._get_tcp_pose()
            if tcp is not None and registry:
                tcp_pos = tcp[:3]
                dists = [(i, float(np.linalg.norm(
                    e["midpoint_world"] - np.asarray(tcp_pos))))
                    for i, e in enumerate(registry)]
                dists.sort(key=lambda x: x[1])
                target_entry = registry[dists[0][0]]
                self.get_logger().info(
                    f"  F3 fallback: picked registry[{dists[0][0]}] "
                    f"(distance to TCP: {dists[0][1]*1000:.1f}mm)")
        if target_entry is None:
            return None, None, None
        # Pick port_0 or port_1 based on target_port_index
        if target_port_index == 0:
            target_port_world = target_entry["port_0_world"]
        else:
            target_port_world = target_entry["port_1_world"]
        target_yaw = target_entry.get("yaw_world")
        self.get_logger().info(
            f"Phase 0.8 TARGET FOUND: port_{target_port_index} of "
            f"mount_{target_entry.get('mount_idx', '?')} at world "
            f"({target_port_world[0]:+.4f}, {target_port_world[1]:+.4f}, "
            f"{target_port_world[2]:+.4f})  yaw=" +
            (f"{math.degrees(target_yaw):+.2f}°"
             if target_yaw is not None else "—"))
        return target_entry, target_port_world, target_yaw
    # ── End V14 survey/registry methods ──────────────────────────────────
    # ── Phase 0.5: BIDIRECTIONAL Scout ───────────────────────────────────
    def _scout_in_direction(self, task, get_observation, move_robot,
                            send_feedback, direction_sign, label):
        """Walk gripper along gripper-local Y, running short detection bursts.
        Args:
          direction_sign: +1 for FORWARD (gripper-local -Y → cameras face this way)
                          -1 for BACKWARD (gripper-local +Y)
          label:          short tag for logs ("forward" / "backward")
        Returns detection dict (same format as _detect_port_pose) or None.
        Uses relaxed scout thresholds (SCOUT_MIN_RECORDS, SCOUT_CONF_THRESH).
        """
        tcp = self._get_tcp_pose()
        if tcp is None:
            self.get_logger().warn(f"Scout [{label}]: cannot read TCP — aborting")
            return None
        qx, qy, qz, qw = tcp[3], tcp[4], tcp[5], tcp[6]
        R = _quat_to_rot(qx, qy, qz, qw)
        # base_dir is "forward" (gripper-local -Y in base). Multiply by sign.
        base_dir = -R @ np.array([0.0, 1.0, 0.0])
        base_dir[2] = 0.0
        n = float(np.linalg.norm(base_dir))
        if n < 1e-6:
            self.get_logger().warn(
                f"Scout [{label}]: forward direction degenerate — defaulting to +Y base")
            base_dir = np.array([0.0, 1.0, 0.0])
        else:
            base_dir = base_dir / n
        step_dir = base_dir * direction_sign
        self.get_logger().info(
            f"Phase 0.5 [{label}]: step_dir in base = "
            f"({step_dir[0]:+.3f}, {step_dir[1]:+.3f}, {step_dir[2]:+.3f})")
        for step in range(1, SCOUT_MAX_STEPS + 1):
            tcp_now = self._get_tcp_pose()
            if tcp_now is None:
                continue
            new_x = tcp_now[0] + step_dir[0] * SCOUT_STEP_M
            new_y = tcp_now[1] + step_dir[1] * SCOUT_STEP_M
            self.get_logger().info(
                f"Scout [{label}] {step}/{SCOUT_MAX_STEPS}: TCP → "
                f"({new_x:+.4f}, {new_y:+.4f}, {tcp_now[2]:+.4f})")
            send_feedback(f"Scout {label} {step}/{SCOUT_MAX_STEPS}")
            self._send_pose(
                move_robot, new_x, new_y, tcp_now[2],
                qx, qy, qz, qw,
                stiffness=HOMING_STIFFNESS, damping=HOMING_DAMPING)
            self.sleep_for(SCOUT_STABILIZE_S)
            det = self._detect_port_pose(
                task, get_observation, send_feedback,
                duration=SCOUT_DURATION_S, quiet=True,
                min_records=SCOUT_MIN_RECORDS,
                conf_threshold=SCOUT_CONF_THRESH)
            if det is not None:
                self.get_logger().info(
                    f"✓ Scout [{label}] step {step}: port detected — proceeding")
                send_feedback(f"Port found at scout [{label}] step {step}")
                return det
            self.get_logger().info(
                f"  Scout [{label}] step {step}: not visible — continuing")
        return None
    def _scout_for_port(self, task, get_observation, move_robot, send_feedback):
        """Bidirectional scout: BACKWARD first, then FORWARD as fallback.
        Rationale: v10's forward-only scout failed T2 with 0.19m miss. The
        deterministic T2 randomization likely puts the board behind/below
        the camera at spawn. Going backward FIRST tests the new direction
        on the hard case before falling back to v10 behavior.
        """
        SCOUT_TOTAL_MM = SCOUT_MAX_STEPS * SCOUT_STEP_M * 1000
        self.get_logger().warn(
            f"Phase 0.5: BIDIRECTIONAL SCOUT — port not visible at spawn. "
            f"Trying BACKWARD ({SCOUT_TOTAL_MM:.0f}mm) first, then FORWARD.")
        send_feedback("Bidirectional scouting...")
        # Capture spawn TCP — we return here between directions
        spawn_tcp = self._get_tcp_pose()
        if spawn_tcp is None:
            self.get_logger().warn("Scout: cannot read spawn TCP — aborting")
            return None
        self.get_logger().info(
            f"Scout: spawn TCP=({spawn_tcp[0]:+.4f}, {spawn_tcp[1]:+.4f}, "
            f"{spawn_tcp[2]:+.4f})")
        # ── BACKWARD scan first ──────────────────────────────────────
        self.get_logger().info("Phase 0.5: starting BACKWARD scan (new direction)")
        det = self._scout_in_direction(
            task, get_observation, move_robot, send_feedback,
            direction_sign=-1, label="backward")
        if det is not None:
            self.get_logger().info("✓ BACKWARD scout succeeded — skipping forward")
            return det
        # Backward failed → return to spawn
        self.get_logger().info(
            "Backward failed — returning to spawn before forward scan")
        send_feedback("Returning to spawn...")
        self._send_pose(
            move_robot,
            spawn_tcp[0], spawn_tcp[1], spawn_tcp[2],
            spawn_tcp[3], spawn_tcp[4], spawn_tcp[5], spawn_tcp[6],
            stiffness=HOMING_STIFFNESS, damping=HOMING_DAMPING)
        self.sleep_for(SCOUT_STABILIZE_S * 2)   # extra time to settle at spawn
        # ── FORWARD scan fallback (v10 behavior) ─────────────────────
        self.get_logger().info("Phase 0.5: starting FORWARD scan (v10 fallback)")
        det = self._scout_in_direction(
            task, get_observation, move_robot, send_feedback,
            direction_sign=+1, label="forward")
        if det is not None:
            self.get_logger().info("✓ FORWARD scout succeeded")
            return det
        self.get_logger().warn(
            f"Bidirectional scout exhausted ({2*SCOUT_MAX_STEPS} steps total) — "
            f"V9 ACT fallback")
        return None
    # ── Phase 1.7: Visual servo alignment ────────────────────────────────
    def _publish_servo_debug(self, bgr, iteration, port_type, target,
                             ref_pixel, err_mag, status):
        try:
            dbg = bgr.copy()
            ref_u, ref_v = int(ref_pixel[0]), int(ref_pixel[1])
            if target is not None:
                corners = np.asarray(target["corners"], dtype=np.int32).reshape(-1, 1, 2)
                cv2.polylines(dbg, [corners], True, (255, 255, 0), 2)
                port_u = int(target["center"][0])
                port_v = int(target["center"][1])
                cv2.circle(dbg, (port_u, port_v), 8, (0, 255, 0), -1)
                cv2.circle(dbg, (port_u, port_v), 10, (0, 0, 0), 2)
                cv2.arrowedLine(dbg, (port_u, port_v), (ref_u, ref_v),
                                (0, 255, 255), 2, tipLength=0.15)
            cv2.circle(dbg, (ref_u, ref_v), 14, (0, 0, 255), 2)
            cv2.line(dbg, (ref_u - 22, ref_v), (ref_u + 22, ref_v),
                     (0, 0, 255), 2)
            cv2.line(dbg, (ref_u, ref_v - 22), (ref_u, ref_v + 22),
                     (0, 0, 255), 2)
            cv2.circle(dbg, (ref_u, ref_v), 3, (0, 0, 255), -1)
            font = cv2.FONT_HERSHEY_SIMPLEX
            cv2.putText(dbg, f"iter {iteration:02d}  {status}",
                        (20, 40), font, 0.9, (0, 0, 0), 5)
            cv2.putText(dbg, f"iter {iteration:02d}  {status}",
                        (20, 40), font, 0.9, (255, 255, 255), 2)
            err_str = f"err={err_mag:.1f}px" if err_mag != float("inf") else "err=N/A"
            cv2.putText(dbg, f"{port_type}  {err_str}",
                        (20, 75), font, 0.7, (0, 0, 0), 4)
            cv2.putText(dbg, f"{port_type}  {err_str}",
                        (20, 75), font, 0.7, (255, 255, 255), 2)
            if target is not None:
                cv2.putText(dbg, f"conf={target['conf']:.2f}",
                            (20, 105), font, 0.6, (0, 0, 0), 3)
                cv2.putText(dbg, f"conf={target['conf']:.2f}",
                            (20, 105), font, 0.6, (255, 255, 255), 1)
            msg = Image()
            msg.header.stamp = self._parent_node.get_clock().now().to_msg()
            msg.header.frame_id = f"{SERVO_CAMERA}_camera"
            msg.height = int(dbg.shape[0])
            msg.width  = int(dbg.shape[1])
            msg.encoding = "bgr8"
            msg.is_bigendian = 0
            msg.step = int(dbg.shape[1] * 3)
            msg.data = dbg.tobytes()
            self._servo_debug_pub.publish(msg)
        except Exception as e:
            self.get_logger().warn(f"Servo debug publish failed: {e}")
    def _visual_servo_align(self, get_observation, move_robot,
                            port_type, target_port_index, send_feedback,
                            ref_pixel=None, stage_name="",
                            target_module_name=None,
                            # V14 NEW: world-pose lock (D3) — reject YOLO
                            # detections more than SERVO_WORLD_LOCK_RADIUS_M
                            # away from this stored pose
                            registered_target_world=None):
        if ref_pixel is None:
            ref_pixel = SERVO_STAGE1_REF_PIXEL
        ref_u, ref_v = float(ref_pixel[0]), float(ref_pixel[1])
        corners_3d = SFP_CORNERS_3D if port_type == "sfp" else SC_CORNERS_3D
        stage_label = stage_name if stage_name else "visual servo"
        send_feedback(f"Phase 1.7: {stage_label}")
        self.get_logger().info(
            f"Phase 1.7 [{stage_label}]: ref=({ref_u:.0f}, {ref_v:.0f})  "
            f"tol=yolo:{SERVO_PIXEL_TOLERANCE}px / reg:{SERVO_PIXEL_TOLERANCE_REG}px  "
            f"max_iter={SERVO_MAX_ITERATIONS}")
        # V18: initialize servo_mode HERE (before any return path), so every
        # bail records the mode the servo was in. Phase 3 recovery uses this
        # to pick spiral pattern + nudge — regardless of whether the servo
        # converged. The user's rule: "if it used yolo that spiral approach,
        # if it used reg then front nudge and that spiral approach".
        servo_mode = "yolo"
        tcp_init = self._get_tcp_pose()
        if tcp_init is None:
            self.get_logger().warn(
                f"Servo {stage_label}: cannot read initial TCP — abort")
            # V18: even though we never started, leave mode as "yolo"
            # (default). Downstream Phase 3 will use YOLO spiral pattern.
            self._last_servo_mode = servo_mode
            return False
        locked_z  = tcp_init[2]
        locked_qx = tcp_init[3]
        locked_qy = tcp_init[4]
        locked_qz = tcp_init[5]
        locked_qw = tcp_init[6]
        self.get_logger().info(
            f"  locked Z={locked_z:.4f}  "
            f"locked quat=({locked_qx:+.3f}, {locked_qy:+.3f}, "
            f"{locked_qz:+.3f}, {locked_qw:+.3f})")
        last_err_mag = float("inf")
        converged_frames = 0
        no_detect_count = 0
        # V15: hybrid servo. Start in "yolo" mode (live detection).
        # After SERVO_YOLO_FALLBACK_MISSES consecutive misses, switch
        # to "reg" mode (project registered_target_world). Once in
        # reg mode, stay there for the rest of the trial (no flapping
        # back and forth).
        # (servo_mode is initialized above, before tcp_init check)
        for iteration in range(SERVO_MAX_ITERATIONS):
            obs = get_observation()
            if obs is None:
                self.sleep_for(0.1)
                continue
            img_attr = f"{SERVO_CAMERA}_image"
            if not hasattr(obs, img_attr):
                self.get_logger().warn(
                    f"Servo: obs missing {img_attr} — abort")
                # V18: record current mode before bailing
                self._last_servo_mode = servo_mode
                return False
            try:
                bgr = _imgmsg_to_bgr(getattr(obs, img_attr))
            except Exception as e:
                self.get_logger().warn(f"Servo: image decode failed: {e}")
                continue
            Kd = self._camera_intrinsics(obs, SERVO_CAMERA)
            if Kd is None:
                continue
            K, D = Kd
            fx, fy = K[0, 0], K[1, 1]
            T_base_cam = self._lookup_T(
                "base_link", f"{SERVO_CAMERA}_camera/optical")
            if T_base_cam is None:
                self.sleep_for(0.1)
                continue
            tcp_pose_iter = self._get_tcp_pose()
            current_tcp_pos = (tcp_pose_iter[:3]
                               if tcp_pose_iter is not None else None)
            # ── Per-iteration mode resolution ──
            target = None
            port_u = None
            port_v = None
            Z_cam = None
            if servo_mode == "yolo":
                detections = self._yolo_detect(
                    bgr, conf_threshold=SERVO_CONF_THRESH)
                target = self._select_target_detection(
                    detections, port_type, target_port_index,
                    target_module_name=target_module_name,
                    current_tcp_pos=current_tcp_pos,
                    K=K, D=D, T_base_cam=T_base_cam)
                # V14 D3: world-pose lock — reject detection if its PnP
                # world position is too far from the registered target.
                if (target is not None
                        and SERVO_WORLD_LOCK_ENABLED
                        and registered_target_world is not None):
                    pnp = _solve_pnp_best_perm(
                        SFP_CORNERS_3D if port_type == "sfp" else SC_CORNERS_3D,
                        target["corners"], K, D)
                    if pnp is not None:
                        rvec, tvec, _, _ = pnp
                        R_cp, _ = cv2.Rodrigues(rvec)
                        T_cp = np.eye(4)
                        T_cp[:3, :3] = R_cp
                        T_cp[:3, 3]  = tvec.flatten()
                        T_bp = T_base_cam @ T_cp
                        pos_world = T_bp[:3, 3]
                        err_m = float(np.linalg.norm(
                            pos_world - np.asarray(registered_target_world)))
                        if err_m > SERVO_WORLD_LOCK_RADIUS_M:
                            if iteration % 5 == 0:
                                self.get_logger().info(
                                    f"  iter {iteration}: detection rejected by "
                                    f"world-pose lock ({err_m*1000:.1f}mm > "
                                    f"{SERVO_WORLD_LOCK_RADIUS_M*1000:.0f}mm)")
                            target = None
                if target is None:
                    # YOLO miss path
                    no_detect_count += 1
                    if iteration % 5 == 0:
                        self.get_logger().info(
                            f"  iter {iteration} [yolo]: no detection "
                            f"({no_detect_count} consecutive misses)")
                    self._publish_servo_debug(
                        bgr, iteration, port_type, None,
                        (ref_u, ref_v), float("inf"), "NO_DETECT")
                    # V15: switch to registered-pose mode if available
                    if (SERVO_YOLO_FALLBACK_TO_REG
                            and no_detect_count >= SERVO_YOLO_FALLBACK_MISSES
                            and registered_target_world is not None):
                        self.get_logger().warn(
                            f"V15 SERVO: YOLO lost port "
                            f"{no_detect_count} consecutive iters — "
                            f"FALLING BACK TO REGISTERED POSE for "
                            f"the rest of the servo. Reg target = "
                            f"({registered_target_world[0]:+.4f}, "
                            f"{registered_target_world[1]:+.4f}, "
                            f"{registered_target_world[2]:+.4f})")
                        servo_mode = "reg"
                        no_detect_count = 0
                        # fall through below — this iteration runs as reg
                    elif no_detect_count > 15:
                        # No reg pose to fall back to, and YOLO is dead
                        self.get_logger().warn(
                            f"Servo: persistent no-detect after "
                            f"{no_detect_count} misses and no "
                            f"registered-pose fallback — bailing")
                        # V18: record current mode (yolo, since no reg fallback)
                        self._last_servo_mode = servo_mode
                        return False
                    else:
                        self.sleep_for(0.1)
                        continue
                else:
                    # YOLO target accepted — extract pixel + depth
                    no_detect_count = 0
                    port_u = float(target["center"][0])
                    port_v = float(target["center"][1])
                    pnp = _solve_pnp_best_perm(
                        corners_3d, target["corners"], K, D)
                    if pnp is None:
                        self.sleep_for(0.1)
                        continue
                    rvec, tvec, _, _ = pnp
                    Z_cam = float(tvec.flatten()[2])
                    if Z_cam <= 0.02:
                        self.get_logger().warn(
                            f"Servo: bad PnP depth Z={Z_cam:.3f} — skip")
                        continue
            # If we just switched to reg mode this iteration (or were
            # already in reg mode), do the projection.
            if servo_mode == "reg":
                T_cam_base = np.linalg.inv(T_base_cam)
                port_world_h = np.array([
                    float(registered_target_world[0]),
                    float(registered_target_world[1]),
                    float(registered_target_world[2]),
                    1.0,
                ])
                port_in_cam = T_cam_base @ port_world_h
                Z_cam = float(port_in_cam[2])
                if Z_cam <= 0.02:
                    self.get_logger().warn(
                        f"Servo (reg): bad projection Z={Z_cam:.3f} — skip")
                    self.sleep_for(0.1)
                    continue
                port_u = float(K[0, 0] * port_in_cam[0] / Z_cam + K[0, 2])
                port_v = float(K[1, 1] * port_in_cam[1] / Z_cam + K[1, 2])
                target = None  # debug marker only
            # ── Common: pixel error → delta → move ──
            err_u = port_u - ref_u
            err_v = port_v - ref_v
            err_mag = math.hypot(err_u, err_v)
            # V15.7: REG mode uses tighter tolerance (deterministic
            # projection, no per-frame YOLO noise). YOLO mode keeps the
            # original loose threshold (live detection is noisier).
            current_tolerance = (SERVO_PIXEL_TOLERANCE_REG
                                 if servo_mode == "reg"
                                 else SERVO_PIXEL_TOLERANCE)
            if iteration % 3 == 0 or err_mag < current_tolerance:
                self.get_logger().info(
                    f"  iter {iteration} [{servo_mode}]: "
                    f"port=({port_u:.0f}, {port_v:.0f})  "
                    f"err=({err_u:+.0f}, {err_v:+.0f})  mag={err_mag:.1f}px  "
                    f"tol={current_tolerance}px")
            if err_mag < current_tolerance:
                converged_frames += 1
                if converged_frames >= SERVO_CONVERGE_FRAMES:
                    self.get_logger().info(
                        f"✓ Servo CONVERGED at iter {iteration} "
                        f"(mode={servo_mode}, err={err_mag:.1f}px, "
                        f"tol={current_tolerance}px). "
                        f"Stabilizing "
                        f"{SERVO_POST_CONVERGE_STABILIZE_S}s before "
                        f"downstream phases.")
                    self._publish_servo_debug(
                        bgr, iteration, port_type, target,
                        (ref_u, ref_v), err_mag, "CONVERGED")
                    # V18: record the servo mode regardless of converge
                    # outcome — same field, same downstream logic. The
                    # mode here is whatever the servo was in (yolo or
                    # reg via fallback). Phase 3 recovery uses it for
                    # spiral pattern selection and nudge trigger.
                    self._last_servo_mode = servo_mode
                    # V15: extra post-convergence stabilization
                    self.sleep_for(SERVO_POST_CONVERGE_STABILIZE_S)
                    return True
                self._publish_servo_debug(
                    bgr, iteration, port_type, target,
                    (ref_u, ref_v), err_mag, "near")
                self.sleep_for(SERVO_SETTLE_S)
                continue
            else:
                converged_frames = 0
            self._publish_servo_debug(
                bgr, iteration, port_type, target,
                (ref_u, ref_v), err_mag, "step")
            dX_cam = err_u * Z_cam / fx
            dY_cam = err_v * Z_cam / fy
            dX_cam *= SERVO_GAIN
            dY_cam *= SERVO_GAIN
            R_base_cam = T_base_cam[:3, :3]
            delta_base = R_base_cam @ np.array([dX_cam, dY_cam, 0.0])
            delta_base[2] = 0.0
            delta_mag = float(np.linalg.norm(delta_base))
            if delta_mag > SERVO_MAX_STEP_M:
                delta_base *= (SERVO_MAX_STEP_M / delta_mag)
                self.get_logger().info(
                    f"  iter {iteration}: clamping step "
                    f"{delta_mag*1000:.1f}mm → {SERVO_MAX_STEP_M*1000:.0f}mm")
            tcp = self._get_tcp_pose()
            if tcp is None:
                continue
            new_x = tcp[0] + delta_base[0]
            new_y = tcp[1] + delta_base[1]
            self._send_pose(
                move_robot, new_x, new_y, locked_z,
                locked_qx, locked_qy, locked_qz, locked_qw,
                stiffness=SERVO_STIFFNESS, damping=SERVO_DAMPING)
            last_err_mag = err_mag
            self.sleep_for(SERVO_SETTLE_S)
        self.get_logger().warn(
            f"Servo did NOT converge in {SERVO_MAX_ITERATIONS} iterations "
            f"(mode={servo_mode}, last err={last_err_mag:.1f}px) — "
            f"proceeding anyway")
        # V18: record the servo's final mode regardless of convergence.
        # Phase 3 recovery picks spiral pattern + nudge based on this mode,
        # not on whether convergence happened. If servo_mode is "reg", the
        # 4mm forward nudge fires and the REG-first spiral pattern runs.
        # If "yolo", the YOLO-first spiral pattern runs with no nudge.
        self._last_servo_mode = servo_mode
        return False
    # ── Phase 0.1 (v12): Spawn-Z return between trials ──────────────────
    def _maybe_return_to_spawn_z(self, move_robot):
        """v14.6: Robust spawn-Z lift with retries + verification.
        Trial 3 in v14.5 showed the lift command completing without the
        robot actually moving (Z went 0.2263 → 0.2266) — likely because
        the cable was still snagged in the previous trial's port.
        Now uses HIGHER stiffness, LONGER settle, MULTIPLE attempts,
        and Z verification on each attempt.
        """
        tcp = self._get_tcp_pose()
        if tcp is None:
            self.get_logger().warn("Spawn-Z check: cannot read TCP — skipping")
            return
        cx, cy, cz = tcp[0], tcp[1], tcp[2]
        qx, qy, qz, qw = tcp[3], tcp[4], tcp[5], tcp[6]
        if self.spawn_z is None:
            self.spawn_z = cz
            self.get_logger().info(
                f"━━━ SPAWN Z CAPTURED ━━━\n"
                f"  spawn_z = {self.spawn_z:+.4f}  "
                f"(subsequent trials will lift back here if they start lower)\n"
                f"━━━━━━━━━━━━━━━━━━━━━━━")
            return
        z_drop = self.spawn_z - cz
        if z_drop < SPAWN_Z_RETURN_THRESHOLD_M:
            self.get_logger().info(
                f"Spawn-Z check: TCP Z={cz:+.4f} within "
                f"{SPAWN_Z_RETURN_THRESHOLD_M*1000:.0f}mm of spawn "
                f"Z={self.spawn_z:+.4f} (drop={z_drop*1000:+.1f}mm) — no lift needed")
            return
        self.get_logger().info(
            f"━━━ SPAWN-Z LIFT (robot plummeted from previous trial) ━━━\n"
            f"  Current TCP Z = {cz:+.4f}\n"
            f"  Spawn Z       = {self.spawn_z:+.4f}\n"
            f"  Drop          = {z_drop*1000:+.1f}mm "
            f"(threshold {SPAWN_Z_RETURN_THRESHOLD_M*1000:.0f}mm)\n"
            f"  → Lifting straight up with HIGH stiffness to break free of cable snag\n"
            f"━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
        # v14.6: Aggressive lift — much higher stiffness (Z) to overcome
        # cable snag. Standard HOMING_STIFFNESS Z is 150; bump to 400 for lift.
        LIFT_STIFFNESS = [
            150.0, 0,    0,    0,    0,    0,
            0,    150.0, 0,    0,    0,    0,
            0,    0,    400.0, 0,    0,    0,   # Z stiffness 4x higher
            0,    0,    0,    50.0, 0,    0,
            0,    0,    0,    0,    50.0, 0,
            0,    0,    0,    0,    0,    50.0]
        LIFT_DAMPING = [
            60.0,  0,    0,    0,    0,    0,
            0,    60.0,  0,    0,    0,    0,
            0,    0,    80.0,  0,    0,    0,
            0,    0,    0,    20.0, 0,    0,
            0,    0,    0,    0,    20.0, 0,
            0,    0,    0,    0,    0,    20.0]
        # Lift TARGET should be ABOVE spawn-Z (extra margin) so the IK
        # command has room to actually reach spawn_z (impedance always
        # leaves some residual).
        LIFT_TARGET_Z = self.spawn_z + 0.005  # 5mm above spawn
        MAX_ATTEMPTS = 3
        SUCCESS_TOLERANCE_M = 0.01    # 10mm of spawn_z is "close enough"
        for attempt in range(1, MAX_ATTEMPTS + 1):
            self._send_pose(
                move_robot, cx, cy, LIFT_TARGET_Z,
                qx, qy, qz, qw,
                stiffness=LIFT_STIFFNESS, damping=LIFT_DAMPING)
            # v14.6: longer settle (3s base, +10s/m) — lift through
            # cable resistance is SLOW even with high stiffness.
            settle_s = 3.0 + z_drop * 10.0
            self.get_logger().info(
                f"  Attempt {attempt}/{MAX_ATTEMPTS}: lifting to "
                f"Z={LIFT_TARGET_Z:+.4f}, settling {settle_s:.2f}s "
                f"(target {z_drop*1000:.0f}mm lift)")
            self.sleep_for(settle_s)
            current_z = self._get_tcp_z()
            if current_z is None:
                self.get_logger().warn(
                    f"  Attempt {attempt}: cannot read TCP — continuing")
                continue
            residual = self.spawn_z - current_z
            self.get_logger().info(
                f"  Attempt {attempt} result: Z={current_z:+.4f} "
                f"(residual={residual*1000:+.1f}mm)")
            if abs(residual) < SUCCESS_TOLERANCE_M:
                self.get_logger().info(
                    f"✓ Spawn-Z lift SUCCESS after {attempt} attempt(s)")
                return
            # Update current Z for next attempt's z_drop calc
            z_drop = self.spawn_z - current_z
            if z_drop < SPAWN_Z_RETURN_THRESHOLD_M:
                self.get_logger().info(
                    f"✓ Spawn-Z lift SUCCESS after {attempt} attempt(s) "
                    f"(within threshold)")
                return
        self.get_logger().warn(
            f"⚠ Spawn-Z lift FAILED after {MAX_ATTEMPTS} attempts — "
            f"continuing anyway. Survey/homing will use current Z.")
    # ── Phase 1: Canonical homing ────────────────────────────────────────
    def _canonical_homing(self, move_robot, T_base_port, port_type,
                          send_feedback, include_tilt=False):
        port_pos = T_base_port[:3, 3]
        yaw_port = _rot_to_yaw(T_base_port[:3, :3])
        offset_in_port = (SFP_OFFSET_IN_PORT if port_type == "sfp"
                          else SC_OFFSET_IN_PORT)
        Rz = _yaw_to_rot(yaw_port)
        offset_in_base = Rz @ offset_in_port
        target_pos = port_pos + offset_in_base
        R_target = Rz @ R_Y_PI
        # v14.4: optionally compose the plug-straight tilt into the homing
        # target orientation. The wrist arrives at the port already tilted
        # by 21° (sfp) / 33° (sc), so Phase 1.6 becomes a no-op and the
        # camera's view stays consistent throughout the approach. Fixes the
        # "adjacent NIC enters frame after tilt" issue seen in trials 1 & 2.
        if include_tilt:
            if port_type == "sfp":
                local_tilt_deg = -21.0
                local_yaw_deg  =  +3.0
            else:
                local_tilt_deg = -33.0
                local_yaw_deg  =  +9.0
            ax = math.radians(local_tilt_deg)
            cx_, sx_ = math.cos(ax), math.sin(ax)
            R_local_x = np.array([
                [1.0, 0.0, 0.0],
                [0.0, cx_, -sx_],
                [0.0, sx_,  cx_],
            ])
            az = math.radians(local_yaw_deg)
            cz_, sz_ = math.cos(az), math.sin(az)
            R_local_z = np.array([
                [ cz_, -sz_, 0.0],
                [ sz_,  cz_, 0.0],
                [ 0.0,  0.0, 1.0],
            ])
            R_local = R_local_x @ R_local_z
            R_target = R_target @ R_local
            self.get_logger().info(
                f"Phase 1: include_tilt=True — composing R_local "
                f"(tilt={local_tilt_deg:+.0f}°, yaw={local_yaw_deg:+.0f}°) "
                f"into homing target orientation. Phase 1.6 will be skipped.")
        qx, qy, qz, qw = _rot_to_quat(R_target)
        tcp = self._get_tcp_pose()
        if tcp is None:
            self.get_logger().warn(
                "Phase 1: cannot read TCP — skipping homing")
            return False
        spawn_z = tcp[2]
        if target_pos[2] > spawn_z:
            self.get_logger().info(
                f"Phase 1: clamping target Z {target_pos[2]:+.4f} → "
                f"{spawn_z:+.4f} (spawn-Z safety limit)")
            target_pos[2] = spawn_z
        delta = target_pos - tcp[:3]
        dist  = float(np.linalg.norm(delta))
        # v14.6 BUGFIX: also check orientation difference. When include_tilt
        # is True, current quat is identity but target quat has the 21°/33°
        # tilt baked in. If we only check XY distance, Phase 1 wrongly
        # short-circuits and the tilt is never applied (trial 1 sanity-check
        # symptom: locked quat = (1,0,0,0), descent stalls at Z=0.23).
        # Quaternion "distance": angle = 2*acos(|dot(q1,q2)|).
        cur_q = tcp[3:7]  # (qx, qy, qz, qw)
        # Normalize both quats to be safe
        cur_qn = np.asarray(cur_q) / (np.linalg.norm(cur_q) + 1e-12)
        tgt_q  = np.array([qx, qy, qz, qw])
        tgt_qn = tgt_q / (np.linalg.norm(tgt_q) + 1e-12)
        dot = abs(float(np.dot(cur_qn, tgt_qn)))
        dot = min(1.0, dot)
        angle_diff_deg = math.degrees(2.0 * math.acos(dot))
        ORIENT_MIN_DIFF_DEG = 2.0  # 2° threshold
        self.get_logger().info(
            f"Phase 1: port=({port_pos[0]:+.4f}, {port_pos[1]:+.4f}, "
            f"{port_pos[2]:+.4f})  yaw={math.degrees(yaw_port):+.1f}°")
        self.get_logger().info(
            f"Phase 1: target_tcp=({target_pos[0]:+.4f}, {target_pos[1]:+.4f}, "
            f"{target_pos[2]:+.4f})")
        self.get_logger().info(
            f"Phase 1: current_tcp=({tcp[0]:+.4f}, {tcp[1]:+.4f}, "
            f"{tcp[2]:+.4f})")
        self.get_logger().info(
            f"Phase 1: distance to target = {dist*100:.2f} cm, "
            f"orient diff = {angle_diff_deg:.1f}°")
        if dist > HOMING_MAX_DIST_M:
            self.get_logger().warn(
                f"Phase 1: target {dist*100:.1f}cm away exceeds limit "
                f"{HOMING_MAX_DIST_M*100:.0f}cm — REJECTING homing")
            send_feedback(f"Homing rejected — target too far ({dist*100:.0f}cm)")
            return False
        # v14.6: skip ONLY if BOTH position and orientation already match
        if dist < HOMING_MIN_MOVE_M and angle_diff_deg < ORIENT_MIN_DIFF_DEG:
            self.get_logger().info(
                f"Phase 1: already at target "
                f"(dist={dist*1000:.1f}mm, orient={angle_diff_deg:.1f}°) "
                f"— skipping move")
            send_feedback("Already at target")
            return True
        if dist < HOMING_MIN_MOVE_M:
            self.get_logger().info(
                f"Phase 1: XY already close ({dist*1000:.1f}mm) but "
                f"orientation differs by {angle_diff_deg:.1f}° "
                f"— sending pose to apply tilt")
        send_feedback(f"Homing {dist*100:.1f}cm to canonical pose")
        self._send_pose(
            move_robot,
            target_pos[0], target_pos[1], target_pos[2],
            qx, qy, qz, qw,
            stiffness=HOMING_STIFFNESS,
            damping=HOMING_DAMPING,
        )
        self.get_logger().info(
            f"Phase 1: settling for {HOMING_SETTLE_TIME}s")
        self.sleep_for(HOMING_SETTLE_TIME)
        final_tcp = self._get_tcp_pose()
        if final_tcp is not None:
            final_err = float(np.linalg.norm(target_pos - final_tcp[:3]))
            self.get_logger().info(
                f"Phase 1: final TCP=({final_tcp[0]:+.4f}, "
                f"{final_tcp[1]:+.4f}, {final_tcp[2]:+.4f}) "
                f"error={final_err*100:.2f}cm")
        self.get_logger().info("Phase 1: homing complete")
        return True
    # ── Phase 1.6: Rotate gripper so plug points straight down ──────────
    def _set_plug_straight(self, move_robot, T_base_port, port_type,
                           send_feedback):
        if port_type == "sfp":
            local_tilt_deg = -21.0
            local_yaw_deg  =  +3.0
        else:
            local_tilt_deg = -33.0
            local_yaw_deg  =  +9.0
        tcp = self._get_tcp_pose()
        if tcp is None:
            self.get_logger().warn(
                "Phase 1.6: cannot read TCP — skipping plug-straight")
            return False
        R_TCP_old = _quat_to_rot(tcp[3], tcp[4], tcp[5], tcp[6])
        ax = math.radians(local_tilt_deg)
        cx, sx = math.cos(ax), math.sin(ax)
        R_local_x = np.array([
            [1.0, 0.0, 0.0],
            [0.0,  cx,  -sx],
            [0.0,  sx,   cx],
        ])
        az = math.radians(local_yaw_deg)
        cz, sz = math.cos(az), math.sin(az)
        R_local_z = np.array([
            [ cz, -sz, 0.0],
            [ sz,  cz, 0.0],
            [0.0, 0.0, 1.0],
        ])
        R_local = R_local_x @ R_local_z
        R_TCP_target = R_TCP_old @ R_local
        cur_q = np.array([tcp[3], tcp[4], tcp[5], tcp[6]], dtype=np.float64)
        tgt_q = np.array(_rot_to_quat(R_TCP_target), dtype=np.float64)
        if float(np.dot(cur_q, tgt_q)) < 0.0:
            tgt_q = -tgt_q
        dot = float(np.clip(np.dot(cur_q, tgt_q), -1.0, 1.0))
        rot_angle_deg = math.degrees(2.0 * math.acos(dot))
        self.get_logger().info(
            f"Phase 1.6: LOCAL-frame tilt+yaw for {port_type}")
        self.get_logger().info(
            f"  R_x_local({local_tilt_deg:+.1f}°) @ R_z_local({local_yaw_deg:+.1f}°)  "
            f"total TCP rotation: {rot_angle_deg:.1f}°")
        self.get_logger().info(
            f"  TCP stays at ({tcp[0]:+.4f}, {tcp[1]:+.4f}, {tcp[2]:+.4f})")
        send_feedback(
            f"Local tilt R_x({local_tilt_deg:+.0f}°) yaw R_z({local_yaw_deg:+.0f}°)")
        N_STEPS  = 20
        DURATION = 2.0
        dt       = DURATION / N_STEPS
        for i in range(1, N_STEPS + 1):
            t = i / N_STEPS
            d = float(np.clip(np.dot(cur_q, tgt_q), -1.0, 1.0))
            if d > 0.9995:
                interp_q = cur_q * (1.0 - t) + tgt_q * t
                interp_q /= np.linalg.norm(interp_q)
            else:
                theta_0 = math.acos(d)
                theta   = theta_0 * t
                s_t     = math.sin(theta)
                s_0     = math.sin(theta_0)
                s0 = math.cos(theta) - d * s_t / s_0
                s1 = s_t / s_0
                interp_q = s0 * cur_q + s1 * tgt_q
            self._send_pose(
                move_robot,
                tcp[0], tcp[1], tcp[2],
                float(interp_q[0]), float(interp_q[1]),
                float(interp_q[2]), float(interp_q[3]),
                stiffness=HOMING_STIFFNESS,
                damping=HOMING_DAMPING)
            self.sleep_for(dt)
        self.sleep_for(0.3)
        final_tcp = self._get_tcp_pose()
        if final_tcp is not None:
            R_final = _quat_to_rot(
                final_tcp[3], final_tcp[4], final_tcp[5], final_tcp[6])
            sinp = max(-1.0, min(1.0, -R_final[2, 0]))
            f_pitch = math.degrees(math.asin(sinp))
            f_roll  = math.degrees(math.atan2(R_final[2, 1], R_final[2, 2]))
            f_yaw   = math.degrees(math.atan2(R_final[1, 0], R_final[0, 0]))
            pos_drift = math.sqrt(
                (final_tcp[0] - tcp[0]) ** 2
                + (final_tcp[1] - tcp[1]) ** 2
                + (final_tcp[2] - tcp[2]) ** 2)
            self.get_logger().info(
                f"  FINAL rpy = ({f_roll:+.1f}, {f_pitch:+.1f}, {f_yaw:+.1f})°  "
                f"pos_drift={pos_drift*1000:.1f}mm")
        self.get_logger().info("Phase 1.6: plug-straight rotation complete")
        return True
    # ── Phase 2: ACT (same as V9) ────────────────────────────────────────
    def _build_act_obs(self, obs_msg, port_type, port_index):
        if ACT_MODE == "blind":
            zero_img = torch.zeros((1, 3, IMG_H, IMG_W), device=self.device)
            left = center = right = zero_img
        else:
            left   = _img_msg_to_act_tensor(
                obs_msg.left_image,   self.device, self.img_mean, self.img_std)
            center = _img_msg_to_act_tensor(
                obs_msg.center_image, self.device, self.img_mean, self.img_std)
            right  = _img_msg_to_act_tensor(
                obs_msg.right_image,  self.device, self.img_mean, self.img_std)
        tcp = obs_msg.controller_state.tcp_pose
        tcp_vec = np.array([
            tcp.position.x, tcp.position.y, tcp.position.z,
            tcp.orientation.x, tcp.orientation.y,
            tcp.orientation.z, tcp.orientation.w,
        ], dtype=np.float32)
        f = obs_msg.wrist_wrench.wrench.force
        t = obs_msg.wrist_wrench.wrench.torque
        wrench = np.array([f.x, f.y, f.z, t.x, t.y, t.z], dtype=np.float32)
        port_enc = np.array([
            float(port_type == "sfp"),
            float(port_type == "sc"),
            float(port_index),
        ], dtype=np.float32)
        state = np.concatenate([tcp_vec, wrench, port_enc])
        state_t = torch.from_numpy(state).float().unsqueeze(0).to(self.device)
        state_t = (state_t - self.state_mean) / self.state_std
        return {
            "observation.images.left":   left,
            "observation.images.center": center,
            "observation.images.right":  right,
            "observation.state":         state_t,
        }
    def _predict_act_chunk(self, obs):
        self.act_policy.reset()
        chunk = []
        with torch.inference_mode():
            for _ in range(CHUNK_SIZE):
                action_t = self.act_policy.select_action(obs)
                action_raw = (action_t * self.action_std) + self.action_mean
                chunk.append(action_raw[0].cpu().numpy())
        return chunk
    def _act_phase(self, task, get_observation, move_robot, send_feedback,
                   port_type, target_port_index):
        self.get_logger().info(
            f"Phase 2: ACT running ({port_type} port {target_port_index})")
        send_feedback(f"Phase 2: ACT ({port_type})")
        step = 0
        z_window = deque(maxlen=Z_STALL_WINDOW)
        stalled = False
        while step < ACT_MAX_STEPS:
            obs_msg = get_observation()
            if obs_msg is None:
                self.sleep_for(0.05)
                continue
            obs   = self._build_act_obs(obs_msg, port_type, target_port_index)
            chunk = self._predict_act_chunk(obs)
            for action_vec in chunk:
                if step >= ACT_MAX_STEPS:
                    break
                self._send_pose(
                    move_robot,
                    action_vec[0], action_vec[1], action_vec[2],
                    action_vec[3], action_vec[4], action_vec[5], action_vec[6],
                )
                step += 1
                self.sleep_for(0.05)
                if port_type == "sfp" and step >= ACT_STALL_CHECK_START:
                    z_now = self._get_tcp_z()
                    if z_now is not None:
                        z_window.append(z_now)
                        if len(z_window) == Z_STALL_WINDOW:
                            spread = max(z_window) - min(z_window)
                            if spread < Z_STALL_THRESHOLD:
                                stalled = True
                                self.get_logger().info(
                                    f"Phase 2: Z stalled at {z_now:.4f} "
                                    f"(spread={spread:.5f}) at step {step}")
                                break
                if stalled:
                    break
            if stalled:
                break
            if step % 150 == 0 and step > 0:
                z_log = z_window[-1] if z_window else -1.0
                self.get_logger().info(
                    f"Phase 2: step={step}  Z={z_log:.4f}")
        last_z = (z_window[-1] if z_window else (self._get_tcp_z() or 0.24))
        return (not stalled, last_z, step)
    # ── Phase 3: Spiral + descent fallback ───────────────────────────────
    def _forced_descent(self, move_robot, x, y, start_z,
                        qx, qy, qz, qw, post_spiral=False):
        """Push TCP straight down from start_z toward INSERTION_TARGET_Z.
        Default behavior (post_spiral=False): v11 original — stop on ANY
        Z stall (5 consecutive steps with <0.2mm movement). Used by:
          - Phase 2 (initial descent after Phase 1.7 servo)
          - _wiggle_and_push (recovery wiggle then push)
        post_spiral=True behavior (v12): trial only ends when Z stalls AND
        Z < INSERTION_SUCCESS_Z. If Z stalls ABOVE SUCCESS_Z, hand off to
        _wiggle_while_pushing (back-front XY wiggle + Z push), then return.
        Used ONLY by:
          - _spiral_search when spiral phase finds a hole
          - _spiral_search when retract phase finds a hole
        """
        if post_spiral:
            self.get_logger().info(
                f"Phase 3a [post-spiral]: forced descent Z={start_z:.4f} → "
                f"{INSERTION_TARGET_Z:.4f}  "
                f"(success when Z < {INSERTION_SUCCESS_Z:.4f}; "
                f"if stalled above, do back-front wiggle while pushing down)")
        else:
            self.get_logger().info(
                f"Phase 3a: forced descent Z={start_z:.4f} → {INSERTION_TARGET_Z:.4f}")
        self.get_logger().info(
            f"  V15: XY LOCKED at ({x:+.4f}, {y:+.4f}) with stiffness "
            f"800 N/m — cable can't pull wrist sideways during descent")
        prev_z = start_z
        stall_count = 0
        z = start_z
        while z > INSERTION_TARGET_Z:
            z -= 0.0005
            # ── cap commanded-vs-actual gap ──
            # Prevents controller's tracking-error timeout from
            # resetting the target if the plug stops moving. Always on
            # for post_spiral; for non-post_spiral, only enabled while
            # actual_z is still above PHASE2_STALL_MIN_Z (where the
            # cable can pull and stall us — we want to keep grinding
            # down without runaway commanded-z).
            if post_spiral:
                actual_now = self._get_tcp_z()
                if actual_now is not None:
                    floor = actual_now - POST_SPIRAL_DESCENT_GAP_CAP_M
                    if z < floor:
                        z = floor
            else:
                # V15: enable gap cap during non-post_spiral descent
                # too, but only while above the rim threshold. Below
                # the threshold we're in real rim contact and the
                # uncapped commanded-z drives the seating force.
                actual_now = self._get_tcp_z()
                if (actual_now is not None
                        and actual_now > PHASE2_STALL_MIN_Z):
                    floor = actual_now - POST_SPIRAL_DESCENT_GAP_CAP_M
                    if z < floor:
                        z = floor
            self._send_pose(
                move_robot, x, y, z, qx, qy, qz, qw,
                stiffness=DESCENT_LOCK_STIFFNESS,
                damping=DESCENT_LOCK_DAMPING)
            self.sleep_for(0.05)
            actual_z = self._get_tcp_z()
            if actual_z is None:
                continue
            movement = abs(actual_z - prev_z)
            if movement < INSERTION_STALL_MM:
                stall_count += 1
                if stall_count >= INSERTION_STALL_STEPS:
                    if not post_spiral:
                        # ── V15 Z GUARD ──
                        # If actual_z is still ABOVE the rim threshold,
                        # the cable is just pulling the wrist back in
                        # the air. Don't honor this stall — reset
                        # counter, cap the commanded-vs-actual gap to
                        # prevent the controller's tracking-error
                        # timeout from resetting the target, and keep
                        # pushing down. Only when actual_z drops below
                        # PHASE2_STALL_MIN_Z (rim level ~0.2499) do we
                        # call the stall real and exit to recovery.
                        if actual_z < PHASE2_STALL_MIN_Z:
                            self.get_logger().info(
                                f"✓ Insertion stall at Z={actual_z:.4f} "
                                f"(stall_count={stall_count}, BELOW "
                                f"threshold {PHASE2_STALL_MIN_Z:.4f} — "
                                f"real rim contact)")
                            self.sleep_for(0.5)
                            return
                        else:
                            self.get_logger().warn(
                                f"  ↺ Spurious stall at Z={actual_z:.4f} "
                                f"(ABOVE threshold "
                                f"{PHASE2_STALL_MIN_Z:.4f}) — cable is "
                                f"holding wrist in the air. Resetting "
                                f"stall counter and capping z-gap, "
                                f"continuing descent.")
                            stall_count = 0
                            # Cap the gap so commanded-z doesn't run
                            # away from actual_z and trip the
                            # controller's tracking-error reset.
                            actual_now = self._get_tcp_z()
                            if actual_now is not None:
                                floor = actual_now - POST_SPIRAL_DESCENT_GAP_CAP_M
                                if z < floor:
                                    z = floor
                            prev_z = actual_z
                            continue
                    # ── post_spiral=True: check if truly inserted ──
                    if actual_z < INSERTION_SUCCESS_Z:
                        self.get_logger().info(
                            f"✓ Truly inserted: stalled at Z={actual_z:.4f} "
                            f"BELOW success threshold {INSERTION_SUCCESS_Z:.4f}")
                        self.sleep_for(0.5)
                        return
                    # Stalled above SUCCESS_Z — wiggle while pushing down
                    self.get_logger().info(
                        f"  ↺ Stalled at Z={actual_z:.4f} ABOVE SUCCESS_Z "
                        f"{INSERTION_SUCCESS_Z:.4f} — engaging back-front wiggle")
                    wiggle_succeeded = self._wiggle_while_pushing(
                        move_robot, x, y, actual_z, qx, qy, qz, qw)
                    if wiggle_succeeded:
                        self.get_logger().info(
                            "✓ Wiggle pushed plug below SUCCESS_Z — inserted")
                    else:
                        self.get_logger().info(
                            "Wiggle complete — did not reach SUCCESS_Z, ending descent")
                    self.sleep_for(0.5)
                    return
            else:
                stall_count = 0
            prev_z = actual_z
        if post_spiral:
            self.get_logger().info(
                f"Phase 3a [post-spiral]: reached target Z={INSERTION_TARGET_Z}")
        else:
            self.get_logger().info("Phase 3a: descent complete (reached target Z)")
        self.sleep_for(0.5)
    # ── v12: wiggle while pushing down (post-spiral stall recovery) ──────
    def _wiggle_while_pushing(self, move_robot, x_center, y_center,
                              start_z, qx, qy, qz, qw):
        """Back-and-forth wiggle in gripper-local Y while pushing Z down.
        Called when post-spiral descent stalls ABOVE INSERTION_SUCCESS_Z.
        Tries to jiggle the plug past whatever is binding it. Returns True
        if Z drops below SUCCESS_Z during the wiggle, False otherwise.
        """
        try:
            import transforms3d.quaternions as txq
            R = txq.quat2mat([qw, qx, qy, qz])
            wiggle_dir = R @ np.array([0.0, 1.0, 0.0])
        except Exception:
            wiggle_dir = np.array([0.0, 1.0, 0.0])
        total_steps = WIGGLE_DESCENT_CYCLES * 2 * WIGGLE_DESCENT_STEPS_PER_HALF
        self.get_logger().info(
            f"  Wiggle+push: ±{WIGGLE_DESCENT_AMPLITUDE_M*1000:.1f}mm back-front "
            f"({WIGGLE_DESCENT_CYCLES} cycles, {total_steps} steps total, "
            f"Z drops {WIGGLE_DESCENT_Z_PUSH_PER_STEP*1000:.1f}mm/step)")
        z = start_z
        step_num = 0
        for cycle in range(WIGGLE_DESCENT_CYCLES):
            for direction in [+1, -1]:   # forward half, then backward half
                for i in range(WIGGLE_DESCENT_STEPS_PER_HALF):
                    # Linear ramp from 0 to ±amplitude over the half-cycle
                    t = (i + 1) / WIGGLE_DESCENT_STEPS_PER_HALF
                    offset = direction * WIGGLE_DESCENT_AMPLITUDE_M * t
                    wx = x_center + wiggle_dir[0] * offset
                    wy = y_center + wiggle_dir[1] * offset
                    # Push Z down each step
                    z -= WIGGLE_DESCENT_Z_PUSH_PER_STEP
                    if z <= INSERTION_TARGET_Z:
                        return True   # hit hard floor
                    # Gap cap (prevent controller tracking-error reset)
                    actual_now = self._get_tcp_z()
                    if actual_now is not None:
                        floor = actual_now - POST_SPIRAL_DESCENT_GAP_CAP_M
                        if z < floor:
                            z = floor
                    self._send_pose(
                        move_robot, wx, wy, z, qx, qy, qz, qw,
                        stiffness=SOFT_STIFFNESS, damping=SOFT_DAMPING)
                    self.sleep_for(0.05)
                    step_num += 1
                    actual_z = self._get_tcp_z()
                    if actual_z is None:
                        continue
                    if step_num % 8 == 0:
                        self.get_logger().info(
                            f"    [wiggle cyc {cycle+1}/{WIGGLE_DESCENT_CYCLES}] "
                            f"step {step_num}/{total_steps}  Z={actual_z:.4f}")
                    if actual_z < INSERTION_SUCCESS_Z:
                        self.get_logger().info(
                            f"  ✓ Wiggle pushed plug below SUCCESS_Z at cycle "
                            f"{cycle+1} step {step_num}: Z={actual_z:.4f}")
                        return True
        final_z = self._get_tcp_z()
        self.get_logger().info(
            f"  Wiggle complete after {total_steps} steps  "
            f"(final Z={final_z if final_z is not None else -1.0:.4f})")
        return False
    def _wiggle_and_push(self, move_robot, stall_z):
        tcp = self._get_tcp_pose()
        if tcp is None:
            self.get_logger().warn("Wiggle: cannot read TCP — skipping")
            return False
        cx, cy = tcp[0], tcp[1]
        qx, qy, qz, qw = tcp[3], tcp[4], tcp[5], tcp[6]
        try:
            import transforms3d.quaternions as txq
            R = txq.quat2mat([qw, qx, qy, qz])
            local_y_world = R @ np.array([0.0, 1.0, 0.0])
        except Exception:
            local_y_world = np.array([0.0, 1.0, 0.0])
        retract_dx = local_y_world[0] * WIGGLE_RETRACT_M
        retract_dy = local_y_world[1] * WIGGLE_RETRACT_M
        target_x = cx + retract_dx
        target_y = cy + retract_dy
        self.get_logger().info(
            f"Wiggle: stall_z={stall_z:.4f}  lift_z={WIGGLE_LIFT_Z:.4f}  "
            f"retract={WIGGLE_RETRACT_M*1000:.1f}mm")
        self.get_logger().info(
            f"  Wiggle 1/3: lifting to Z={WIGGLE_LIFT_Z:.4f}")
        self._send_pose(move_robot, cx, cy, WIGGLE_LIFT_Z,
                        qx, qy, qz, qw,
                        stiffness=SOFT_STIFFNESS, damping=SOFT_DAMPING)
        self.sleep_for(WIGGLE_PAUSE_S)
        self.get_logger().info(
            f"  Wiggle 2/3: retract to ({target_x:+.4f}, {target_y:+.4f})")
        self._send_pose(move_robot, target_x, target_y, WIGGLE_LIFT_Z,
                        qx, qy, qz, qw,
                        stiffness=SOFT_STIFFNESS, damping=SOFT_DAMPING)
        self.sleep_for(WIGGLE_PAUSE_S)
        self.get_logger().info("  Wiggle 3/3: pushing down")
        self._forced_descent(move_robot, target_x, target_y, WIGGLE_LIFT_Z,
                             qx, qy, qz, qw)
        final_z = self._get_tcp_z()
        if final_z is None:
            return False
        progress = stall_z - final_z
        if progress > WIGGLE_SUCCESS_DEPTH:
            self.get_logger().info(
                f"✓ Wiggle succeeded: {stall_z:.4f} → {final_z:.4f} "
                f"({progress*1000:.1f}mm progress)")
            return True
        else:
            self.get_logger().info(
                f"Wiggle insufficient: {stall_z:.4f} → {final_z:.4f} "
                f"({progress*1000:.1f}mm) — falling back to spiral")
            return False
    def _v15_narrow_band_recovery(self, move_robot, stall_z):
        """V15: if Phase 2 stalls in the narrow Z-band just above the
        port rim, lift slightly and nudge forward before going to
        spiral. The cable's free end often hooks on the port rim and
        tugs the wrist back, making spiral search useless because the
        wrist can't descend at all. Lifting to ~Z=0.238 + sliding
        forward 8mm in +Y world frees the rim contact and shifts the
        plug from 'above the rim' to 'above the hole'.
        Returns True if recovery was applied, False if skipped.
        """
        if not V15_RECOVERY_ENABLED:
            return False
        # Narrow-band check
        if not (V15_RECOVERY_STALL_Z_MIN < stall_z < V15_RECOVERY_STALL_Z_MAX):
            self.get_logger().info(
                f"V15 recovery: stall_z={stall_z:.4f} NOT in narrow band "
                f"({V15_RECOVERY_STALL_Z_MIN:.4f}, "
                f"{V15_RECOVERY_STALL_Z_MAX:.4f}) — skipping, "
                f"continuing to spiral")
            return False
        tcp = self._get_tcp_pose()
        if tcp is None:
            self.get_logger().warn(
                "V15 recovery: cannot read TCP — skipping")
            return False
        cx, cy, cz = tcp[0], tcp[1], tcp[2]
        qx, qy, qz, qw = tcp[3], tcp[4], tcp[5], tcp[6]
        # Step 1: LIFT to V15_RECOVERY_LIFT_TARGET_Z (capped at
        # current+LIFT_MAX_M for safety)
        cap_z = cz + V15_RECOVERY_LIFT_MAX_M
        target_z = min(cap_z, V15_RECOVERY_LIFT_TARGET_Z)
        self.get_logger().info(
            f"V15 NARROW-BAND RECOVERY: stall_z={stall_z:.4f} IS in band "
            f"({V15_RECOVERY_STALL_Z_MIN:.4f}, "
            f"{V15_RECOVERY_STALL_Z_MAX:.4f}) — "
            f"cable likely caught on rim. Applying lift + forward nudge.")
        self.get_logger().info(
            f"  V15 recovery [1/2]: LIFT from Z={cz:.4f} to Z={target_z:.4f} "
            f"({(target_z - cz)*1000:+.1f}mm; cap was {cap_z:.4f})")
        self._send_pose(move_robot, cx, cy, target_z, qx, qy, qz, qw)
        self.sleep_for(V15_RECOVERY_SETTLE_S)
        actual_after_lift = self._get_tcp_pose()
        if actual_after_lift is not None:
            self.get_logger().info(
                f"  V15 recovery: after lift TCP="
                f"({actual_after_lift[0]:+.4f}, "
                f"{actual_after_lift[1]:+.4f}, "
                f"{actual_after_lift[2]:+.4f})")
        # Step 2: FORWARD nudge by V15_RECOVERY_FORWARD_M in +Y world
        new_y = cy + V15_RECOVERY_FORWARD_M
        self.get_logger().info(
            f"  V15 recovery [2/2]: FORWARD +Y from {cy:.4f} to "
            f"{new_y:.4f} ({V15_RECOVERY_FORWARD_M*1000:+.1f}mm)")
        self._send_pose(move_robot, cx, new_y, target_z, qx, qy, qz, qw)
        self.sleep_for(V15_RECOVERY_SETTLE_S)
        actual_after_fwd = self._get_tcp_pose()
        if actual_after_fwd is not None:
            self.get_logger().info(
                f"  V15 recovery: after forward TCP="
                f"({actual_after_fwd[0]:+.4f}, "
                f"{actual_after_fwd[1]:+.4f}, "
                f"{actual_after_fwd[2]:+.4f})")
        self.get_logger().info(
            f"V15 NARROW-BAND RECOVERY: complete. Spiral will search "
            f"around new position.")
        return True

    def _reg_pre_spiral_nudge(self, move_robot):
        """V15.10: small forward (toward task board) nudge before spiral.
        Called ONLY when the visual servo CONVERGED in REG mode (i.e., it
        fell back to the registered port pose because YOLO lost the port
        mid-servo, and reached convergence using the deterministic
        projection).
        Rationale: the registered pose can have a small systematic offset
        from the true port location due to PnP fusion error during the
        survey. A 4mm forward nudge biases the spiral starting point
        toward the board, which is the most likely direction the actual
        port sits relative to the registered location. YOLO-mode
        convergence does NOT trigger this nudge — YOLO locked onto the
        live port image so no bias correction is needed.
        Direction = -local_y_world (same as the spiral pattern's "DOWN"
        direction: gripper-local -Y = forward toward the task board).
        """
        tcp = self._get_tcp_pose()
        if tcp is None:
            self.get_logger().warn(
                "V15.10 REG nudge: cannot read TCP — skipping")
            return
        cx, cy, cz = tcp[0], tcp[1], tcp[2]
        qx, qy, qz, qw = tcp[3], tcp[4], tcp[5], tcp[6]
        try:
            import transforms3d.quaternions as txq
            R = txq.quat2mat([qw, qx, qy, qz])
            forward_dir = -R @ np.array([0.0, 1.0, 0.0])   # -local_y_world
        except Exception:
            forward_dir = np.array([0.0, -1.0, 0.0])
        new_x = cx + forward_dir[0] * REG_PRE_SPIRAL_NUDGE_M
        new_y = cy + forward_dir[1] * REG_PRE_SPIRAL_NUDGE_M
        self.get_logger().info(
            f"━━━ V15.10 REG PRE-SPIRAL NUDGE ━━━\n"
            f"  Servo converged in REG mode — applying "
            f"{REG_PRE_SPIRAL_NUDGE_M*1000:.1f}mm forward nudge "
            f"before spiral.\n"
            f"  Before nudge:  ({cx:+.4f}, {cy:+.4f}, {cz:+.4f})\n"
            f"  After nudge:   ({new_x:+.4f}, {new_y:+.4f}, {cz:+.4f})\n"
            f"  Forward dir:   ({forward_dir[0]:+.3f}, "
            f"{forward_dir[1]:+.3f}, {forward_dir[2]:+.3f})\n"
            f"━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
        self._send_pose(
            move_robot, new_x, new_y, cz, qx, qy, qz, qw,
            stiffness=SOFT_STIFFNESS, damping=SOFT_DAMPING)
        self.sleep_for(REG_PRE_SPIRAL_NUDGE_SETTLE_S)
        actual_tcp = self._get_tcp_pose()
        if actual_tcp is not None:
            self.get_logger().info(
                f"V15.10 REG nudge: settled at "
                f"({actual_tcp[0]:+.4f}, {actual_tcp[1]:+.4f}, "
                f"{actual_tcp[2]:+.4f})")
    def _post_act_recovery(self, move_robot, last_z, port_type):
        if last_z < DESCENT_TRIGGER_Z:
            self.get_logger().info(
                f"Recovery: stall_z={last_z:.4f} below trigger "
                f"{DESCENT_TRIGGER_Z:.4f} — trying wiggle first")
            wiggle_ok = self._wiggle_and_push(move_robot, last_z)
            if wiggle_ok:
                return
            self.get_logger().info("Recovery: wiggle didn't help — spiraling")
        else:
            self.get_logger().info(
                f"Recovery: stall_z={last_z:.4f} above trigger "
                f"{DESCENT_TRIGGER_Z:.4f} — checking V15 narrow-band rescue first")
            # V15: if in narrow rim-snag band, lift + nudge forward
            # before spiral. Returns False (and just logs) if stall_z
            # is outside the band — spiral will then run normally.
            self._v15_narrow_band_recovery(move_robot, last_z)
        # V18: small forward nudge BEFORE spiral, based on the SERVO MODE
        # the servo ended in — NOT on whether it converged. If servo ran
        # in REG mode at any point (i.e. YOLO fell back to registered pose
        # because of persistent misses), apply the 4mm forward nudge.
        # If servo stayed in YOLO mode the whole time (whether converged
        # or not), skip the nudge.
        if self._last_servo_mode == "reg":
            self.get_logger().info(
                f"Recovery: servo ended in REG mode "
                f"(_last_servo_mode='reg') — applying pre-spiral "
                f"forward nudge (regardless of convergence).")
            self._reg_pre_spiral_nudge(move_robot)
        else:
            mode_str = (self._last_servo_mode
                        if self._last_servo_mode is not None
                        else "none (servo did not run)")
            self.get_logger().info(
                f"Recovery: servo ended in mode={mode_str} — "
                f"skipping REG pre-spiral nudge.")
        self._spiral_search(move_robot, last_z)
    def _spiral_search(self, move_robot, stall_z):
        """V18: two interleaved spiral patterns, picked by SERVO MODE.
        Mode is whatever the servo was in at exit — converged or not.
        Both patterns use 6 spirals with 5 moves between them, drifting
        the spiral center to cover ~9mm × 6mm of XY territory.
        ── If servo ended in YOLO mode (or servo never ran) ──
        Use the ORIGINAL V12 pattern: RETRACT first.
            Spiral #1 (initial center)
            ↑ RETRACT 3mm
            Spiral #2
            → RIGHT 3mm
            Spiral #3
            ↑ RETRACT 3mm
            Spiral #4
            → RIGHT 3mm
            Spiral #5
            ↑ RETRACT 3mm
            Spiral #6
        Rationale: YOLO locked onto the LIVE port image, so the wrist is
        most likely already aligned within YOLO's pixel tolerance (~7px,
        ~1mm at this depth). Even if it didn't fully converge, YOLO had
        live visual feedback, so the residual error is symmetric — drift
        covers it in either direction.
        ── If servo ended in REG mode ──
        Use the V15.9 pattern: RIGHT first. (Preceded by a 4mm forward
        nudge applied in _post_act_recovery before this method runs.)
            Spiral #1 (initial center)
            → RIGHT 3mm
            Spiral #2
            ↑ RETRACT 3mm
            Spiral #3
            → RIGHT 3mm
            Spiral #4
            ↑ RETRACT 3mm
            Spiral #5
            → RIGHT 3mm
            Spiral #6
        Rationale: REG mode means the servo lost YOLO's live signal and
        was operating off the registered port pose. The registered pose
        can have a systematic offset (often biased back-and-right due to
        PnP fusion error during survey). The REG pre-spiral nudge already
        corrected forward; starting spirals with RIGHT covers the most
        likely remaining offset direction first.
        ── Common to both ──
        RIGHT   = +local_x_world (rightward across the rail)
        RETRACT = +local_y_world (backward, away from task board)
        If Z drops below DESCENT_TRIGGER_Z at any point (during a spiral
        OR a move), descend immediately (post_spiral=True).
        """
        tcp = self._get_tcp_pose()
        if tcp is None:
            self.get_logger().warn("Phase 3: cannot read TCP — skipping spiral")
            return
        cx, cy, cz = tcp[0], tcp[1], tcp[2]
        qx, qy, qz, qw = tcp[3], tcp[4], tcp[5], tcp[6]
        search_z = stall_z - 0.006
        # V15.11: pick spiral move pattern based on servo convergence mode.
        # REG-mode convergence → start with RIGHT (V15.9 pattern).
        # YOLO/no-convergence → start with RETRACT (original V12 pattern).
        use_reg_pattern = (self._last_servo_mode == "reg")
        first_move_name = "RIGHT" if use_reg_pattern else "RETRACT"
        mode_str = (self._last_servo_mode
                    if self._last_servo_mode is not None
                    else "none")
        self.get_logger().info(
            f"Phase 3b: interleaved spiral pattern  "
            f"{SPIRAL_N_TOTAL} spirals, "
            f"servo_mode='{mode_str}' → first move = {first_move_name} "
            f"({'V15.9 REG pattern' if use_reg_pattern else 'V12 YOLO pattern'})")
        self.get_logger().info(
            f"Phase 3b: stall_z={stall_z:.4f}  search_z={search_z:.4f}  "
            f"trigger={DESCENT_TRIGGER_Z:.4f}  initial center=({cx:+.4f}, {cy:+.4f})  "
            f"spiral_radius={SPIRAL_MAX_RADIUS_M*1000:.1f}mm  move={SPIRAL_STEP_NUDGE_M*1000:.1f}mm")
        try:
            import transforms3d.quaternions as txq
            R = txq.quat2mat([qw, qx, qy, qz])
            local_y_world = R @ np.array([0.0, 1.0, 0.0])
            local_x_world = R @ np.array([1.0, 0.0, 0.0])
        except Exception:
            local_y_world = np.array([0.0, 1.0, 0.0])
            local_x_world = np.array([1.0, 0.0, 0.0])
        spiral_base = _spiral_offsets(SPIRAL_MAX_RADIUS_M, 4, 24)
        SPIRAL_STEPS    = 80
        MOVE_STEPS      = 30
        MOVE_PER_STEP_M = SPIRAL_STEP_NUDGE_M / MOVE_STEPS   # 0.1 mm/step (3mm total)
        accum_dx, accum_dy = 0.0, 0.0
        for spiral_idx in range(SPIRAL_N_TOTAL):
            # ── Move phase (before every spiral EXCEPT the first) ────
            if spiral_idx > 0:
                # V15.11: alternation depends on servo mode (set above).
                # Move index = spiral_idx - 1 (0,1,2,3,4)
                #
                # REG pattern (use_reg_pattern=True):
                #   move_idx 0 → RIGHT
                #   move_idx 1 → RETRACT
                #   move_idx 2 → RIGHT
                #   move_idx 3 → RETRACT
                #   move_idx 4 → RIGHT
                #
                # YOLO/none pattern (use_reg_pattern=False):
                #   move_idx 0 → RETRACT
                #   move_idx 1 → RIGHT
                #   move_idx 2 → RETRACT
                #   move_idx 3 → RIGHT
                #   move_idx 4 → RETRACT
                #
                # RIGHT   = +local_x_world (rightward across rail)
                # RETRACT = +local_y_world (backward, away from board)
                move_idx = spiral_idx - 1
                if use_reg_pattern:
                    move_is_right = (move_idx % 2 == 0)   # REG: even → RIGHT
                else:
                    move_is_right = (move_idx % 2 == 1)   # YOLO: odd → RIGHT
                move_name = "RIGHT" if move_is_right else "RETRACT"
                move_dir  = local_x_world if move_is_right else local_y_world
                self.get_logger().info(
                    f"  Move {move_idx+1}/5: {move_name} "
                    f"{SPIRAL_STEP_NUDGE_M*1000:.1f}mm over {MOVE_STEPS} steps")
                hole_found_during_move = False
                for i in range(MOVE_STEPS):
                    accum_dx += move_dir[0] * MOVE_PER_STEP_M
                    accum_dy += move_dir[1] * MOVE_PER_STEP_M
                    self._send_pose(
                        move_robot,
                        cx + accum_dx, cy + accum_dy, search_z,
                        qx, qy, qz, qw,
                        stiffness=SOFT_STIFFNESS, damping=SOFT_DAMPING)
                    self.sleep_for(0.05)
                    actual_z = self._get_tcp_z()
                    if actual_z is None:
                        continue
                    if actual_z < DESCENT_TRIGGER_Z:
                        tcp_now = self._get_tcp_pose()
                        if tcp_now is not None:
                            self.get_logger().info(
                                f"Hole found during {move_name} move "
                                f"step {i}  Z={actual_z:.4f} — descending")
                            self._forced_descent(
                                move_robot,
                                tcp_now[0], tcp_now[1], actual_z,
                                qx, qy, qz, qw,
                                post_spiral=True)
                            return
                if hole_found_during_move:
                    return
            # ── Spiral phase ─────────────────────────────────────────
            self.get_logger().info(
                f"  Spiral {spiral_idx+1}/{SPIRAL_N_TOTAL}  "
                f"({SPIRAL_STEPS} steps, center offset "
                f"+{accum_dx*1000:+.1f}mm/{accum_dy*1000:+.1f}mm from initial)")
            for step in range(SPIRAL_STEPS):
                dx, dy = spiral_base[step % len(spiral_base)]
                self._send_pose(
                    move_robot,
                    cx + accum_dx + dx, cy + accum_dy + dy, search_z,
                    qx, qy, qz, qw,
                    stiffness=SOFT_STIFFNESS, damping=SOFT_DAMPING)
                self.sleep_for(0.05)
                actual_z = self._get_tcp_z()
                if actual_z is None:
                    continue
                if step % 24 == 0:
                    self.get_logger().info(
                        f"  [spiral {spiral_idx+1}] step {step}  Z={actual_z:.4f}")
                if actual_z < DESCENT_TRIGGER_Z:
                    tcp_now = self._get_tcp_pose()
                    if tcp_now is not None:
                        self.get_logger().info(
                            f"Hole found at spiral {spiral_idx+1} "
                            f"step {step}  Z={actual_z:.4f} — descending")
                        self._forced_descent(
                            move_robot,
                            tcp_now[0], tcp_now[1], actual_z,
                            qx, qy, qz, qw,
                            post_spiral=True)
                        return
        self.get_logger().info(
            f"Phase 3b: {SPIRAL_N_TOTAL} spirals complete — hole not found")
    # ── Vanilla V9 fallback (when detection failed) ──────────────────────
    def _vanilla_v9_act(self, task, get_observation, move_robot,
                        send_feedback, port_type, target_port_index):
        self.get_logger().info(
            "Vanilla V9 fallback: ACT-from-home with stall handling")
        success, last_z, n_steps = self._act_phase(
            task, get_observation, move_robot, send_feedback,
            port_type, target_port_index)
        use_recovery = (
            port_type == "sfp"
            or (port_type == "sc" and SC_PIPELINE_ENABLED)
        )
        if use_recovery and not success:
            self.get_logger().info(
                f"Vanilla V9: ACT stalled at Z={last_z:.4f} — recovery")
            self._post_act_recovery(move_robot, last_z, port_type)
            self.sleep_for(POST_FALLBACK_PAUSE)
        self.get_logger().info(
            f"Trial complete (steps={n_steps}) — "
            f"holding {POST_TRIAL_PAUSE}s before returning")
        self.sleep_for(POST_TRIAL_PAUSE)
        return True
    # ── Main entry point ─────────────────────────────────────────────────
    def insert_cable(
        self,
        task: Task,
        get_observation: GetObservationCallback,
        move_robot: MoveRobotCallback,
        send_feedback: SendFeedbackCallback,
    ):
        port_type, target_port_index = _parse_port_meta(task.port_name)
        self.get_logger().info(
            f"\n{'='*60}\n"
            f"MultiViewACTPolicy — v13 + survey + registry\n"
            f"  plug={task.plug_name}  port={task.port_name}  type={port_type}\n"
            f"{'='*60}")
        # ── v14.6: SPAWN-Z LIFT IS NOW THE VERY FIRST ACTION ─────────
        # Per user request: before ANY detection/scout/survey, check
        # current TCP Z. If the previous trial left the robot plummeted
        # into a port, lift back to spawn-Z FIRST. Everything else
        # (Phase 0, survey, homing) runs at the proper altitude.
        self._maybe_return_to_spawn_z(move_robot)
        # V15.10: reset per-trial servo convergence mode tracker.
        # Old value would otherwise leak from previous trial into this
        # trial's Phase 3 recovery decision (REG nudge or not).
        self._last_servo_mode = None
        # ── Full Task message dump — debug what the engine is actually asking for
        self.get_logger().info(
            f"━━━ TASK MESSAGE FROM ENGINE ━━━\n"
            f"  task.plug_name           = '{getattr(task, 'plug_name', 'N/A')}'\n"
            f"  task.plug_type           = '{getattr(task, 'plug_type', 'N/A')}'\n"
            f"  task.port_name           = '{getattr(task, 'port_name', 'N/A')}'\n"
            f"  task.port_type           = '{getattr(task, 'port_type', 'N/A')}'\n"
            f"  task.target_module_name  = '{getattr(task, 'target_module_name', 'N/A')}'\n"
            f"  task.cable_name          = '{getattr(task, 'cable_name', 'N/A')}'\n"
            f"  task.cable_type          = '{getattr(task, 'cable_type', 'N/A')}'\n"
            f"  task.time_limit          = {getattr(task, 'time_limit', 'N/A')}\n"
            f"  ↳ parsed port_index      = {target_port_index}  ({'port_0/rightmost' if target_port_index == 0 else 'port_1/leftmost'})\n"
            f"━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
        use_full_pipeline = (
            port_type == "sfp"
            or (port_type == "sc" and SC_PIPELINE_ENABLED)
        )
        if not use_full_pipeline:
            self.get_logger().info(
                f"{port_type.upper()}: SC_PIPELINE_ENABLED=False — "
                f"skipping detection/homing/servo, going straight to V9 ACT")
            return self._vanilla_v9_act(
                task, get_observation, move_robot, send_feedback,
                port_type, target_port_index)
        # ── Phase 0: Initial detection at spawn ──────────────────────
        det = self._detect_port_pose(task, get_observation, send_feedback)
        # ── Phase 0.3: BIDIRECTIONAL Scout (if Phase 0 empty) ─────────
        if det is None and SCOUT_ENABLED:
            det = self._scout_for_port(
                task, get_observation, move_robot, send_feedback)
        if det is None:
            self.get_logger().warn(
                "Detection failed (and bidirectional scout exhausted) — V9 fallback")
            send_feedback("Detection failed, V9 fallback")
            return self._vanilla_v9_act(
                task, get_observation, move_robot, send_feedback,
                port_type, target_port_index)
        T_base_port = det["T_base_port"]
        # ── V14 NEW: Phase 0.5–0.8 — SURVEY → REGISTRY → TARGET LOOKUP ──
        # All YOLO detection (multi-angle) happens here. After this,
        # Phase 1 (homing), 1.6 (tilt), and 1.7 (servo) use the registry.
        self._registered_target_world = None  # default: D3 lock disabled
        if SURVEY_ENABLED and port_type == "sfp":
            target_module_idx = _parse_module_idx(
                getattr(task, "target_module_name", None))
            (target_entry,
             target_port_world,
             target_yaw_world) = self._run_pre_homing_survey_and_find_target(
                task, get_observation, move_robot, send_feedback,
                target_module_idx, target_port_index)
            if target_entry is not None and target_port_world is not None:
                # Override Phase 0's T_base_port with registry data
                T_base_port = np.eye(4)
                if target_yaw_world is not None:
                    T_base_port[:3, :3] = _yaw_to_rot(target_yaw_world)
                else:
                    T_base_port[:3, :3] = det["T_base_port"][:3, :3]
                T_base_port[:3, 3] = target_port_world
                # Store for D3 visual-servo world-pose lock
                self._registered_target_world = target_port_world.copy()
                self.get_logger().info(
                    f"V14: registry override — homing target set to "
                    f"({target_port_world[0]:+.4f}, "
                    f"{target_port_world[1]:+.4f}, "
                    f"{target_port_world[2]:+.4f})")
                # ── V15 PRE-HOMING APPROACH ──
                # Walk along survey square edges to a corner near the
                # target. Reduces cable tangling when the target is far
                # from spawn (e.g., mount_4).
                self._v15_pre_homing_approach(
                    target_port_world[:2], move_robot, send_feedback)
            else:
                self.get_logger().warn(
                    "V14 survey failed to find target — falling back to "
                    "Phase 0's fused pose for homing")
        # ── Phase 1: Canonical homing (v14.6 PUSHED: includes tilt) ──
        # The wrist arrives at the port ALREADY tilted to its final
        # orientation. This eliminates the "adjacent NIC enters frame
        # after Phase 1.6 tilt" problem seen in earlier trials.
        if use_full_pipeline:
            self.get_logger().info(
                f"Pausing {PRE_HOMING_PAUSE}s before homing")
            self.sleep_for(PRE_HOMING_PAUSE)
            homed = self._canonical_homing(
                move_robot, T_base_port, port_type, send_feedback,
                include_tilt=True)
            if homed:
                self.get_logger().info(
                    f"Pausing {POST_HOMING_PAUSE}s after homing")
                self.sleep_for(POST_HOMING_PAUSE)
            else:
                self.get_logger().warn(
                    "Canonical homing rejected — running ACT from current pose")
                send_feedback("Homing rejected, ACT from current pose")
        else:
            self.get_logger().info(
                f"{port_type.upper()} port: pipeline disabled — naked ACT only")
            homed = False
        # ── Phase 1.6: SKIPPED in v14.6 (tilt is part of homing) ─────
        plug_straight_ok = homed  # if we homed, we tilted
        if homed and use_full_pipeline:
            self.get_logger().info(
                f"Phase 1.6: SKIPPED — tilt was composed into homing target. "
                f"Post-homing settle for {POST_TILT_SETTLE_S}s "
                f"(scene/camera stabilization before YOLO)")
            send_feedback(f"Phase 1.6 skipped; stabilizing {POST_TILT_SETTLE_S}s")
            self.sleep_for(POST_TILT_SETTLE_S)
        if DEBUG_STOP_AFTER_PLUG_STRAIGHT:
            self.get_logger().info(
                f"\n{'='*60}\n"
                f"DEBUG MODE — stopping after plug-straight rotation.\n"
                f"  homing OK: {homed}\n"
                f"  plug-straight OK: {plug_straight_ok}\n"
                f"  Holding {POST_TRIAL_PAUSE}s before returning.\n"
                f"{'='*60}")
            send_feedback("DEBUG: stopped after plug-straight")
            self.sleep_for(POST_TRIAL_PAUSE)
            return True
        # ── Phase 1.7: Visual servo alignment ───────────────────────
        if SERVO_ENABLED and use_full_pipeline and homed:
            self.get_logger().info(
                f"Phase 1.7: starting visual servo "
                f"(align {port_type} port → reference pixel → stabilize → descend)")
            converged = self._visual_servo_align(
                get_observation, move_robot,
                port_type, target_port_index, send_feedback,
                ref_pixel=SERVO_STAGE1_REF_PIXEL,
                stage_name=f"align {port_type} port",
                target_module_name=getattr(task, "target_module_name", None),
                registered_target_world=getattr(
                    self, "_registered_target_world", None))
            self.get_logger().info(
                f"Phase 1.7: servo done (converged={converged}). "
                f"Stabilizing {SERVO_STAGE1_STABILIZE_S}s...")
            send_feedback(f"Servo done, stabilizing {SERVO_STAGE1_STABILIZE_S}s")
            self.sleep_for(SERVO_STAGE1_STABILIZE_S)
            self.get_logger().info(
                f"Phase 1.7: pre-descent settle {PRE_DESCENT_STABILIZE_S}s")
            self.sleep_for(PRE_DESCENT_STABILIZE_S)
        else:
            converged = False
        # ── Phase 2: ACT (or direct descent if ACT_MODE == "skip") ───
        if ACT_MODE == "skip" and use_full_pipeline:
            self.get_logger().info(
                f"Phase 2: ACT_MODE=skip — direct forced descent on {port_type}")
            send_feedback(f"Phase 2: direct descent ({port_type})")
            tcp = self._get_tcp_pose()
            if tcp is not None:
                cx, cy, cz = tcp[0], tcp[1], tcp[2]
                qx, qy, qz, qw = tcp[3], tcp[4], tcp[5], tcp[6]
                self._forced_descent(move_robot, cx, cy, cz, qx, qy, qz, qw)
                last_z = self._get_tcp_z()
                if last_z is None:
                    last_z = cz
            else:
                last_z = 0.24
            success = last_z < 0.20
            n_steps = 0
        else:
            success, last_z, n_steps = self._act_phase(
                task, get_observation, move_robot, send_feedback,
                port_type, target_port_index)
        self.get_logger().info(f"Pausing {POST_ACT_PAUSE}s after ACT")
        self.sleep_for(POST_ACT_PAUSE)
        # ── Phase 3: Fallback if descent / ACT stalled ──────────────
        if use_full_pipeline and not success:
            self.get_logger().info(
                f"Phase 2 stalled at Z={last_z:.4f} — engaging Phase 3 recovery")
            send_feedback("Stalled, recovering")
            self._post_act_recovery(move_robot, last_z, port_type)
            self.sleep_for(POST_FALLBACK_PAUSE)
        self.get_logger().info(
            f"Trial complete (steps={n_steps}) — "
            f"holding {POST_TRIAL_PAUSE}s before returning")
        self.sleep_for(POST_TRIAL_PAUSE)
        return True
