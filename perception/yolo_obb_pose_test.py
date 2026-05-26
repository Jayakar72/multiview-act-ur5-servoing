#!/usr/bin/env python3
"""
yolo_obb_pose_test.py — Stage C verification node.

For each YOLO-OBB detection:
  1. Solve PnP on the 4 OBB corners (using known slot dimensions in 3D)
  2. Transform the resulting pose to base_link via TF
  3. Look up the ground-truth port pose via TF (requires ground_truth:=true)
  4. Match detection to closest GT port (by class + image projection)
  5. Log per-detection deltas (mm, deg) to a CSV
  6. Print summary statistics on shutdown

NO robot commands sent. Run alongside CheatCode driving a real trial.

Usage:
    Same 3-terminal recipe as Stage B. Terminal 1 MUST use ground_truth:=true.

    Terminal 1 (eval container):
        distrobox enter -r aic_eval -- /entrypoint.sh \\
            ground_truth:=true start_aic_engine:=true

    Terminal 2 (CheatCode policy):
        cd ~/ws_aic_new/src/aic
        pixi run ros2 run aic_model aic_model --ros-args \\
            -p use_sim_time:=true \\
            -p policy:=aic_example_policies.ros.CheatCode

    Terminal 3 (this script):
        cd ~/ws_aic_new/src/aic
        pixi run python3 scripts/yolo_obb_pose_test.py

Output (default ~/aic_yolo_test_poses/):
    {left,center,right}/<cam>_NNNNN.jpg   — annotated frames with deltas
    pose_log.csv                          — per-detection deltas
    summary.txt                           — aggregate statistics
"""

import argparse
import csv
import math
import os
import time
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.time import Time
from rclpy.duration import Duration
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy
from sensor_msgs.msg import Image, CameraInfo
from tf2_ros import Buffer, TransformListener, TransformException


# ── Slot 3D dimensions in port_link_entrance frame (Z=0 = opening face) ─

# SFP slot is a rectangle 13.4mm × 8.4mm
SFP_W, SFP_H = 0.0134, 0.0084
# SC slot is roughly square ~10.2mm
SC_W, SC_H = 0.0102, 0.0102

CLASS_NAMES  = {0: "sfp", 1: "sc"}
CLASS_COLORS = {0: (0, 255, 0), 1: (0, 165, 255)}

# Yaw period for symmetry handling:
#   SFP rectangle is symmetric under 180° rotation → period π
#   SC square is symmetric under 90° rotation → period π/2
YAW_PERIOD_BY_CLASS = {0: math.pi, 1: math.pi / 2}


def make_corners_3d(w, h):
    """3D corners CCW starting from bottom-left, in port_link_entrance frame."""
    return np.array([
        [-w / 2, -h / 2, 0.0],   # BL
        [+w / 2, -h / 2, 0.0],   # BR
        [+w / 2, +h / 2, 0.0],   # TR
        [-w / 2, +h / 2, 0.0],   # TL
    ], dtype=np.float64)


CORNERS_3D = {0: make_corners_3d(SFP_W, SFP_H),
              1: make_corners_3d(SC_W, SC_H)}


# ── Helpers ──────────────────────────────────────────────────────────────

def imgmsg_to_bgr(msg):
    if msg.encoding == "rgb8":
        arr = np.frombuffer(msg.data, dtype=np.uint8).reshape(
            msg.height, msg.width, 3)
        return cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)
    elif msg.encoding == "bgr8":
        return np.frombuffer(msg.data, dtype=np.uint8).reshape(
            msg.height, msg.width, 3).copy()
    raise ValueError(f"Unsupported encoding: {msg.encoding}")


def quat_to_rot(qx, qy, qz, qw):
    return np.array([
        [1-2*qy*qy-2*qz*qz, 2*qx*qy-2*qz*qw,   2*qx*qz+2*qy*qw],
        [2*qx*qy+2*qz*qw,   1-2*qx*qx-2*qz*qz, 2*qy*qz-2*qx*qw],
        [2*qx*qz-2*qy*qw,   2*qy*qz+2*qx*qw,   1-2*qx*qx-2*qy*qy],
    ])


def transform_to_T(t):
    T = np.eye(4)
    T[:3, :3] = quat_to_rot(t.rotation.x, t.rotation.y, t.rotation.z, t.rotation.w)
    T[:3, 3] = [t.translation.x, t.translation.y, t.translation.z]
    return T


def rot_to_yaw(R):
    """Yaw around world Z (radians)."""
    return math.atan2(R[1, 0], R[0, 0])


def wrap_yaw_diff(diff, period):
    """Wrap a yaw difference into [-period/2, +period/2]."""
    while diff > period / 2:
        diff -= period
    while diff < -period / 2:
        diff += period
    return diff


def solve_pnp_best_perm(corners_3d, corners_2d, K, dist):
    """Try all 4 rotational permutations of the 2D corners → pick min reproj err.
    Returns (rvec, tvec, reproj_err_px, best_perm) or None."""
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


def percentile(values, p):
    if not values:
        return float("nan")
    arr = np.asarray(values, dtype=np.float64)
    return float(np.percentile(arr, p))


# ── Node ─────────────────────────────────────────────────────────────────

class YoloPoseTester(Node):
    def __init__(self, weights, out_dir, save_every, max_frames, conf):
        super().__init__("yolo_obb_pose_tester")

        from ultralytics import YOLO

        self.out_dir = Path(out_dir)
        self.out_dir.mkdir(parents=True, exist_ok=True)
        for sub in ("left", "center", "right"):
            (self.out_dir / sub).mkdir(exist_ok=True)

        self.save_every = max(1, int(save_every))
        self.max_frames = int(max_frames) if max_frames else None
        self.conf       = float(conf)

        self.frame_counts     = {"left": 0, "center": 0, "right": 0}
        self.saved_counts     = {"left": 0, "center": 0, "right": 0}
        self.detection_counts = {"left": 0, "center": 0, "right": 0}

        self.K_by_cam    = {}    # cam → {K, w, h, d}
        self.port_frames = []    # [(frame_name, class_id), ...]

        self.tf_buffer = Buffer(cache_time=Duration(seconds=5.0))
        self.tf_listener = TransformListener(self.tf_buffer, self)

        self.get_logger().info(f"Loading YOLO from: {weights}")
        self.model = YOLO(weights)

        # ── CSV ───────────────────────────────────────────────────────
        self.csv_path = self.out_dir / "pose_log.csv"
        self.csv_file = open(self.csv_path, "w", newline="")
        self.csv_writer = csv.writer(self.csv_file)
        self.csv_writer.writerow([
            "t_sec", "camera", "class", "conf",
            "matched_port",
            "pnp_x", "pnp_y", "pnp_z", "pnp_yaw_deg",
            "gt_x",  "gt_y",  "gt_z",  "gt_yaw_deg",
            "dx_mm", "dy_mm", "dz_mm", "d_mag_mm",
            "dyaw_deg",
            "reproj_err_px", "rotation_perm",
            "cam_to_port_cm",
            "match_proj_dist_px",
        ])

        # In-memory log for summary stats
        self.records = []

        # ── Subscriptions ─────────────────────────────────────────────
        qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST, depth=1)
        for cam in ("left", "center", "right"):
            self.create_subscription(
                Image, f"/{cam}_camera/image",
                lambda msg, c=cam: self._on_image(msg, c), qos)
            self.create_subscription(
                CameraInfo, f"/{cam}_camera/camera_info",
                lambda msg, c=cam: self._on_camera_info(msg, c), 10)

        self.create_timer(2.0, self._discover_port_frames)
        self.create_timer(5.0, self._log_status)

        self.get_logger().info(
            f"out_dir={self.out_dir}\n"
            f"  CSV={self.csv_path}\n"
            f"  conf={self.conf}, save_every={self.save_every}, "
            f"max_frames={self.max_frames}")

    # ── Discovery ─────────────────────────────────────────────────────────

    def _on_camera_info(self, msg, cam):
        if cam not in self.K_by_cam:
            K = np.array(msg.k, dtype=np.float64).reshape(3, 3)
            d = np.array(msg.d, dtype=np.float64) if len(msg.d) else np.zeros(5)
            self.K_by_cam[cam] = {"K": K, "w": msg.width, "h": msg.height, "d": d}
            self.get_logger().info(
                f"[{cam}] intrinsics: fx={K[0,0]:.1f} cx={K[0,2]:.1f} "
                f"{msg.width}x{msg.height}")

    def _discover_port_frames(self):
        try:
            frames_str = self.tf_buffer.all_frames_as_string()
        except Exception:
            return
        new_ports, seen = [], set()
        for line in frames_str.split("\n"):
            parts = line.split()
            if len(parts) < 2:
                continue
            f = parts[1]
            if f in seen:
                continue
            if "sfp_port" in f and "entrance" in f:
                new_ports.append((f, 0))
                seen.add(f)
            elif "sc_port" in f and "entrance" in f:
                new_ports.append((f, 1))
                seen.add(f)
        if len(new_ports) != len(self.port_frames):
            self.port_frames = new_ports
            self.get_logger().info(
                f"port frames discovered ({len(self.port_frames)}): "
                f"{[p[0] for p in self.port_frames]}")

    def _log_status(self):
        seen  = sum(self.frame_counts.values())
        saved = sum(self.saved_counts.values())
        det   = sum(self.detection_counts.values())
        self.get_logger().info(
            f"received={seen} saved={saved} detections={det} | "
            f"L={self.saved_counts['left']} "
            f"C={self.saved_counts['center']} "
            f"R={self.saved_counts['right']} | "
            f"records={len(self.records)} ports={len(self.port_frames)}")

    # ── Image handler ─────────────────────────────────────────────────────

    def _on_image(self, msg, cam):
        if self.max_frames and self.saved_counts[cam] >= self.max_frames:
            return
        idx = self.frame_counts[cam]
        self.frame_counts[cam] += 1
        if idx % self.save_every != 0:
            return
        if cam not in self.K_by_cam:
            return
        if not self.port_frames:
            return

        try:
            bgr = imgmsg_to_bgr(msg)
        except Exception as e:
            self.get_logger().warn(f"[{cam}] decode failed: {e}")
            return

        K = self.K_by_cam[cam]["K"]
        D = self.K_by_cam[cam]["d"]
        cam_frame = f"{cam}_camera/optical"
        stamp = Time.from_msg(msg.header.stamp)

        # YOLO inference
        results = self.model.predict(bgr, conf=self.conf,
                                     verbose=False, imgsz=640)
        if not results:
            return
        result = results[0]
        n_det = 0 if result.obb is None else len(result.obb)
        if n_det > 0:
            self.detection_counts[cam] += 1

        # Camera pose in base
        try:
            tf_bc = self.tf_buffer.lookup_transform(
                "base_link", cam_frame, stamp,
                timeout=Duration(seconds=0.05))
        except TransformException:
            return
        T_base_cam = transform_to_T(tf_bc.transform)

        # Lookup all GT port poses (in base) + project center to image
        gt_ports = []
        for port_frame, port_cls in self.port_frames:
            try:
                tf_bp = self.tf_buffer.lookup_transform(
                    "base_link", port_frame, stamp,
                    timeout=Duration(seconds=0.05))
            except TransformException:
                continue
            T_base_port = transform_to_T(tf_bp.transform)
            gx, gy, gz = T_base_port[:3, 3]
            gyaw = rot_to_yaw(T_base_port[:3, :3])

            # project center to image
            T_cam_port = np.linalg.inv(T_base_cam) @ T_base_port
            X, Y, Z = T_cam_port[:3, 3]
            uv = None
            if Z > 0.01:
                u = K[0, 0] * X / Z + K[0, 2]
                v = K[1, 1] * Y / Z + K[1, 2]
                uv = (u, v)
            gt_ports.append((port_frame, port_cls, gx, gy, gz, gyaw, uv))

        # Annotate base
        annotated = bgr.copy()
        for port_frame, _, _, _, _, _, uv in gt_ports:
            if uv is None:
                continue
            cv2.drawMarker(annotated, (int(uv[0]), int(uv[1])),
                           (255, 255, 0), cv2.MARKER_CROSS, 14, 2)

        # Process detections
        if n_det > 0:
            corners_arr = result.obb.xyxyxyxy.cpu().numpy()
            cls_arr     = result.obb.cls.cpu().numpy().astype(int)
            conf_arr    = result.obb.conf.cpu().numpy()

            for i in range(n_det):
                cls_id = int(cls_arr[i])
                conf   = float(conf_arr[i])
                corners_2d = corners_arr[i].astype(np.float64)
                color  = CLASS_COLORS.get(cls_id, (255, 255, 255))

                pnp_3d = CORNERS_3D[cls_id]
                pnp = solve_pnp_best_perm(pnp_3d, corners_2d, K, D)
                if pnp is None:
                    continue
                rvec, tvec, reproj_err, perm = pnp

                # PnP pose in camera frame → base
                R_cp, _ = cv2.Rodrigues(rvec)
                T_cam_port_pnp = np.eye(4)
                T_cam_port_pnp[:3, :3] = R_cp
                T_cam_port_pnp[:3, 3]  = tvec.flatten()
                T_base_port_pnp = T_base_cam @ T_cam_port_pnp
                px, py, pz = T_base_port_pnp[:3, 3]
                pyaw = rot_to_yaw(T_base_port_pnp[:3, :3])

                cam_dist = float(np.linalg.norm(tvec))

                # Match to closest GT port of same class (by image projection)
                det_center = corners_2d.mean(axis=0)
                best, best_d = None, float("inf")
                for entry in gt_ports:
                    pf, pc, gx, gy, gz, gy_yaw, uv = entry
                    if pc != cls_id or uv is None:
                        continue
                    d = math.hypot(uv[0] - det_center[0],
                                   uv[1] - det_center[1])
                    if d < best_d:
                        best_d = d
                        best = entry
                if best is None or best_d > 100.0:
                    continue
                pf, _, gx, gy, gz, gyaw, _ = best

                dx_mm = (px - gx) * 1000
                dy_mm = (py - gy) * 1000
                dz_mm = (pz - gz) * 1000
                d_mag = math.sqrt(dx_mm**2 + dy_mm**2 + dz_mm**2)

                period = YAW_PERIOD_BY_CLASS.get(cls_id, math.pi)
                dyaw   = wrap_yaw_diff(pyaw - gyaw, period)
                dyaw_deg = math.degrees(dyaw)

                t_sec = stamp.nanoseconds / 1e9
                row = [
                    f"{t_sec:.3f}", cam, CLASS_NAMES[cls_id], f"{conf:.3f}",
                    pf,
                    f"{px:.4f}", f"{py:.4f}", f"{pz:.4f}",
                    f"{math.degrees(pyaw):.2f}",
                    f"{gx:.4f}", f"{gy:.4f}", f"{gz:.4f}",
                    f"{math.degrees(gyaw):.2f}",
                    f"{dx_mm:.2f}", f"{dy_mm:.2f}", f"{dz_mm:.2f}",
                    f"{d_mag:.2f}",
                    f"{dyaw_deg:.2f}",
                    f"{reproj_err:.2f}", perm,
                    f"{cam_dist*100:.2f}",
                    f"{best_d:.1f}",
                ]
                self.csv_writer.writerow(row)
                self.records.append({
                    "camera": cam,
                    "class":  CLASS_NAMES[cls_id],
                    "conf":   conf,
                    "d_mag_mm": d_mag,
                    "dx_mm":  dx_mm, "dy_mm": dy_mm, "dz_mm": dz_mm,
                    "dyaw_deg": dyaw_deg,
                    "reproj_err_px": reproj_err,
                    "cam_to_port_cm": cam_dist * 100,
                })

                # annotate
                pts = corners_2d.astype(np.int32)
                cv2.polylines(annotated, [pts], True, color, 2)
                cv2.circle(annotated, tuple(pts[0]), 5, (255, 0, 255), -1)
                label = (f"{CLASS_NAMES[cls_id]} c={conf:.2f} "
                         f"d={d_mag:.1f}mm dy={dyaw_deg:+.1f}°")
                tx, ty = int(det_center[0]) - 90, int(det_center[1]) - 18
                ty = max(ty, 14)
                cv2.putText(annotated, label, (tx, ty),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1)

        self.csv_file.flush()

        seq = self.saved_counts[cam]
        out_path = self.out_dir / cam / f"{cam}_{seq:05d}.jpg"
        cv2.imwrite(str(out_path), annotated, [cv2.IMWRITE_JPEG_QUALITY, 90])
        self.saved_counts[cam] += 1

    # ── Summary ───────────────────────────────────────────────────────────

    def write_summary(self):
        try:
            self.csv_file.close()
        except Exception:
            pass

        if not self.records:
            self.get_logger().warn("No records logged — nothing to summarize.")
            return

        path = self.out_dir / "summary.txt"
        f = open(path, "w")

        def write(s=""):
            print(s)
            f.write(s + "\n")

        write("=" * 70)
        write(f"POSE VERIFICATION SUMMARY  (n={len(self.records)} detections)")
        write("=" * 70)

        # Group by class
        by_class = defaultdict(list)
        for r in self.records:
            by_class[r["class"]].append(r)

        for cls, rs in sorted(by_class.items()):
            d_mags = [r["d_mag_mm"] for r in rs]
            dxs    = [abs(r["dx_mm"]) for r in rs]
            dys    = [abs(r["dy_mm"]) for r in rs]
            dzs    = [abs(r["dz_mm"]) for r in rs]
            dyaws  = [abs(r["dyaw_deg"]) for r in rs]
            reprs  = [r["reproj_err_px"] for r in rs]
            confs  = [r["conf"] for r in rs]
            dists  = [r["cam_to_port_cm"] for r in rs]

            write()
            write(f"── {cls.upper()}  (n={len(rs)}) ─────────────────────")
            write(f"  conf:        median={percentile(confs,50):.2f}  "
                  f"min={min(confs):.2f}  max={max(confs):.2f}")
            write(f"  cam→port:    median={percentile(dists,50):.1f}cm  "
                  f"range=[{min(dists):.1f}, {max(dists):.1f}]cm")
            write(f"  reproj err:  median={percentile(reprs,50):.2f}px  "
                  f"p95={percentile(reprs,95):.2f}px")
            write(f"  position |Δ|: median={percentile(d_mags,50):.2f}mm  "
                  f"p75={percentile(d_mags,75):.2f}mm  "
                  f"p95={percentile(d_mags,95):.2f}mm  "
                  f"max={max(d_mags):.2f}mm")
            write(f"    |dx|:      median={percentile(dxs,50):.2f}mm  "
                  f"p95={percentile(dxs,95):.2f}mm")
            write(f"    |dy|:      median={percentile(dys,50):.2f}mm  "
                  f"p95={percentile(dys,95):.2f}mm")
            write(f"    |dz|:      median={percentile(dzs,50):.2f}mm  "
                  f"p95={percentile(dzs,95):.2f}mm")
            write(f"  |dyaw|:      median={percentile(dyaws,50):.2f}°  "
                  f"p95={percentile(dyaws,95):.2f}°  "
                  f"max={max(dyaws):.2f}°")

        # Per-camera breakdown
        write()
        write("── BY CAMERA ──────────────────────────────────────────")
        by_cam = defaultdict(list)
        for r in self.records:
            by_cam[r["camera"]].append(r)
        for cam in ("left", "center", "right"):
            rs = by_cam.get(cam, [])
            if not rs:
                write(f"  {cam:6s}: no detections")
                continue
            d_mags = [r["d_mag_mm"]   for r in rs]
            dyaws  = [abs(r["dyaw_deg"]) for r in rs]
            reprs  = [r["reproj_err_px"] for r in rs]
            write(f"  {cam:6s}: n={len(rs):4d}  "
                  f"|Δ| median={percentile(d_mags,50):.2f}mm  "
                  f"p95={percentile(d_mags,95):.2f}mm  "
                  f"|dyaw| p95={percentile(dyaws,95):.2f}°  "
                  f"reproj p95={percentile(reprs,95):.2f}px")

        # Position accuracy by camera→port distance bucket
        write()
        write("── BY DISTANCE TO PORT ────────────────────────────────")
        buckets = [(0, 10), (10, 20), (20, 30), (30, 50), (50, 200)]
        for lo, hi in buckets:
            rs = [r for r in self.records if lo <= r["cam_to_port_cm"] < hi]
            if not rs:
                continue
            d_mags = [r["d_mag_mm"] for r in rs]
            dyaws  = [abs(r["dyaw_deg"]) for r in rs]
            write(f"  {lo:3d}-{hi:3d}cm: n={len(rs):4d}  "
                  f"|Δ| median={percentile(d_mags,50):.2f}mm  "
                  f"p95={percentile(d_mags,95):.2f}mm  "
                  f"|dyaw| p95={percentile(dyaws,95):.2f}°")

        write()
        write("=" * 70)
        write(f"CSV: {self.csv_path}")
        write(f"Annotated images: {self.out_dir}/{{left,center,right}}/")
        write("=" * 70)
        f.close()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--weights", default=os.path.expanduser(
        "~/aic_yolo_runs/sfp_obb_v1/weights/last.pt"))
    p.add_argument("--out-dir", default=os.path.expanduser(
        "~/aic_yolo_test_poses"))
    p.add_argument("--save-every", type=int, default=5,
        help="Save every Nth frame (1=all)")
    p.add_argument("--max-frames", type=int, default=400,
        help="Max saved frames per camera (0=unlimited)")
    p.add_argument("--conf", type=float, default=0.4,
        help="YOLO confidence threshold (lower than Stage B for OOD coverage)")
    args = p.parse_args()

    rclpy.init()
    node = YoloPoseTester(args.weights, args.out_dir,
                          args.save_every, args.max_frames, args.conf)
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        try:
            node.write_summary()
        except Exception as e:
            node.get_logger().error(f"Summary failed: {e}")
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
