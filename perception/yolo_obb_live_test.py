#!/usr/bin/env python3
"""
yolo_obb_live_test.py — Stage B verification node.

Subscribes to all 3 wrist cameras, runs YOLO-OBB on every Nth frame,
saves annotated images to disk for offline review.

NO robot commands are sent. Run this alongside the eval container with
CheatCode driving the trial — it just observes.

Usage:
    # Default settings (saves every 5th frame, 300 max per camera, conf 0.5)
    python3 yolo_obb_live_test.py

    # Custom settings
    python3 yolo_obb_live_test.py \
        --weights ~/aic_yolo_runs/sfp_obb_v1/weights/best.pt \
        --out-dir ~/aic_yolo_test \
        --save-every 3 \
        --max-frames 500 \
        --conf 0.4

Workflow:
    Terminal 1 (eval container):
        distrobox enter -r aic_eval -- /entrypoint.sh \
            ground_truth:=true start_aic_engine:=true

    Terminal 2 (CheatCode policy):
        ros2 run aic_model aic_model --ros-args -p use_sim_time:=true \
            -p policy:=aic_example_policies.ros.CheatCode

    Terminal 3 (this script — runs anywhere with ROS2 + ultralytics):
        python3 yolo_obb_live_test.py

Output:
    ~/aic_yolo_test/{left,center,right}/{left,center,right}_NNNNN.jpg

Each annotated frame shows:
    - OBB rectangle (green=sfp, orange=sc)
    - First corner highlighted as magenta dot, second as yellow (shows ordering)
    - Long-axis line through box center (blue) — gripper's target alignment
    - Label: class, confidence, OBB angle in degrees
    - Inference FPS
"""

import argparse
import os
import time
from pathlib import Path

import cv2
import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy
from sensor_msgs.msg import Image


# ── Class colors (BGR) and names ──────────────────────────────────────────
CLASS_COLORS = {
    0: (0, 255, 0),    # sfp_slot — green
    1: (0, 165, 255),  # sc_slot  — orange
}
CLASS_NAMES = {0: "sfp", 1: "sc"}


def imgmsg_to_bgr(msg):
    """Convert sensor_msgs/Image to BGR ndarray. No cv_bridge dependency."""
    if msg.encoding == "rgb8":
        arr = np.frombuffer(msg.data, dtype=np.uint8).reshape(
            msg.height, msg.width, 3)
        return cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)
    elif msg.encoding == "bgr8":
        return np.frombuffer(msg.data, dtype=np.uint8).reshape(
            msg.height, msg.width, 3).copy()
    elif msg.encoding == "mono8":
        gray = np.frombuffer(msg.data, dtype=np.uint8).reshape(
            msg.height, msg.width)
        return cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    raise ValueError(f"Unsupported image encoding: {msg.encoding}")


def annotate(img_bgr, result, fps=None):
    """Draw all OBB detections on a copy of the image."""
    out = img_bgr.copy()

    if result.obb is None or len(result.obb) == 0:
        cv2.putText(out, "NO DETECTIONS", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
    else:
        corners_arr = result.obb.xyxyxyxy.cpu().numpy()  # (N, 4, 2)
        xywhr_arr   = result.obb.xywhr.cpu().numpy()     # (N, 5)
        cls_arr     = result.obb.cls.cpu().numpy().astype(int)
        conf_arr    = result.obb.conf.cpu().numpy()

        for i in range(len(corners_arr)):
            cls_id  = int(cls_arr[i])
            conf    = float(conf_arr[i])
            corners = corners_arr[i].astype(np.int32)
            color   = CLASS_COLORS.get(cls_id, (255, 255, 255))

            # OBB rectangle
            cv2.polylines(out, [corners], isClosed=True,
                          color=color, thickness=2)

            # Highlight corner ordering: corner 0 = magenta, corner 1 = yellow
            cv2.circle(out, tuple(corners[0]), 6, (255, 0, 255), -1)
            cv2.circle(out, tuple(corners[1]), 4, (0, 255, 255), -1)

            # Long axis line through center (the slot's "insertion direction")
            cx, cy, w, h, angle = xywhr_arr[i]
            long_half = max(w, h) / 2
            long_angle = angle if w >= h else angle + np.pi / 2
            x0 = int(cx - long_half * np.cos(long_angle))
            y0 = int(cy - long_half * np.sin(long_angle))
            x1 = int(cx + long_half * np.cos(long_angle))
            y1 = int(cy + long_half * np.sin(long_angle))
            cv2.line(out, (x0, y0), (x1, y1), (255, 0, 0), 2)

            # Label above the box
            label = (f"{CLASS_NAMES.get(cls_id, '?')} {conf:.2f} "
                     f"a={np.degrees(angle):.0f}d")
            ty = int(cy - long_half - 8)
            ty = max(ty, 14)
            tx = max(int(cx) - 60, 4)
            cv2.putText(out, label, (tx, ty),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2)

    if fps is not None:
        cv2.putText(out, f"YOLO {fps:.1f} FPS",
                    (10, out.shape[0] - 12),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    return out


class YoloLiveTester(Node):
    def __init__(self, weights_path, out_dir, save_every, max_frames, conf):
        super().__init__("yolo_obb_live_tester")

        # Heavy import here so node init doesn't block on it during ROS startup
        from ultralytics import YOLO

        self.out_dir = Path(out_dir)
        self.out_dir.mkdir(parents=True, exist_ok=True)
        for sub in ("left", "center", "right"):
            (self.out_dir / sub).mkdir(exist_ok=True)

        self.save_every = max(1, int(save_every))
        self.max_frames = int(max_frames) if max_frames else None
        self.conf       = float(conf)

        # per-camera counters
        self.frame_counts = {"left": 0, "center": 0, "right": 0}
        self.saved_counts = {"left": 0, "center": 0, "right": 0}
        self.det_counts   = {"left": 0, "center": 0, "right": 0}

        self.get_logger().info(f"Loading YOLO weights from: {weights_path}")
        self.model = YOLO(weights_path)
        self.get_logger().info(f"Model task={self.model.task}, "
                               f"classes={self.model.names}")
        self.get_logger().info(f"Saving annotated images to: {self.out_dir}")
        self.get_logger().info(f"save_every={self.save_every}  "
                               f"max_frames={self.max_frames}  conf={self.conf}")

        # Cameras publish with BEST_EFFORT
        qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST,
            depth=1,
        )

        self.create_subscription(
            Image, "/left_camera/image",
            lambda msg: self._on_image(msg, "left"), qos)
        self.create_subscription(
            Image, "/center_camera/image",
            lambda msg: self._on_image(msg, "center"), qos)
        self.create_subscription(
            Image, "/right_camera/image",
            lambda msg: self._on_image(msg, "right"), qos)

        self._last_log_t = time.time()
        self._last_log_total = 0
        self.create_timer(5.0, self._log_status)

    def _log_status(self):
        total_seen = sum(self.frame_counts.values())
        total_saved = sum(self.saved_counts.values())
        total_det = sum(self.det_counts.values())
        dt = time.time() - self._last_log_t
        recent_rate = (total_seen - self._last_log_total) / dt if dt > 0 else 0
        self._last_log_t = time.time()
        self._last_log_total = total_seen
        self.get_logger().info(
            f"received={total_seen}  saved={total_saved}  "
            f"with_detections={total_det}  recv_rate={recent_rate:.1f}Hz "
            f"| L={self.saved_counts['left']} "
            f"C={self.saved_counts['center']} "
            f"R={self.saved_counts['right']}")

    def _on_image(self, msg, cam):
        # cap reached for this camera?
        if self.max_frames and self.saved_counts[cam] >= self.max_frames:
            return

        idx = self.frame_counts[cam]
        self.frame_counts[cam] += 1
        if idx % self.save_every != 0:
            return

        try:
            bgr = imgmsg_to_bgr(msg)
        except Exception as e:
            self.get_logger().warn(f"[{cam}] decode failed: {e}")
            return

        t0 = time.time()
        results = self.model.predict(
            bgr, conf=self.conf, verbose=False, imgsz=640)
        dt = time.time() - t0
        fps = (1.0 / dt) if dt > 0 else 0.0

        if not results:
            return
        result = results[0]

        n_det = 0 if result.obb is None else len(result.obb)
        if n_det > 0:
            self.det_counts[cam] += 1

        annotated = annotate(bgr, result, fps=fps)
        seq = self.saved_counts[cam]
        out_path = self.out_dir / cam / f"{cam}_{seq:05d}.jpg"
        cv2.imwrite(str(out_path), annotated,
                    [cv2.IMWRITE_JPEG_QUALITY, 90])
        self.saved_counts[cam] += 1


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--weights", default=os.path.expanduser(
        "~/aic_yolo_runs/sfp_obb_v1/weights/best.pt"),
        help="Path to YOLOv8-OBB best.pt")
    p.add_argument("--out-dir", default=os.path.expanduser(
        "~/aic_yolo_test"),
        help="Where to write annotated images")
    p.add_argument("--save-every", type=int, default=5,
        help="Save every Nth frame per camera (1=all)")
    p.add_argument("--max-frames", type=int, default=300,
        help="Max saved frames per camera (0=unlimited)")
    p.add_argument("--conf", type=float, default=0.5,
        help="YOLO confidence threshold")
    args = p.parse_args()

    rclpy.init()
    node = YoloLiveTester(
        weights_path=args.weights,
        out_dir=args.out_dir,
        save_every=args.save_every,
        max_frames=args.max_frames,
        conf=args.conf,
    )
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info("Interrupted, shutting down.")
    finally:
        # Final stats
        total_seen = sum(node.frame_counts.values())
        total_saved = sum(node.saved_counts.values())
        total_det = sum(node.det_counts.values())
        node.get_logger().info(
            f"FINAL: received={total_seen} saved={total_saved} "
            f"with_detections={total_det}")
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
