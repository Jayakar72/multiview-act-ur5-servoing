#!/usr/bin/env python3
"""
collect_port_data.py — Auto-label SFP/SC port detection data using GT TF.

Runs as a ROS 2 node alongside the simulation. For each camera frame:
  1. Looks up GT pose of every SFP/SC port via /tf
  2. Projects the port's 4 corners (known 3D rectangle) onto the image
  3. Saves image + YOLOv8-OBB label + debug overlay

Output structure:
  <save_dir>/<run_id>/
    images/<base>.png           — raw camera image (used for training)
    labels/<base>.txt           — YOLO-OBB labels, normalized to [0,1]
    debug/<base>_debug.jpg      — overlay for visual inspection

YOLO-OBB label format (one line per visible port):
  class_id x1 y1 x2 y2 x3 y3 x4 y4    (all corners normalized [0,1])
  class_id 0 = sfp_slot, 1 = sc_slot

Usage (3 terminals):

  Terminal 1 — eval sim with GT enabled:
    distrobox enter -r aic_eval -- /entrypoint.sh \\
      ground_truth:=true \\
      start_aic_engine:=true \\
      aic_engine_config_file:=<path>/config_rail0.yaml

  Terminal 2 — CheatCode driver:
    cd ~/ws_aic_new/src/aic
    pixi run ros2 run aic_model aic_model --ros-args \\
      -p use_sim_time:=true \\
      -p policy:=aic_example_policies.ros.CheatCode

  Terminal 3 — this collector:
    cd ~/ws_aic_new/src/aic
    pixi run python collect_port_data.py --ros-args -p run_id:=rail0

Repeat for each config (rail0, rail1, rail2, rail3_4, sample_config).

Parameters:
  run_id    (str)   — subdirectory under save_dir; default 'default'
  save_dir  (str)   — output root; default '~/aic_yolo_data'
  subsample (int)   — save every Nth frame per camera; default 3
  classes   (str)   — 'sfp,sc' or 'sfp' or 'sc'; default 'sfp,sc'
  cameras   (str)   — 'left,center,right'; default 'left,center,right'
  save_debug (bool) — save annotated overlays; default true
  margin_px (float) — skip if any corner is within this px of edge; default 5
  min_size_px (float) — skip if 2D bbox is smaller than this; default 8
"""

import os
import time
import cv2
import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.time import Time
from rclpy.duration import Duration
from rclpy.qos import QoSProfile, QoSReliabilityPolicy, QoSHistoryPolicy
from sensor_msgs.msg import Image, CameraInfo
from tf2_ros import Buffer, TransformListener, TransformException


def imgmsg_to_bgr(msg):
    """Convert sensor_msgs/Image to a BGR uint8 numpy array, no cv_bridge."""
    h, w = msg.height, msg.width
    enc = msg.encoding.lower() if msg.encoding else ''
    buf = np.frombuffer(msg.data, dtype=np.uint8)
    if enc == 'bgr8':
        return buf.reshape(h, w, 3)
    if enc == 'rgb8':
        rgb = buf.reshape(h, w, 3)
        return cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    if enc in ('mono8', '8uc1'):
        gray = buf.reshape(h, w)
        return cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    if enc == 'bgra8':
        return cv2.cvtColor(buf.reshape(h, w, 4), cv2.COLOR_BGRA2BGR)
    if enc == 'rgba8':
        return cv2.cvtColor(buf.reshape(h, w, 4), cv2.COLOR_RGBA2BGR)
    # Last resort: try to interpret as 3-channel
    if buf.size == h * w * 3:
        return buf.reshape(h, w, 3)
    raise ValueError(f"Unsupported image encoding: {msg.encoding!r}")


# ── Slot dimensions (3D, in port_link_entrance frame, Z=0 = opening face) ────

# SFP slot: from task_board_description.md
SFP_W = 0.0134   # 13.4 mm (long axis)
SFP_H = 0.0084   # 8.4 mm  (short axis)

# SC slot: round connector, treated as a square ~10mm
SC_W = 0.0102
SC_H = 0.0102

CLASS_IDS = {'sfp': 0, 'sc': 1}


def make_corners_3d(w, h):
    """4 corners of a slot in port_link_entrance frame, ordered CCW from BL."""
    return np.array([
        [-w/2, -h/2, 0.0],   # bottom-left
        [+w/2, -h/2, 0.0],   # bottom-right
        [+w/2, +h/2, 0.0],   # top-right
        [-w/2, +h/2, 0.0],   # top-left
    ], dtype=np.float64)


def quat_to_rot(qx, qy, qz, qw):
    return np.array([
        [1-2*qy*qy-2*qz*qz,  2*qx*qy-2*qz*qw,    2*qx*qz+2*qy*qw],
        [2*qx*qy+2*qz*qw,    1-2*qx*qx-2*qz*qz,  2*qy*qz-2*qx*qw],
        [2*qx*qz-2*qy*qw,    2*qy*qz+2*qx*qw,    1-2*qx*qx-2*qy*qy],
    ])


def transform_to_T(transform):
    t = transform.translation
    r = transform.rotation
    T = np.eye(4)
    T[:3, :3] = quat_to_rot(r.x, r.y, r.z, r.w)
    T[:3, 3] = [t.x, t.y, t.z]
    return T


class PortDataCollector(Node):
    def __init__(self):
        super().__init__('port_data_collector')

        self.declare_parameter('run_id', 'default')
        self.declare_parameter(
            'save_dir', os.path.expanduser('~/aic_yolo_data'))
        self.declare_parameter('subsample', 3)
        self.declare_parameter('classes', 'sfp,sc')
        self.declare_parameter('cameras', 'left,center,right')
        self.declare_parameter('save_debug', True)
        self.declare_parameter('margin_px', 5.0)
        self.declare_parameter('min_size_px', 8.0)
        # Skip a port whose TF data is older than this. Prevents stale
        # transforms from a previous trial (after entity despawn) being
        # treated as live data.
        self.declare_parameter('max_tf_age_s', 0.5)

        # If running alongside a sim publishing /clock, the user passes
        # `--ros-args -p use_sim_time:=true`. We don't force it here; the
        # TF lookups below use Time() (latest available) which works
        # regardless.

        self.run_id    = self.get_parameter('run_id').value
        save_dir       = self.get_parameter('save_dir').value
        self.subsample = self.get_parameter('subsample').value
        self.save_debug = self.get_parameter('save_debug').value
        self.margin_px = self.get_parameter('margin_px').value
        self.min_size_px = self.get_parameter('min_size_px').value
        self.max_tf_age_s = self.get_parameter('max_tf_age_s').value

        self.classes = [c.strip() for c in self.get_parameter('classes').value.split(',')]
        self.cameras = [c.strip() for c in self.get_parameter('cameras').value.split(',')]

        run_dir = os.path.join(save_dir, self.run_id)
        self.images_dir = os.path.join(run_dir, 'images')
        self.labels_dir = os.path.join(run_dir, 'labels')
        self.debug_dir  = os.path.join(run_dir, 'debug')
        os.makedirs(self.images_dir, exist_ok=True)
        os.makedirs(self.labels_dir, exist_ok=True)
        if self.save_debug:
            os.makedirs(self.debug_dir, exist_ok=True)

        self.tf_buffer = Buffer(cache_time=Duration(seconds=10.0))
        self.tf_listener = TransformListener(self.tf_buffer, self)

        self.intrinsics = {}    # cam → {K, w, h}
        self.frame_count = {cam: 0 for cam in self.cameras}
        self.port_frames = []   # list of (frame_name, port_type, corners_3d)
        self.saved_count = 0
        self.skip_reasons = {
            'no_intrinsics': 0, 'no_ports': 0,
            'tf_fail': 0, 'tf_stale': 0,
            'behind_cam': 0, 'too_far': 0,
            'corner_negative_z': 0,
            'outside_image': 0, 'too_small': 0,
        }
        self.start_time = time.time()

        qos = QoSProfile(
            reliability=QoSReliabilityPolicy.BEST_EFFORT,
            history=QoSHistoryPolicy.KEEP_LAST, depth=5,
        )
        for cam in self.cameras:
            self.create_subscription(
                Image, f'/{cam}_camera/image',
                lambda msg, c=cam: self._image_cb(msg, c), qos)
            self.create_subscription(
                CameraInfo, f'/{cam}_camera/camera_info',
                lambda msg, c=cam: self._info_cb(msg, c), 10)

        self.create_timer(2.0, self._discover_ports)
        self.create_timer(15.0, self._print_stats)

        self.get_logger().info(
            "PortDataCollector started\n"
            f"  run_id     : {self.run_id}\n"
            f"  save dir   : {run_dir}\n"
            f"  classes    : {self.classes}\n"
            f"  cameras    : {self.cameras}\n"
            f"  subsample  : every {self.subsample} frames\n"
            f"  save debug : {self.save_debug}"
        )

    def _info_cb(self, msg, cam):
        if cam not in self.intrinsics:
            K = np.array(msg.k).reshape(3, 3)
            self.intrinsics[cam] = {'K': K, 'w': msg.width, 'h': msg.height}
            self.get_logger().info(
                f"Got intrinsics: {cam}, fx={K[0,0]:.1f}, "
                f"image={msg.width}x{msg.height}")

    def _discover_ports(self):
        try:
            frames_str = self.tf_buffer.all_frames_as_string()
        except Exception:
            return
        new_ports = []
        seen = set()
        for line in frames_str.split('\n'):
            parts = line.split()
            if len(parts) < 2:
                continue
            frame = parts[1]
            if frame in seen:
                continue
            if 'sfp' in self.classes and 'sfp_port' in frame and 'entrance' in frame:
                new_ports.append((frame, 'sfp', make_corners_3d(SFP_W, SFP_H)))
                seen.add(frame)
            elif 'sc' in self.classes and 'sc_port' in frame and 'entrance' in frame:
                new_ports.append((frame, 'sc', make_corners_3d(SC_W, SC_H)))
                seen.add(frame)
        if len(new_ports) != len(self.port_frames):
            self.port_frames = new_ports
            self.get_logger().info(
                f"Ports discovered ({len(self.port_frames)}): "
                f"{[p[0] for p in self.port_frames]}")

    def _print_stats(self):
        elapsed = time.time() - self.start_time
        rate = self.saved_count / max(elapsed, 0.1)
        self.get_logger().info(
            f"saved={self.saved_count}  rate={rate:.1f}/s  "
            f"frames_seen={sum(self.frame_count.values())}  "
            f"skips={self.skip_reasons}")

    def _image_cb(self, msg, cam):
        self.frame_count[cam] += 1
        if self.frame_count[cam] % self.subsample != 0:
            return

        if cam not in self.intrinsics:
            self.skip_reasons['no_intrinsics'] += 1
            return
        if not self.port_frames:
            self.skip_reasons['no_ports'] += 1
            return

        try:
            img = imgmsg_to_bgr(msg)
        except Exception as e:
            self.get_logger().warn(
                f"image conv failed: {e}", throttle_duration_sec=5.0)
            return

        cam_frame = f'{cam}_camera/optical'
        K = self.intrinsics[cam]['K']
        img_w = self.intrinsics[cam]['w']
        img_h = self.intrinsics[cam]['h']

        labels = []
        debug_img = img.copy() if self.save_debug else None

        for port_frame, port_type, corners_3d in self.port_frames:
            try:
                # Use Time() = latest available transform. This avoids
                # sim-time vs wall-clock mismatch issues.
                tf = self.tf_buffer.lookup_transform(
                    cam_frame, port_frame, Time())
            except TransformException as e:
                self.skip_reasons['tf_fail'] += 1
                if self.skip_reasons['tf_fail'] in (1, 50):
                    self.get_logger().warn(
                        f"TF lookup failed: {cam_frame} <- {port_frame}: {e}")
                continue

            # Reject stale TF: when entities despawn between trials, the
            # buffer keeps their last known transform for several seconds.
            # Using that produces "ports glued to gripper" labels. Skip if
            # the cached TF is older than max_tf_age_s.
            try:
                tf_time = Time.from_msg(tf.header.stamp)
                age_ns = (self.get_clock().now() - tf_time).nanoseconds
                age_s = age_ns / 1e9
                if age_s > self.max_tf_age_s:
                    self.skip_reasons['tf_stale'] += 1
                    continue
            except Exception:
                # If time math fails for any reason, fall through and
                # use the transform anyway (best effort).
                pass

            T = transform_to_T(tf.transform)
            depth = T[2, 3]
            if depth < 0.05:
                self.skip_reasons['behind_cam'] += 1
                continue
            if depth > 1.5:
                self.skip_reasons['too_far'] += 1
                continue

            # Transform 3D corners to camera frame
            corners_cam = (T[:3, :3] @ corners_3d.T).T + T[:3, 3]
            if (corners_cam[:, 2] <= 0.01).any():
                self.skip_reasons['corner_negative_z'] += 1
                continue

            # Project to image
            corners_img = np.zeros((4, 2), dtype=np.float64)
            for i in range(4):
                X, Y, Z = corners_cam[i]
                corners_img[i, 0] = K[0, 0] * X / Z + K[0, 2]
                corners_img[i, 1] = K[1, 1] * Y / Z + K[1, 2]

            mn = corners_img.min(axis=0)
            mx = corners_img.max(axis=0)
            if (mn[0] < self.margin_px or mn[1] < self.margin_px or
                mx[0] > img_w - self.margin_px or
                mx[1] > img_h - self.margin_px):
                self.skip_reasons['outside_image'] += 1
                continue

            box_w = mx[0] - mn[0]
            box_h = mx[1] - mn[1]
            if box_w < self.min_size_px or box_h < self.min_size_px:
                self.skip_reasons['too_small'] += 1
                continue

            class_id = CLASS_IDS[port_type]
            corners_norm = corners_img.copy()
            corners_norm[:, 0] /= img_w
            corners_norm[:, 1] /= img_h

            label_parts = [str(class_id)]
            for x, y in corners_norm:
                label_parts.append(f"{x:.6f}")
                label_parts.append(f"{y:.6f}")
            labels.append(' '.join(label_parts))

            if self.save_debug:
                pts = corners_img.astype(int)
                color = (0, 255, 0) if port_type == 'sfp' else (0, 165, 255)
                for i in range(4):
                    cv2.line(debug_img, tuple(pts[i]),
                             tuple(pts[(i+1) % 4]), color, 2)
                # Mark first corner
                cv2.circle(debug_img, tuple(pts[0]), 3, (255, 0, 0), -1)
                # Class label
                cv2.putText(debug_img, port_type,
                            (int(pts[0, 0]) - 5, int(pts[0, 1]) - 5),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)

        if not labels:
            return

        ts_ns = msg.header.stamp.sec * 1_000_000_000 + msg.header.stamp.nanosec
        base = f'{cam}_{ts_ns}'
        cv2.imwrite(os.path.join(self.images_dir, f'{base}.png'), img)
        with open(os.path.join(self.labels_dir, f'{base}.txt'), 'w') as f:
            f.write('\n'.join(labels) + '\n')
        if self.save_debug:
            cv2.imwrite(os.path.join(self.debug_dir, f'{base}_debug.jpg'),
                        debug_img,
                        [cv2.IMWRITE_JPEG_QUALITY, 75])
        self.saved_count += 1


def main():
    rclpy.init()
    node = PortDataCollector()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.get_logger().info(
            f"FINAL: saved={node.saved_count}  "
            f"to {node.images_dir}  "
            f"skips={node.skip_reasons}")
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
