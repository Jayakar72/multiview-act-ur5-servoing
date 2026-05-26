#
# ATOMIC Team — AI for Industry Challenge
# CheatCode Demo Recorder for ACT Training
#
# Wraps CheatCode's insert_cable() via callback interception.
# Every move_robot() call is intercepted: observation is captured
# immediately before, action is extracted from the MotionUpdate,
# and both are appended to the episode buffer.
#
# At episode end the buffer is written to:
#   ~/aic_demos/episode_NNNN.h5
#
# HDF5 schema (per episode):
#   obs/images/left    (N, H, W, 3)  uint8
#   obs/images/center  (N, H, W, 3)  uint8
#   obs/images/right   (N, H, W, 3)  uint8
#   obs/tcp_pose       (N, 7)        float32  [x,y,z, qx,qy,qz,qw]
#   obs/wrench         (N, 6)        float32  [fx,fy,fz, tx,ty,tz]
#   actions/tcp_pose   (N, 7)        float32  [x,y,z, qx,qy,qz,qw]
#   metadata/          (attrs)       port_name, plug_name, module_name,
#                                    port_type, port_index, n_steps,
#                                    image_h, image_w, noise_mm
#
# USAGE:
#   Terminal 1 — simulator with ground truth ON:
#     /entrypoint.sh ground_truth:=true start_aic_engine:=true
#
#   Terminal 2 — recorder:
#     cd ~/ws_aic/src/aic
#     pixi run ros2 run aic_model aic_model \
#       --ros-args -p use_sim_time:=true \
#       -p policy:=aic_example_policies.ros.AtomicRecorder
#
#   Restart the simulator between runs to get different board configs.
#   Each simulator run gives 3 episodes (one per trial).
#
# NOISE:
#   Set env var RECORDER_NOISE_MM to perturb starting pose.
#   Default 3.0 mm. Set to 0 for clean reference episodes.
#

import glob
import os
import time

import cv2
import h5py
import numpy as np

from aic_control_interfaces.msg import MotionUpdate, TrajectoryGenerationMode
from aic_example_policies.ros.CheatCode import CheatCode
from aic_model.policy import (
    GetObservationCallback,
    MoveRobotCallback,
    Policy,
    SendFeedbackCallback,
)
from aic_task_interfaces.msg import Task
from geometry_msgs.msg import Point, Pose, Quaternion
from rclpy.time import Time
from tf2_ros import TransformException


# ── Configuration ─────────────────────────────────────────────────────────────

IMG_H = 360
IMG_W = 480

RECORDER_NOISE_MM = float(os.environ.get("RECORDER_NOISE_MM", "3.0"))

DEMO_DIR = os.path.expanduser("~/aic_demos")


# ── Helpers ───────────────────────────────────────────────────────────────────

def _img_msg_to_numpy(img_msg, h=IMG_H, w=IMG_W):
    raw = np.frombuffer(img_msg.data, dtype=np.uint8).reshape(
        img_msg.height, img_msg.width, 3
    )
    if (img_msg.height, img_msg.width) != (h, w):
        raw = cv2.resize(raw, (w, h), interpolation=cv2.INTER_AREA)
    return raw


def _pose_to_vec(pose: Pose) -> np.ndarray:
    p = pose.position
    q = pose.orientation
    return np.array([p.x, p.y, p.z, q.x, q.y, q.z, q.w], dtype=np.float32)


def _next_episode_path(demo_dir: str) -> str:
    os.makedirs(demo_dir, exist_ok=True)
    existing = sorted(glob.glob(os.path.join(demo_dir, "episode_*.h5")))
    return os.path.join(demo_dir, f"episode_{len(existing):04d}.h5")


def _parse_port_meta(port_name: str):
    parts = port_name.lower().split("_")
    port_type = parts[0]
    port_index = 0
    for part in reversed(parts):
        if part.isdigit():
            port_index = int(part)
            break
    return port_type, port_index


# ── Episode buffer ─────────────────────────────────────────────────────────────

class EpisodeBuffer:

    def __init__(self):
        self.left_imgs   = []
        self.center_imgs = []
        self.right_imgs  = []
        self.tcp_poses   = []
        self.wrenches    = []
        self.act_poses   = []

    def append(self, obs_dict, tcp_vec, wrench_vec, action_vec):
        self.left_imgs.append(obs_dict["left"])
        self.center_imgs.append(obs_dict["center"])
        self.right_imgs.append(obs_dict["right"])
        self.tcp_poses.append(tcp_vec)
        self.wrenches.append(wrench_vec)
        self.act_poses.append(action_vec)

    def __len__(self):
        return len(self.act_poses)

    def save(self, path: str, task: Task):
        port_type, port_index = _parse_port_meta(task.port_name)

        with h5py.File(path, "w") as f:
            obs  = f.create_group("obs")
            imgs = obs.create_group("images")
            imgs.create_dataset("left",   data=np.stack(self.left_imgs),
                                compression="gzip", compression_opts=4)
            imgs.create_dataset("center", data=np.stack(self.center_imgs),
                                compression="gzip", compression_opts=4)
            imgs.create_dataset("right",  data=np.stack(self.right_imgs),
                                compression="gzip", compression_opts=4)
            obs.create_dataset("tcp_pose", data=np.stack(self.tcp_poses))
            obs.create_dataset("wrench",   data=np.stack(self.wrenches))

            act = f.create_group("actions")
            act.create_dataset("tcp_pose", data=np.stack(self.act_poses))

            meta = f.create_group("metadata")
            meta.attrs["port_name"]   = task.port_name
            meta.attrs["plug_name"]   = task.plug_name
            meta.attrs["module_name"] = task.target_module_name
            meta.attrs["port_type"]   = port_type
            meta.attrs["port_index"]  = port_index
            meta.attrs["n_steps"]     = len(self)
            meta.attrs["image_h"]     = IMG_H
            meta.attrs["image_w"]     = IMG_W
            meta.attrs["noise_mm"]    = RECORDER_NOISE_MM


# ── Policy ────────────────────────────────────────────────────────────────────

class AtomicRecorder(Policy):
    """
    AIC policy that runs CheatCode and records every (obs, action) step.
    Requires ground_truth:=true in Terminal 1.
    """

    def __init__(self, parent_node):
        super().__init__(parent_node)
        self.cheatcode   = CheatCode(parent_node)
        self.trial_count = 0
        os.makedirs(DEMO_DIR, exist_ok=True)
        self.get_logger().info(
            f"AtomicRecorder init  |  "
            f"save={DEMO_DIR}  noise={RECORDER_NOISE_MM:.1f}mm  "
            f"img={IMG_W}x{IMG_H}"
        )

    def _get_tcp_pose(self):
        try:
            tf = self._parent_node._tf_buffer.lookup_transform(
                "base_link", "gripper/tcp", Time()
            )
            t = tf.transform.translation
            r = tf.transform.rotation
            return Pose(
                position=Point(x=t.x, y=t.y, z=t.z),
                orientation=Quaternion(x=r.x, y=r.y, z=r.z, w=r.w),
            )
        except TransformException:
            return None

    def _obs_to_dict(self, obs) -> dict:
        tcp = self._get_tcp_pose()
        tcp_vec = _pose_to_vec(tcp) if tcp is not None else np.zeros(7, dtype=np.float32)

        f = obs.wrist_wrench.wrench.force
        t = obs.wrist_wrench.wrench.torque
        wrench_vec = np.array([f.x, f.y, f.z, t.x, t.y, t.z], dtype=np.float32)

        return {
            "left":   _img_msg_to_numpy(obs.left_image),
            "center": _img_msg_to_numpy(obs.center_image),
            "right":  _img_msg_to_numpy(obs.right_image),
            "tcp":    tcp_vec,
            "wrench": wrench_vec,
        }

    def _apply_start_noise(self, move_robot_fn):
        if RECORDER_NOISE_MM <= 0.0:
            return

        tcp = self._get_tcp_pose()
        if tcp is None:
            self.get_logger().warn("Cannot read TCP pose — skipping noise")
            return

        noise_m = RECORDER_NOISE_MM * 1e-3
        dx, dy  = np.random.default_rng().normal(0, noise_m, size=2)

        noisy_pose = Pose(
            position=Point(
                x=tcp.position.x + dx,
                y=tcp.position.y + dy,
                z=tcp.position.z,
            ),
            orientation=tcp.orientation,
        )

        msg = MotionUpdate()
        msg.header.frame_id = "base_link"
        msg.pose = noisy_pose
        msg.target_stiffness = [
            90.0, 0, 0, 0, 0, 0,
            0, 90.0, 0, 0, 0, 0,
            0, 0, 90.0, 0, 0, 0,
            0, 0, 0, 50.0, 0, 0,
            0, 0, 0, 0, 50.0, 0,
            0, 0, 0, 0, 0, 50.0,
        ]
        msg.target_damping = [
            50.0, 0, 0, 0, 0, 0,
            0, 50.0, 0, 0, 0, 0,
            0, 0, 50.0, 0, 0, 0,
            0, 0, 0, 20.0, 0, 0,
            0, 0, 0, 0, 20.0, 0,
            0, 0, 0, 0, 0, 20.0,
        ]
        msg.trajectory_generation_mode.mode = TrajectoryGenerationMode.MODE_POSITION

        move_robot_fn(motion_update=msg)
        time.sleep(0.3)

        self.get_logger().info(
            f"  Noise applied: dx={dx*1e3:+.1f}mm  dy={dy*1e3:+.1f}mm"
        )

    def insert_cable(
        self,
        task: Task,
        get_observation: GetObservationCallback,
        move_robot: MoveRobotCallback,
        send_feedback: SendFeedbackCallback,
    ):
        self.trial_count += 1
        port_type, port_index = _parse_port_meta(task.port_name)
        ep_path = _next_episode_path(DEMO_DIR)

        self.get_logger().info(
            f"─── AtomicRecorder trial {self.trial_count} ───\n"
            f"  plug={task.plug_name}  port={task.port_name}  "
            f"type={port_type}  idx={port_index}\n"
            f"  saving → {ep_path}"
        )

        buf = EpisodeBuffer()

        # nudge starting pose for trajectory diversity
        self._apply_start_noise(move_robot)

        # ── wrapped callbacks ─────────────────────────────────────────────

        def recording_get_observation():
            return get_observation()

        def recording_move_robot(motion_update: MotionUpdate = None, **kwargs):
            """
            Intercept CheatCode's move_robot calls.
            CheatCode calls move_robot(motion_update=msg) as a keyword arg
            so we must accept that signature.
            """
            msg = motion_update

            if msg is not None and hasattr(msg, "pose") and msg.pose is not None:
                obs      = get_observation()
                obs_dict = self._obs_to_dict(obs)
                act_vec  = _pose_to_vec(msg.pose)
                buf.append(
                    obs_dict   = obs_dict,
                    tcp_vec    = obs_dict["tcp"],
                    wrench_vec = obs_dict["wrench"],
                    action_vec = act_vec,
                )

            # pass through to the real controller — use keyword arg to match CheatCode
            return move_robot(motion_update=msg)

        # ── run CheatCode ─────────────────────────────────────────────────

        result = self.cheatcode.insert_cable(
            task,
            recording_get_observation,
            recording_move_robot,
            send_feedback,
        )

        # ── save ──────────────────────────────────────────────────────────

        n = len(buf)
        if n > 0:
            buf.save(ep_path, task)
            self.get_logger().info(
                f"  ✓ Saved {n} steps → {ep_path}"
            )
        else:
            self.get_logger().warn(
                "  No steps recorded.\n"
                "  Make sure Terminal 1 uses ground_truth:=true"
            )

        all_eps = sorted(glob.glob(os.path.join(DEMO_DIR, "episode_*.h5")))
        self.get_logger().info(
            f"  Total episodes on disk: {len(all_eps)}"
        )

        return result
