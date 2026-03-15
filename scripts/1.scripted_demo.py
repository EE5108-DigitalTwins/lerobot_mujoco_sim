#!/usr/bin/env python
"""
9.scripted_demo.py — Automated pick-and-place demonstration generator.

Generates demonstrations programmatically using a Finite State Machine (FSM)
controller with IK-based waypoint planning. Output is saved directly to
LeRobotDataset format, identical to 1.collect_data.py, so the same training
pipeline (3.train.py, 7.pi0_deploy.py, …) works without modification.

FSM phases:
  approach  → swing EEF above the block
  descend   → lower to grasp height
  grasp     → close gripper around block
  lift      → raise block clear of table
  transport → swing arm over bin
  place     → lower block above bin opening
  release   → open gripper, let block fall in
  retract   → raise above bin walls (required by check_success)

Usage (run from project root):
    python scripts/9.scripted_demo.py --num-demo 100
    python scripts/9.scripted_demo.py --num-demo 1000 --no-render --seed 42
    python scripts/9.scripted_demo.py --num-demo 50 --steps-per-phase 15

Key tuning knobs (all exposed as CLI args):
    --approach-height   EEF above block centre for safe approach  (default 0.12 m)
    --grasp-height      EEF above block centre at close point     (default 0.01 m)
    --lift-height       EEF above block centre after grasping     (default 0.20 m)
    --place-height      EEF above bin body origin for release     (default 0.13 m)
    --retract-height    EEF above bin body origin after release   (default 0.25 m)
    --steps-per-phase   Control steps per motion segment          (default 20)
    --sim-substeps      Physics steps per control step            (default 25)
"""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import argparse
import shutil
import time
from dataclasses import dataclass

import numpy as np
from PIL import Image
import mujoco

from mujoco_env.y_env import SimpleEnv
from mujoco_env.ik import solve_ik
from mujoco_env.transforms import rpy2r
from so101_inverse_kinematics import get_inverse_kinematics
from lerobot.common.datasets.lerobot_dataset import LeRobotDataset

# ---------------------------------------------------------------------------
# Configuration dataclass
# ---------------------------------------------------------------------------

@dataclass
class ScriptedConfig:
    # Dataset
    seed: int | None = None
    repo_name: str = "so101_pnp_scripted"
    num_demo: int = 10
    root: str = str(PROJECT_ROOT / "data" / "demo_data_so101_scripted")
    task_name: str = "Put green block in the bin"
    xml_path: str = str(PROJECT_ROOT / "asset" / "scene_so101_y.xml")
    mug_body_name: str = "body_obj_block_3"
    plate_body_name: str = "body_obj_bin"
    env_robot_profile: str = "so101"
    robot_type: str = "so101"
    fps: int = 20
    image_size: int = 256
    image_writer_threads: int = 10
    image_writer_processes: int = 5
    cleanup_images: bool = True
    delete_existing_dataset: bool = False

    # Rendering
    render: bool = True

    # Motion timing
    # sim_substeps=25 matches the 20 Hz recording rate used in 1.collect_data.py
    # (MuJoCo default dt=0.002 s → 500 Hz simulation → 25 steps per 20 Hz frame).
    steps_per_phase: int = 40       # control steps per motion segment
    sim_substeps: int = 25          # physics steps per control step
    settle_steps: int = 50          # extra physics steps after gripper open/close
    max_ik_tick: int = 500          # IK solver iterations per waypoint
    ik_err_th: float = 0.015        # IK positional error threshold (metres)
    max_ik_err_skip: float = 0.05   # skip episode if any waypoint exceeds this
    max_retries: int = 3            # IK-failure retries before giving up

    # Height offsets for waypoints (metres)
    # approach / grasp / lift are relative to the block body-centre position.
    # place / retract are relative to the bin body-origin position.
    approach_height: float = 0.12
    grasp_height: float = 0.01
    lift_height: float = 0.20
    place_height: float = 0.13
    retract_height: float = 0.25

    # Spawn bounds (passed straight to SimpleEnv)
    # Defaults mirror configs/collect_data.yaml so scripted demos use the
    # same object spawn region as interactive data collection.
    spawn_x_min: float = 0.20
    spawn_x_max: float = 0.40
    spawn_y_min: float = 0.02
    spawn_y_max: float = 0.15
    spawn_z_min: float = 0.815
    spawn_z_max: float = 0.815
    # The interactive collector uses a tight tabletop area with four blocks and
    # a fixed bin. To avoid frequent sampling failures in this constrained
    # region, use more permissive minimum-distance settings than the original
    # SimpleEnv defaults.
    spawn_min_dist: float = 0.05
    spawn_xy_margin: float = 0.0
    spawn_fallback_min_dist: float = 0.03


# ---------------------------------------------------------------------------
# Argument parser
# ---------------------------------------------------------------------------

def parse_args() -> ScriptedConfig:
    d = ScriptedConfig()
    p = argparse.ArgumentParser(
        description="Scripted pick-and-place demonstration generator for SO-101.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--seed", type=int, default=d.seed)
    p.add_argument("--repo-name", default=d.repo_name)
    p.add_argument("--num-demo", type=int, default=d.num_demo,
                   help="Number of *successful* episodes to save.")
    p.add_argument("--root", default=d.root, help="Dataset save directory.")
    p.add_argument("--task-name", default=d.task_name)
    p.add_argument("--xml-path", default=d.xml_path)
    p.add_argument("--mug-body-name", default=d.mug_body_name)
    p.add_argument("--plate-body-name", default=d.plate_body_name)
    p.add_argument("--env-robot-profile", default=d.env_robot_profile,
                   choices=["so101", "so100", "omy"])
    p.add_argument("--robot-type", default=d.robot_type)
    p.add_argument("--fps", type=int, default=d.fps)
    p.add_argument("--image-size", type=int, default=d.image_size)
    p.add_argument("--image-writer-threads", type=int, default=d.image_writer_threads)
    p.add_argument("--image-writer-processes", type=int, default=d.image_writer_processes)
    p.add_argument("--cleanup-images", action=argparse.BooleanOptionalAction,
                   default=d.cleanup_images)
    p.add_argument("--delete-existing-dataset", action=argparse.BooleanOptionalAction,
                   default=d.delete_existing_dataset)
    p.add_argument("--render", action=argparse.BooleanOptionalAction, default=d.render,
                   help="Show the MuJoCo viewer. Disable with --no-render for faster generation.")
    p.add_argument("--steps-per-phase", type=int, default=d.steps_per_phase)
    p.add_argument("--sim-substeps", type=int, default=d.sim_substeps)
    p.add_argument("--settle-steps", type=int, default=d.settle_steps)
    p.add_argument("--max-ik-tick", type=int, default=d.max_ik_tick)
    p.add_argument("--ik-err-th", type=float, default=d.ik_err_th)
    p.add_argument("--max-ik-err-skip", type=float, default=d.max_ik_err_skip)
    p.add_argument("--max-retries", type=int, default=d.max_retries)
    p.add_argument("--approach-height", type=float, default=d.approach_height)
    p.add_argument("--grasp-height", type=float, default=d.grasp_height)
    p.add_argument("--lift-height", type=float, default=d.lift_height)
    p.add_argument("--place-height", type=float, default=d.place_height)
    p.add_argument("--retract-height", type=float, default=d.retract_height)
    p.add_argument("--spawn-x-min", type=float, default=d.spawn_x_min)
    p.add_argument("--spawn-x-max", type=float, default=d.spawn_x_max)
    p.add_argument("--spawn-y-min", type=float, default=d.spawn_y_min)
    p.add_argument("--spawn-y-max", type=float, default=d.spawn_y_max)
    p.add_argument("--spawn-z-min", type=float, default=d.spawn_z_min)
    p.add_argument("--spawn-z-max", type=float, default=d.spawn_z_max)
    p.add_argument("--spawn-min-dist", type=float, default=d.spawn_min_dist)
    p.add_argument("--spawn-xy-margin", type=float, default=d.spawn_xy_margin)
    p.add_argument("--spawn-fallback-min-dist", type=float, default=d.spawn_fallback_min_dist)

    a = p.parse_args()
    cfg = ScriptedConfig(
        seed=a.seed,
        repo_name=a.repo_name,
        num_demo=a.num_demo,
        root=a.root,
        task_name=a.task_name,
        xml_path=a.xml_path,
        mug_body_name=a.mug_body_name,
        plate_body_name=a.plate_body_name,
        env_robot_profile=a.env_robot_profile,
        robot_type=a.robot_type,
        fps=a.fps,
        image_size=a.image_size,
        image_writer_threads=a.image_writer_threads,
        image_writer_processes=a.image_writer_processes,
        cleanup_images=a.cleanup_images,
        delete_existing_dataset=a.delete_existing_dataset,
        render=a.render,
        steps_per_phase=a.steps_per_phase,
        sim_substeps=a.sim_substeps,
        settle_steps=a.settle_steps,
        max_ik_tick=a.max_ik_tick,
        ik_err_th=a.ik_err_th,
        max_ik_err_skip=a.max_ik_err_skip,
        max_retries=a.max_retries,
        approach_height=a.approach_height,
        grasp_height=a.grasp_height,
        lift_height=a.lift_height,
        place_height=a.place_height,
        retract_height=a.retract_height,
        spawn_x_min=a.spawn_x_min,
        spawn_x_max=a.spawn_x_max,
        spawn_y_min=a.spawn_y_min,
        spawn_y_max=a.spawn_y_max,
        spawn_z_min=a.spawn_z_min,
        spawn_z_max=a.spawn_z_max,
        spawn_min_dist=a.spawn_min_dist,
        spawn_xy_margin=a.spawn_xy_margin,
        spawn_fallback_min_dist=a.spawn_fallback_min_dist,
    )
    if not Path(cfg.xml_path).is_absolute():
        cfg.xml_path = str((PROJECT_ROOT / cfg.xml_path).resolve())
    return cfg


# ---------------------------------------------------------------------------
# Environment + dataset factories
# ---------------------------------------------------------------------------

def build_env(cfg: ScriptedConfig) -> SimpleEnv:
    """
    Create SimpleEnv using the same control mode conventions as 1.collect_data.py.

    SO-101 uses joint-space delta control for teleoperation; we mirror that here
    so that scripted trajectories exercise the same dynamics, while still
    storing absolute joint states in the dataset.
    """
    action_type = "delta_joint_angle" if cfg.env_robot_profile == "so101" else "joint_angle"
    return SimpleEnv(
        cfg.xml_path,
        action_type=action_type,
        robot_profile=cfg.env_robot_profile,
        seed=cfg.seed,
        state_type="joint_angle",
        mug_body_name=cfg.mug_body_name,
        plate_body_name=cfg.plate_body_name,
        spawn_x_range=(cfg.spawn_x_min, cfg.spawn_x_max),
        spawn_y_range=(cfg.spawn_y_min, cfg.spawn_y_max),
        spawn_z_range=(cfg.spawn_z_min, cfg.spawn_z_max),
        spawn_min_dist=cfg.spawn_min_dist,
        spawn_xy_margin=cfg.spawn_xy_margin,
        spawn_fallback_min_dist=cfg.spawn_fallback_min_dist,
    )


def build_dataset(cfg: ScriptedConfig) -> LeRobotDataset:
    """Create (or re-use) a LeRobotDataset compatible with 1.collect_data.py."""
    root_path = Path(cfg.root)
    if root_path.exists():
        if cfg.delete_existing_dataset:
            shutil.rmtree(root_path)
            print(f"[scripted_demo] Deleted existing dataset: {cfg.root}")
        else:
            # Avoid overwriting; find an unused suffix
            candidate = Path(f"{cfg.root}_scripted")
            idx = 0
            while candidate.exists():
                idx += 1
                candidate = Path(f"{cfg.root}_scripted_{idx}")
            cfg.root = str(candidate)
            print(f"[scripted_demo] Using new dataset root: {cfg.root}")

    action_dim = 7 if cfg.env_robot_profile == "omy" else 6
    return LeRobotDataset.create(
        repo_id=cfg.repo_name,
        root=cfg.root,
        robot_type=cfg.robot_type,
        fps=cfg.fps,
        features={
            "observation.image": {
                "dtype": "image",
                "shape": (cfg.image_size, cfg.image_size, 3),
                "names": ["height", "width", "channels"],
            },
            "observation.wrist_image": {
                "dtype": "image",
                "shape": (cfg.image_size, cfg.image_size, 3),
                "names": ["height", "width", "channel"],
            },
            "observation.state": {
                "dtype": "float32",
                "shape": (6,),
                "names": ["state"],  # [x, y, z, roll, pitch, yaw]
            },
            "action": {
                "dtype": "float32",
                "shape": (action_dim,),
                "names": ["action"],  # absolute joint angles + gripper_norm
            },
            "obj_init": {
                "dtype": "float32",
                "shape": (6,),
                "names": ["obj_init"],  # [obj_xyz, bin_xyz] initial positions
            },
        },
        image_writer_threads=cfg.image_writer_threads,
        image_writer_processes=cfg.image_writer_processes,
    )


# ---------------------------------------------------------------------------
# Low-level helpers
# ---------------------------------------------------------------------------

GRIPPER_OPEN   = 0.0
GRIPPER_CLOSED = 1.0

# Grasping orientation: pitch 90° (Ry) — the SO-101 home/default orientation.
# With this rotation the arm extends forward (+X) and the gripper jaws face
# the approach direction.  Matches the IK target used in SimpleEnv.reset().
_GRASP_R = rpy2r(np.deg2rad([0.0, 90.0, 0.0]))


def _get_arm_q(env: SimpleEnv) -> np.ndarray:
    """Current arm joint angles (5-vector, no gripper)."""
    return env.env.get_qpos_joints(joint_names=env.joint_names).copy()


def _solve_wp(
    env: SimpleEnv,
    p_trgt: np.ndarray,
    R_trgt,
    q_init: np.ndarray,
    max_ik_tick: int,
    ik_err_th: float,
    p_tip_trgt: np.ndarray = None,
) -> tuple[np.ndarray, float]:
    """
    Solve IK for one waypoint.  Returns (q_solution, residual_error).

    For the SO-101 arm, prefer the geometric inverse kinematics from
    `so101_inverse_kinematics.py` (course-style).  For other robot profiles,
    fall back to the generic Jacobian-based IK solver.
    """
    # Geometric IK branch: SO-101 analytic solution using desired tip position.
    is_so101 = getattr(env, "robot_profile", None) == "so101"
    if is_so101:
        # p_tip_trgt is the desired gripperframe (jaw-tip) position in world coords.
        # The analytic IK expects tip, not palm.
        joint_cfg = get_inverse_kinematics(p_tip_trgt if p_tip_trgt is not None else p_trgt)
        q_geom = np.array(
            [
                joint_cfg["shoulder_pan"],
                joint_cfg["shoulder_lift"],
                joint_cfg["elbow_flex"],
                joint_cfg["wrist_flex"],
                joint_cfg["wrist_roll"],
            ],
            dtype=np.float32,
        )
        # Use the geometric IK as the q_init seed for numerical IK refinement.
        q_init = q_geom

    # SO-101 is 5-DOF and cannot track a full 3-D orientation target; passing
    # R_trgt would make the Jacobian solver minimise an unachievable rotation
    # error that dominates err_stack and inflates the residual metric used for
    # the skip check.  Use position-only IK for SO-101.
    ik_R = None if is_so101 else R_trgt

    # Default: numeric Jacobian-based IK.
    q, err_stack, _ = solve_ik(
        env=env.env,
        joint_names_for_ik=env.joint_names,
        body_name_trgt=env.tcp_body_name,
        q_init=q_init,
        p_trgt=p_trgt,
        R_trgt=ik_R,
        max_ik_tick=max_ik_tick,
        ik_err_th=ik_err_th,
        restore_state=True,
        verbose_warning=True,
    )
    # Return only position-error magnitude for the skip threshold check.
    # err_stack is 3-element (position only) when R_trgt is None.
    return q, float(np.linalg.norm(err_stack[:3]))


def _resize(img: np.ndarray, size: int) -> np.ndarray:
    return np.array(Image.fromarray(img).resize((size, size)))


# ---------------------------------------------------------------------------
# Episode buffer (accumulates frames; committed on success, discarded on fail)
# ---------------------------------------------------------------------------

class _EpisodeBuf:
    def __init__(self) -> None:
        self._frames: list[dict] = []

    def append(self, agent_img, wrist_img, ee_pose, joint_state):
        self._frames.append({
            "agent_img":   agent_img,
            "wrist_img":   wrist_img,
            "ee_pose":     ee_pose,
            "joint_state": joint_state,
        })

    def commit(self, dataset: LeRobotDataset, task: str, obj_init: np.ndarray):
        for f in self._frames:
            dataset.add_frame({
                "observation.image":       f["agent_img"],
                "observation.wrist_image": f["wrist_img"],
                "observation.state":       f["ee_pose"],
                "action":                  f["joint_state"],
                "obj_init":                obj_init,
            }, task=task)
        dataset.save_episode()
        self._frames.clear()

    def discard(self, dataset: LeRobotDataset):
        dataset.clear_episode_buffer()
        self._frames.clear()

    def __len__(self) -> int:
        return len(self._frames)


# ---------------------------------------------------------------------------
# FSM pick-and-place controller
# ---------------------------------------------------------------------------

class PickPlaceFSM:
    """
    Finite-state-machine scripted pick-and-place for the SO-101.

    Waypoints are computed via IK before the episode starts; phases are then
    executed by linearly interpolating joint angles.

    Height parameters (all in metres):
        approach_height / grasp_height / lift_height  — relative to p_obj[2]
        place_height / retract_height                  — relative to p_bin[2]

    The retract_height must put the gripper-frame site (jaw tip) above
    bin_wall_top + 0.015 m for check_success() to pass.
    Given gripperframe ≈ TCP_Z with the Ry(90°) orientation:
        retract_height > p_bin[2] + 0.076 + 0.015 - p_bin[2] = 0.091 m
    The default of 0.25 m provides ample margin.
    """

    def __init__(self, env: SimpleEnv, cfg: ScriptedConfig) -> None:
        self.env = env
        self.cfg = cfg
        # Local (TCP-frame) offset from the gripper palm ('gripper' body) to the
        # jaw-tip site ('gripperframe'). Populated lazily on first episode and
        # reused so that IK can target the palm while the tip aligns with the
        # desired block/bin positions under a fixed "face-down" orientation.
        self._tcp_tip_offset_local: np.ndarray | None = None

    # ------------------------------------------------------------------
    # Primitive motions
    # ------------------------------------------------------------------

    def _physics_steps(self, n: int):
        for _ in range(n):
            self.env.step_env()

    def _control_step(
        self,
        q_arm: np.ndarray,
        gripper: float,
        buf: _EpisodeBuf,
        record: bool = True,
    ):
        """Apply one control step, run physics, optionally record."""
        # For SO-101 we use delta-joint control (same as teleop).  Internally
        # the FSM still reasons in absolute joint space; we convert to deltas
        # here so SimpleEnv.step() integrates from its stored last_q.
        if self.env.action_type == "delta_joint_angle":
            # SimpleEnv.last_q holds the last commanded arm joints (no gripper).
            prev_q = getattr(self.env, "last_q", _get_arm_q(self.env))
            dq_arm = q_arm - prev_q
            action_arm = dq_arm
        else:
            action_arm = q_arm

        action = np.concatenate([action_arm, [gripper]], dtype=np.float32)
        joint_state = self.env.step(action)     # sets servo targets → returns actual qpos
        self._physics_steps(self.cfg.sim_substeps)

        if record:
            agent_raw, wrist_raw = self.env.grab_image()
            sz = self.cfg.image_size
            buf.append(
                _resize(agent_raw, sz),
                _resize(wrist_raw, sz),
                self.env.get_ee_pose(),
                joint_state,
            )
        if self.cfg.render:
            self.env.render()

    def _move(
        self,
        q_from: np.ndarray,
        q_to: np.ndarray,
        gripper: float,
        buf: _EpisodeBuf,
        n_steps: int | None = None,
    ):
        """Interpolate joints from q_from → q_to over n_steps control steps."""
        if n_steps is None:
            n_steps = self.cfg.steps_per_phase
        for i in range(1, n_steps + 1):
            alpha = i / n_steps
            q_interp = q_from + alpha * (q_to - q_from)
            self._control_step(q_interp, gripper, buf)

    def _hold(
        self,
        q: np.ndarray,
        gripper: float,
        buf: _EpisodeBuf,
        n_steps: int = 10,
    ):
        """Hold position for n_steps, then run settle_steps extra physics steps."""
        for _ in range(n_steps):
            self._control_step(q, gripper, buf)
        # Extra physics settle (not recorded) so gripper fully opens/closes
        action = np.concatenate([q, [gripper]], dtype=np.float32)
        self.env.step(action)
        self._physics_steps(self.cfg.settle_steps)

    # ------------------------------------------------------------------
    # Full episode
    # ------------------------------------------------------------------

    def run_episode(self, buf: _EpisodeBuf) -> bool:
        """
        Execute one scripted episode and return True on success.

        Returns False (without touching the dataset) if:
          - IK residual is too large for any waypoint, OR
          - The environment's check_success() returns False after retract.
        """
        env = self.env
        cfg = self.cfg

        # --- Current world state ---
        p_obj = env.env.get_p_body(env.mug_body_name).copy()
        p_bin = env.env.get_p_body(env.plate_body_name).copy()
        obj_init = env.obj_init_pose.copy()

        # Lazily compute palm-to-tip offset in the TCP local frame using the
        # 'gripperframe' site when available.
        if self._tcp_tip_offset_local is None:
            try:
                site_id = mujoco.mj_name2id(
                    env.env.model, mujoco.mjtObj.mjOBJ_SITE, "gripperframe"
                )
                if site_id >= 0:
                    p_tip_now = env.env.data.site_xpos[site_id].copy()
                    p_palm_now, R_palm_now = env.env.get_pR_body(env.tcp_body_name)
                    # Store offset in the palm/TCP local frame so it can be
                    # rotated consistently by a fixed target orientation.
                    self._tcp_tip_offset_local = R_palm_now.T @ (p_tip_now - p_palm_now)
                else:
                    self._tcp_tip_offset_local = np.zeros(3, dtype=np.float32)
            except Exception:
                self._tcp_tip_offset_local = np.zeros(3, dtype=np.float32)

        # --- Waypoint positions ---
        # p_obj[2] is the block body-centre after physics settling (~table_z + 0.0125 m).
        # p_bin[2] is the bin body-origin (fixed in XML at 0.80 m).
        # Tip-centric waypoints (desired positions for the jaw-tip site in world).
        p_tip_approach  = np.array([p_obj[0], p_obj[1], p_obj[2] + cfg.approach_height])
        p_tip_grasp     = np.array([p_obj[0], p_obj[1], p_obj[2] + cfg.grasp_height])
        p_tip_lift      = np.array([p_obj[0], p_obj[1], p_obj[2] + cfg.lift_height])
        p_tip_above_bin = np.array([p_bin[0], p_bin[1], p_obj[2] + cfg.lift_height])
        p_tip_place     = np.array([p_bin[0], p_bin[1], p_bin[2] + cfg.place_height])
        p_tip_retract   = np.array([p_bin[0], p_bin[1], p_bin[2] + cfg.retract_height])

        # Fixed "face-down" palm orientation: matches the SO-101 default pose
        # used in SimpleEnv.reset(). With this fixed R, the local palm→tip
        # offset becomes a constant world-frame vector at all waypoints.
        R_target = _GRASP_R
        offset_local = (
            self._tcp_tip_offset_local
            if self._tcp_tip_offset_local is not None
            else np.zeros(3, dtype=np.float32)
        )
        offset_world = R_target @ offset_local

        # Convert desired tip positions to TCP/palm targets using the cached
        # local offset and fixed orientation. These palm targets are used by
        # the numeric IK path; the geometric path in _solve_wp() instead uses
        # the tip positions directly as analytic targets.
        p_approach  = p_tip_approach  - offset_world
        p_grasp     = p_tip_grasp     - offset_world
        p_lift      = p_tip_lift      - offset_world
        p_above_bin = p_tip_above_bin - offset_world
        p_place     = p_tip_place     - offset_world
        p_retract   = p_tip_retract   - offset_world

        # Orientation policy: enforce a fixed face-down palm at all waypoints.
        R_approach  = R_target
        R_grasp     = R_target
        R_lift      = R_target
        R_above_bin = R_target
        R_place     = R_target
        R_retract   = R_target

        # Debug: compare IK targets against current TCP and object/bin positions.
        p_tcp, _ = env.env.get_pR_body(body_name=env.tcp_body_name)
        print("[DEBUG] p_tcp =", p_tcp, "p_obj =", p_obj, "p_bin =", p_bin)
        for name, p_wp in [
            ("approach", p_tip_approach),
            ("grasp", p_tip_grasp),
            ("lift", p_tip_lift),
            ("above_bin", p_tip_above_bin),
            ("place", p_tip_place),
            ("retract", p_tip_retract),
        ]:
            print(f"[DEBUG] {name}: {p_wp}")

        # --- Solve IK for all waypoints (arm-only, no gripper) ---
        # The numeric IK always targets the palm (gripper body); palm targets are
        # used as p_trgt so the body-frame error is measured correctly.
        # For SO-101 the geometric IK seed uses the tip (gripperframe site)
        # positions, passed separately via p_tip_trgt.
        p_palm_targets = (p_approach,  p_grasp,  p_lift,  p_above_bin,  p_place,  p_retract)
        p_tip_targets  = (p_tip_approach, p_tip_grasp, p_tip_lift, p_tip_above_bin,
                          p_tip_place, p_tip_retract)
        R_targets      = (R_approach, R_grasp, R_lift, R_above_bin, R_place, R_retract)
        tip_args = p_tip_targets if cfg.env_robot_profile == "so101" else (None,) * 6
        print("    [IK] solving 6 waypoints…", end=" ", flush=True)
        q0          = _get_arm_q(env)
        q_approach, e0 = _solve_wp(env, p_palm_targets[0], R_targets[0], q0,          cfg.max_ik_tick, cfg.ik_err_th, p_tip_trgt=tip_args[0])
        q_grasp,    e1 = _solve_wp(env, p_palm_targets[1], R_targets[1], q_approach,  cfg.max_ik_tick, cfg.ik_err_th, p_tip_trgt=tip_args[1])
        q_lift,     e2 = _solve_wp(env, p_palm_targets[2], R_targets[2], q_grasp,     cfg.max_ik_tick, cfg.ik_err_th, p_tip_trgt=tip_args[2])
        q_above_bin,e3 = _solve_wp(env, p_palm_targets[3], R_targets[3], q_lift,      cfg.max_ik_tick, cfg.ik_err_th, p_tip_trgt=tip_args[3])
        q_place,    e4 = _solve_wp(env, p_palm_targets[4], R_targets[4], q_above_bin, cfg.max_ik_tick, cfg.ik_err_th, p_tip_trgt=tip_args[4])
        q_retract,  e5 = _solve_wp(env, p_palm_targets[5], R_targets[5], q_place,     cfg.max_ik_tick, cfg.ik_err_th, p_tip_trgt=tip_args[5])

        max_err = max(e0, e1, e2, e3, e4, e5)
        print(f"max IK err = {max_err:.4f} m")

        if max_err > cfg.max_ik_err_skip:
            print(f"    [skip] IK error {max_err:.4f} > threshold {cfg.max_ik_err_skip:.4f}")
            return False

        # --- Execute phases ---
        print("    phase 1/8: approach")
        self._move(q0, q_approach, GRIPPER_OPEN, buf)

        print("    phase 2/8: descend")
        self._move(q_approach, q_grasp, GRIPPER_OPEN, buf)

        print("    phase 3/8: grasp  (close gripper)")
        self._hold(q_grasp, GRIPPER_CLOSED, buf, n_steps=10)

        print("    phase 4/8: lift")
        self._move(q_grasp, q_lift, GRIPPER_CLOSED, buf)

        print("    phase 5/8: transport")
        self._move(q_lift, q_above_bin, GRIPPER_CLOSED, buf)

        print("    phase 6/8: place  (descend over bin)")
        self._move(q_above_bin, q_place, GRIPPER_CLOSED, buf)

        print("    phase 7/8: release (open gripper)")
        self._hold(q_place, GRIPPER_OPEN, buf, n_steps=10)

        print("    phase 8/8: retract")
        self._move(q_place, q_retract, GRIPPER_OPEN, buf)

        # Let block settle fully in bin before checking
        action = np.concatenate([q_retract, [GRIPPER_OPEN]], dtype=np.float32)
        self.env.step(action)
        self._physics_steps(50)

        success = bool(env.check_success())
        return success


# ---------------------------------------------------------------------------
# Main collection loop
# ---------------------------------------------------------------------------

def collect(env: SimpleEnv, dataset: LeRobotDataset, cfg: ScriptedConfig):
    fsm = PickPlaceFSM(env, cfg)

    saved     = 0
    attempted = 0
    t_start   = time.time()

    while saved < cfg.num_demo:
        attempted += 1
        print(f"\n[episode {attempted}]  saved {saved}/{cfg.num_demo}")

        buf      = _EpisodeBuf()
        obj_init = env.obj_init_pose.copy()   # captured before physics changes it
        success  = fsm.run_episode(buf)

        if success and len(buf) > 0:
            buf.commit(dataset, cfg.task_name, obj_init)
            saved += 1
            elapsed = time.time() - t_start
            rate    = saved / elapsed
            eta     = (cfg.num_demo - saved) / rate if rate > 0 else float("inf")
            print(f"    ✓ saved  |  total={saved}  rate={rate:.2f} eps/s  ETA={eta:.0f}s")
        else:
            buf.discard(dataset)
            print("    ✗ failed — resetting and retrying")

        # Use a fixed seed only for the initial environment construction; subsequent
        # resets should randomise the block positions so scripted demos cover a
        # diverse workspace instead of repeating the same spawn configuration.
        env.reset()

    elapsed = time.time() - t_start
    print(f"\n[done] {saved} demos saved in {elapsed:.1f}s")
    print(f"       attempted={attempted}")
    print(f"       success_rate={(saved / max(attempted, 1)) * 100:.1f}%")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    cfg = parse_args()
    print(f"[scripted_demo] robot={cfg.env_robot_profile}  target={cfg.num_demo} demos")
    print(f"[scripted_demo] dataset root: {cfg.root}")
    print(f"[scripted_demo] heights  approach={cfg.approach_height:.3f}"
          f"  grasp={cfg.grasp_height:.3f}"
          f"  lift={cfg.lift_height:.3f}"
          f"  place={cfg.place_height:.3f}"
          f"  retract={cfg.retract_height:.3f}")
    print(f"[scripted_demo] motion   steps_per_phase={cfg.steps_per_phase}"
          f"  sim_substeps={cfg.sim_substeps}"
          f"  settle_steps={cfg.settle_steps}")

    env     = build_env(cfg)
    dataset = build_dataset(cfg)

    try:
        collect(env, dataset, cfg)
    finally:
        env.env.close_viewer()
        if cfg.cleanup_images:
            images_dir = Path(cfg.root) / "images"
            if images_dir.exists():
                shutil.rmtree(images_dir)
                print(f"[scripted_demo] Cleaned up raw images at {images_dir}")


if __name__ == "__main__":
    main()
