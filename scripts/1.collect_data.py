#!/usr/bin/env python
######################################################################################
# Collect Demonstration from Keyboard

# Collect demonstration data for the given environment.
# The task is to pick an object and place in a designated location. The environment recognizes the success if the object is in the designated location, the gripper is opened, and the end-effector is positioned above the object.

# Use WASD for the xy plane, RF for the z-axis, QE for tilt, and ARROWs for the rest of the rotations. 
# SPACEBAR will change your gripper's state, and Z key will reset your environment with discarding the current episode data.

# For overlayed images, 
# - Top Right: Agent View 
# - Bottom Right: Egocentric View
# - Top Left: Left Side View
# - Bottom Left: Top View

import sys
from pathlib import Path

# Get project root directory (parent of scripts/)
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import argparse
import numpy as np
import os
from pathlib import Path
from PIL import Image
import shutil
from dataclasses import dataclass
import yaml
from mujoco_env.y_env import SimpleEnv
from lerobot.common.datasets.lerobot_dataset import LeRobotDataset

@dataclass
class CollectConfig:
    seed: int | None = None
    repo_name: str = 'omy_pnp'
    num_demo: int = 1
    root: str = str(Path(__file__).resolve().parent.parent / 'data' / 'demo_data')
    task_name: str = 'Put red bull can in the bin'
    xml_path: str = str(Path(__file__).resolve().parent.parent / 'asset' / 'example_scene_y.xml')
    mug_body_name: str = 'body_obj_redbull'
    plate_body_name: str = 'body_obj_bin'
    env_robot_profile: str = 'omy'
    offline_local_only: bool = True
    delete_existing_dataset: bool = False
    robot_type: str = 'omy'
    fps: int = 20
    image_size: int = 256
    image_writer_threads: int = 10
    image_writer_processes: int = 5
    cleanup_images: bool = True
    spawn_x_min: float = 0.40
    spawn_x_max: float = 0.56
    spawn_y_min: float = -0.2
    spawn_y_max: float = 0.2
    spawn_z_min: float = 0.82
    spawn_z_max: float = 0.82
    spawn_min_dist: float = 0.2
    spawn_xy_margin: float = 0.0
    spawn_fallback_min_dist: float = 0.1


def parse_args():
    bootstrap_parser = argparse.ArgumentParser(add_help=False)
    bootstrap_parser.add_argument("--config", default="collect_data.yaml", help="Path to YAML config file.")
    bootstrap_args, _ = bootstrap_parser.parse_known_args()

    default_cfg = CollectConfig()
    merged_defaults = default_cfg.__dict__.copy()
    if os.path.exists(bootstrap_args.config):
        with open(bootstrap_args.config, "r", encoding="utf-8") as f:
            yaml_cfg = yaml.safe_load(f) or {}
        unknown_keys = [k for k in yaml_cfg.keys() if k not in merged_defaults]
        if unknown_keys:
            raise ValueError(f"Unknown keys in config file {bootstrap_args.config}: {unknown_keys}")
        merged_defaults.update(yaml_cfg)

    parser = argparse.ArgumentParser(
        description="Collect teleoperation demonstrations.",
        parents=[bootstrap_parser],
    )
    parser.add_argument("--seed", type=int, default=merged_defaults["seed"], help="Random seed. Use None for randomized object positions.")
    parser.add_argument("--repo-name", default=merged_defaults["repo_name"], help="LeRobot dataset repo id.")
    parser.add_argument("--num-demo", type=int, default=merged_defaults["num_demo"], help="Number of demonstrations to collect.")
    parser.add_argument("--root", default=merged_defaults["root"], help="Root directory to save demonstrations.")
    parser.add_argument("--task-name", default=merged_defaults["task_name"], help="Task name stored in dataset.")
    parser.add_argument("--xml-path", default=merged_defaults["xml_path"], help="MuJoCo scene XML path.")
    parser.add_argument("--mug-body-name", default=merged_defaults["mug_body_name"], help="Object body name in scene.")
    parser.add_argument("--plate-body-name", default=merged_defaults["plate_body_name"], help="Target body name in scene.")
    parser.add_argument("--env-robot-profile", default=merged_defaults["env_robot_profile"], choices=["omy", "so100", "so101"], help="Robot kinematic profile used by SimpleEnv.")
    parser.add_argument("--offline-local-only", action=argparse.BooleanOptionalAction, default=merged_defaults["offline_local_only"], help="Use local dataset metadata only and avoid any Hugging Face fallback calls.")
    parser.add_argument("--delete-existing-dataset", action=argparse.BooleanOptionalAction, default=merged_defaults["delete_existing_dataset"], help="Whether to delete existing dataset directory before collection.")
    parser.add_argument("--robot-type", default=merged_defaults["robot_type"], help="Robot type for LeRobotDataset.")
    parser.add_argument("--fps", type=int, default=merged_defaults["fps"], help="Dataset FPS.")
    parser.add_argument("--image-size", type=int, default=merged_defaults["image_size"], help="Square image size for saved frames.")
    parser.add_argument("--image-writer-threads", type=int, default=merged_defaults["image_writer_threads"], help="Image writer thread count.")
    parser.add_argument("--image-writer-processes", type=int, default=merged_defaults["image_writer_processes"], help="Image writer process count.")
    parser.add_argument("--cleanup-images", action=argparse.BooleanOptionalAction, default=merged_defaults["cleanup_images"], help="Whether to remove raw image directory after run.")
    parser.add_argument("--spawn-x-min", type=float, default=merged_defaults["spawn_x_min"], help="Minimum x for object spawn sampling.")
    parser.add_argument("--spawn-x-max", type=float, default=merged_defaults["spawn_x_max"], help="Maximum x for object spawn sampling.")
    parser.add_argument("--spawn-y-min", type=float, default=merged_defaults["spawn_y_min"], help="Minimum y for object spawn sampling.")
    parser.add_argument("--spawn-y-max", type=float, default=merged_defaults["spawn_y_max"], help="Maximum y for object spawn sampling.")
    parser.add_argument("--spawn-z-min", type=float, default=merged_defaults["spawn_z_min"], help="Minimum z for object spawn sampling.")
    parser.add_argument("--spawn-z-max", type=float, default=merged_defaults["spawn_z_max"], help="Maximum z for object spawn sampling.")
    parser.add_argument("--spawn-min-dist", type=float, default=merged_defaults["spawn_min_dist"], help="Minimum distance between spawned objects.")
    parser.add_argument("--spawn-xy-margin", type=float, default=merged_defaults["spawn_xy_margin"], help="XY margin inside spawn bounds.")
    parser.add_argument("--spawn-fallback-min-dist", type=float, default=merged_defaults["spawn_fallback_min_dist"], help="Fallback min distance if initial sampling fails.")

    args = parser.parse_args()
    return CollectConfig(
        seed=args.seed,
        repo_name=args.repo_name,
        num_demo=args.num_demo,
        root=args.root,
        task_name=args.task_name,
        xml_path=args.xml_path,
        mug_body_name=args.mug_body_name,
        plate_body_name=args.plate_body_name,
        env_robot_profile=args.env_robot_profile,
        offline_local_only=args.offline_local_only,
        delete_existing_dataset=args.delete_existing_dataset,
        robot_type=args.robot_type,
        fps=args.fps,
        image_size=args.image_size,
        image_writer_threads=args.image_writer_threads,
        image_writer_processes=args.image_writer_processes,
        cleanup_images=args.cleanup_images,
        spawn_x_min=args.spawn_x_min,
        spawn_x_max=args.spawn_x_max,
        spawn_y_min=args.spawn_y_min,
        spawn_y_max=args.spawn_y_max,
        spawn_z_min=args.spawn_z_min,
        spawn_z_max=args.spawn_z_max,
        spawn_min_dist=args.spawn_min_dist,
        spawn_xy_margin=args.spawn_xy_margin,
        spawn_fallback_min_dist=args.spawn_fallback_min_dist,
    )


def resolve_robot_scene_defaults(config):
    default_omy_xml = str(PROJECT_ROOT / 'asset' / 'example_scene_y.xml')
    so100_scene_xml = str(PROJECT_ROOT / 'asset' / 'example_scene_so100_y.xml')
    default_root = str(PROJECT_ROOT / 'data' / 'demo_data')

    if config.env_robot_profile == 'so100' and config.xml_path == default_omy_xml:
        config.xml_path = so100_scene_xml
    elif config.env_robot_profile == 'so101' and config.xml_path == default_omy_xml:
        config.xml_path = str(PROJECT_ROOT / 'asset' / 'so_arm100' / 'SO101' / 'so101_new_calib.xml')

    if config.root == default_root and config.env_robot_profile == 'so100':
        config.root = str(PROJECT_ROOT / 'data' / 'demo_data_so100')
    elif config.root == default_root and config.env_robot_profile == 'so101':
        config.root = str(PROJECT_ROOT / 'data' / 'demo_data_so101')

    return config


def validate_config(config):
    if config.num_demo <= 0:
        raise ValueError("--num-demo must be > 0")
    if config.fps <= 0:
        raise ValueError("--fps must be > 0")
    if config.image_size <= 0:
        raise ValueError("--image-size must be > 0")
    if config.image_writer_threads <= 0:
        raise ValueError("--image-writer-threads must be > 0")
    if config.image_writer_processes <= 0:
        raise ValueError("--image-writer-processes must be > 0")

    if config.spawn_x_min > config.spawn_x_max:
        raise ValueError("--spawn-x-min must be <= --spawn-x-max")
    if config.spawn_y_min > config.spawn_y_max:
        raise ValueError("--spawn-y-min must be <= --spawn-y-max")
    if config.spawn_z_min > config.spawn_z_max:
        raise ValueError("--spawn-z-min must be <= --spawn-z-max")

    if config.spawn_min_dist < 0:
        raise ValueError("--spawn-min-dist must be >= 0")
    if config.spawn_fallback_min_dist < 0:
        raise ValueError("--spawn-fallback-min-dist must be >= 0")
    if config.spawn_xy_margin < 0:
        raise ValueError("--spawn-xy-margin must be >= 0")

    x_span = config.spawn_x_max - config.spawn_x_min
    y_span = config.spawn_y_max - config.spawn_y_min
    if 2 * config.spawn_xy_margin >= x_span:
        raise ValueError("--spawn-xy-margin is too large for x range")
    if 2 * config.spawn_xy_margin >= y_span:
        raise ValueError("--spawn-xy-margin is too large for y range")


def build_env(config):
    return SimpleEnv(
        config.xml_path,
        robot_profile=config.env_robot_profile,
        seed=config.seed,
        state_type='joint_angle',
        mug_body_name=config.mug_body_name,
        plate_body_name=config.plate_body_name,
        spawn_x_range=(config.spawn_x_min, config.spawn_x_max),
        spawn_y_range=(config.spawn_y_min, config.spawn_y_max),
        spawn_z_range=(config.spawn_z_min, config.spawn_z_max),
        spawn_min_dist=config.spawn_min_dist,
        spawn_xy_margin=config.spawn_xy_margin,
        spawn_fallback_min_dist=config.spawn_fallback_min_dist,
    )


def create_or_load_dataset(config):
    action_dim = 7 if config.env_robot_profile == 'omy' else 6

    def has_minimal_local_meta(root_path: Path) -> bool:
        required_files = [
            root_path / 'meta' / 'info.json',
            root_path / 'meta' / 'tasks.jsonl',
            root_path / 'meta' / 'episodes.jsonl',
        ]
        return all(path.exists() for path in required_files)

    create_new = True
    root_path = Path(config.root)
    if root_path.exists():
        print(f"Directory {config.root} already exists.")
        if config.delete_existing_dataset:
            shutil.rmtree(root_path)
            print(f"Deleted existing dataset directory: {config.root}")
        else:
            if config.offline_local_only:
                print("[collect_data] Offline local-only mode is enabled; skipping existing dataset load and creating a fresh local dataset.")
            elif has_minimal_local_meta(root_path):
                try:
                    print("Load from previous dataset")
                    return LeRobotDataset(config.repo_name, root=config.root)
                except Exception as exc:
                    print(f"[collect_data] Failed to load existing dataset locally: {exc}")
                    print("[collect_data] Falling back to creating a fresh local dataset.")
            else:
                print("[collect_data] Existing directory is missing dataset metadata; creating a fresh local dataset.")

            suffix_root = f"{config.root}_fresh"
            suffix_idx = 0
            candidate = Path(suffix_root)
            while candidate.exists():
                suffix_idx += 1
                candidate = Path(f"{suffix_root}_{suffix_idx}")
            config.root = str(candidate)
            print(f"[collect_data] New dataset root: {config.root}")

    if create_new:
        return LeRobotDataset.create(
            repo_id=config.repo_name,
            root=config.root,
            robot_type=config.robot_type,
            fps=config.fps,
            features={
                "observation.image": {
                    "dtype": "image",
                    "shape": (config.image_size, config.image_size, 3),
                    "names": ["height", "width", "channels"],
                },
                "observation.wrist_image": {
                    "dtype": "image",
                    "shape": (config.image_size, config.image_size, 3),
                    "names": ["height", "width", "channel"],
                },
                "observation.state": {
                    "dtype": "float32",
                    "shape": (6,),
                    "names": ["state"],  # x, y, z, roll, pitch, yaw
                },
                "action": {
                    "dtype": "float32",
                    "shape": (action_dim,),
                    "names": ["action"],
                },
                "obj_init": {
                    "dtype": "float32",
                    "shape": (6,),
                    "names": ["obj_init"],  # initial object pose, not used in training
                },
            },
            image_writer_threads=config.image_writer_threads,
            image_writer_processes=config.image_writer_processes,
        )


def resize_images(agent_image, wrist_image, image_size):
    agent_image = Image.fromarray(agent_image).resize((image_size, image_size))
    wrist_image = Image.fromarray(wrist_image).resize((image_size, image_size))
    return np.array(agent_image), np.array(wrist_image)


def collect_demonstrations(env, dataset, config):
    action = np.zeros(7)
    episode_id = 0
    record_flag = False

    while env.env.is_viewer_alive() and episode_id < config.num_demo:
        env.step_env()
        if env.env.loop_every(HZ=20):
            done = env.check_success()
            if done:
                dataset.save_episode()
                env.reset(seed=config.seed)
                episode_id += 1

            action, reset = env.teleop_robot()
            if not record_flag and sum(action) != 0:
                record_flag = True
                print("Start recording")

            if reset:
                env.reset(seed=config.seed)
                dataset.clear_episode_buffer()
                record_flag = False

            ee_pose = env.get_ee_pose()
            agent_image, wrist_image = env.grab_image()
            agent_image, wrist_image = resize_images(agent_image, wrist_image, config.image_size)
            joint_q = env.step(action)

            if record_flag:
                dataset.add_frame(
                    {
                        "observation.image": agent_image,
                        "observation.wrist_image": wrist_image,
                        "observation.state": ee_pose,
                        "action": joint_q,
                        "obj_init": env.obj_init_pose,
                    },
                    task=config.task_name,
                )

            env.render(teleop=True)


def cleanup_dataset_images(dataset, cleanup_images):
    if not cleanup_images:
        return

    images_dir = dataset.root / 'images'
    if os.path.exists(images_dir):
        shutil.rmtree(images_dir)


def main():
    config = parse_args()
    config = resolve_robot_scene_defaults(config)
    validate_config(config)
    print(f"[collect_data] env_robot_profile={config.env_robot_profile}, xml_path={config.xml_path}")
    env = build_env(config)
    dataset = create_or_load_dataset(config)

    try:
        collect_demonstrations(env, dataset, config)
    finally:
        env.env.close_viewer()
        cleanup_dataset_images(dataset, config.cleanup_images)


if __name__ == "__main__":
    main()