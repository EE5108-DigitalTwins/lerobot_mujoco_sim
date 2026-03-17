#!/usr/bin/env python
"""
Roll out a trained ACT policy in the MuJoCo SO-101 pick-and-place environment.

Students should be able to:
- drop a checkpoint folder on disk (e.g. checkpoints/act_y/)
- run this script without editing code
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
import torchvision
from PIL import Image

# Get project root directory (parent of scripts/)
PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from lerobot.common.datasets.lerobot_dataset import LeRobotDatasetMetadata
from lerobot.common.datasets.utils import dataset_to_policy_features
from lerobot.common.policies.act.configuration_act import ACTConfig
from lerobot.common.policies.act.modeling_act import ACTPolicy
from lerobot.configs.types import FeatureType
from mujoco_env.y_env import SimpleEnv


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Deploy a trained ACT policy in MuJoCo.")
    p.add_argument(
        "--checkpoint",
        default=str(PROJECT_ROOT / "checkpoints" / "act_y"),
        help="Path to ACT checkpoint directory (created by ACTPolicy.save_pretrained).",
    )
    p.add_argument(
        "--dataset-root",
        default=str(PROJECT_ROOT / "data" / "demo_data_so101"),
        help="Local dataset root used to load dataset stats/features for normalization.",
    )
    p.add_argument("--dataset-repo-id", default="so101_pnp", help="LeRobot dataset repo id used for metadata.")
    p.add_argument(
        "--xml-path",
        default=str(PROJECT_ROOT / "asset" / "scene_so101_y.xml"),
        help="MuJoCo scene XML path.",
    )
    p.add_argument("--pick-body-name", default="body_obj_block_2")
    p.add_argument("--place-body-name", default="body_obj_bin")
    p.add_argument("--task", default="Put green block in the bin")
    p.add_argument("--hz", type=int, default=20, help="Control loop rate in Hz.")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument(
        "--device",
        default="auto",
        choices=["auto", "cpu", "cuda"],
        help="Device to run policy on.",
    )
    return p.parse_args()


def resolve_device(device_flag: str) -> torch.device:
    if device_flag == "cpu":
        return torch.device("cpu")
    if device_flag == "cuda":
        return torch.device("cuda")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def main() -> None:
    args = parse_args()
    device = resolve_device(args.device)

    checkpoint_path = Path(args.checkpoint).expanduser()
    if not checkpoint_path.is_absolute():
        checkpoint_path = (PROJECT_ROOT / checkpoint_path).resolve()

    deploy_meta_path = checkpoint_path / "deploy_metadata.json"
    dataset_stats = None
    features_raw = None
    if deploy_meta_path.exists():
        deploy_meta = json.loads(deploy_meta_path.read_text(encoding="utf-8"))
        dataset_stats = deploy_meta.get("stats")
        features_raw = deploy_meta.get("features")
        if deploy_meta.get("dataset_repo_id"):
            args.dataset_repo_id = deploy_meta["dataset_repo_id"]

    if dataset_stats is None or features_raw is None:
        dataset_root = Path(args.dataset_root).expanduser()
        if not dataset_root.is_absolute():
            dataset_root = (PROJECT_ROOT / dataset_root).resolve()

        dataset_metadata = LeRobotDatasetMetadata(args.dataset_repo_id, root=str(dataset_root))
        dataset_stats = dataset_metadata.stats
        features_raw = dataset_metadata.features

    features = dataset_to_policy_features(features_raw)
    output_features = {key: ft for key, ft in features.items() if ft.type is FeatureType.ACTION}
    input_features = {key: ft for key, ft in features.items() if key not in output_features}
    # This environment provides wrist images; ACT configs in this repo typically drop it.
    input_features.pop("observation.wrist_image", None)

    cfg = ACTConfig(
        input_features=input_features,
        output_features=output_features,
        chunk_size=10,
        n_action_steps=1,
        temporal_ensemble_coeff=0.9,
    )

    policy = ACTPolicy.from_pretrained(
        str(checkpoint_path),
        config=cfg,
        dataset_stats=dataset_stats,
    )
    policy.to(device)
    policy.eval()

    env = SimpleEnv(
        args.xml_path,
        action_type="joint_angle",
        pick_body_name=args.pick_body_name,
        place_body_name=args.place_body_name,
    )

    img_transform = torchvision.transforms.ToTensor()

    step = 0
    env.reset(seed=args.seed)
    policy.reset()

    while env.env.is_viewer_alive():
        env.step_env()
        if env.env.loop_every(HZ=args.hz):
            if bool(env.check_success()):
                print("[deploy_act] Success — resetting.")
                policy.reset()
                env.reset(seed=args.seed)
                step = 0

            state = env.get_ee_pose()
            image, wrist_image = env.grab_image()

            image_t = img_transform(Image.fromarray(image).resize((256, 256))).unsqueeze(0).to(device)
            wrist_t = img_transform(Image.fromarray(wrist_image).resize((256, 256))).unsqueeze(0).to(device)

            batch = {
                "observation.state": torch.tensor([state], dtype=torch.float32, device=device),
                "observation.image": image_t,
                "observation.wrist_image": wrist_t,
                "task": [args.task],
                "timestamp": torch.tensor([step / float(args.hz)], dtype=torch.float32, device=device),
            }

            action = policy.select_action(batch)[0].detach().cpu().numpy()
            action = np.asarray(action, dtype=np.float32)
            _ = env.step(action)
            env.render()
            step += 1


if __name__ == "__main__":
    main()
