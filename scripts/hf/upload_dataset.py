#!/usr/bin/env python3
"""
Upload a local LeRobot dataset to the Hugging Face Hub.

This is a thin wrapper around `LeRobotDataset.push_to_hub(...)`.

Auth:
- Set `HF_TOKEN` (or be logged in via `huggingface-cli login`) before running.
"""

import argparse
import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from lerobot.common.datasets.lerobot_dataset import LeRobotDataset  # type: ignore[import-untyped]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Upload a LeRobotDataset to Hugging Face Hub.")
    p.add_argument(
        "--root",
        required=True,
        help="Local dataset root directory (e.g. ./data/demo_data_so101).",
    )
    p.add_argument(
        "--repo-id",
        required=True,
        help='Target Hub repo id like "username/dataset_name".',
    )
    p.add_argument(
        "--private",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Create/upload as a private dataset repo if supported.",
    )
    p.add_argument(
        "--upload-large-folder",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Enable large-folder upload path (recommended for images).",
    )
    p.add_argument(
        "--revision",
        default=None,
        help="Optional git revision/branch name on the Hub (if supported).",
    )
    p.add_argument(
        "--commit-message",
        default="Upload dataset",
        help="Commit message for the Hub commit.",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()

    root = Path(args.root).expanduser()
    if not root.is_absolute():
        root = (PROJECT_ROOT / root).resolve()
    if not root.exists():
        raise FileNotFoundError(f"Dataset root does not exist: {root}")

    # Helpful early failure if someone is running in an offline container.
    if os.environ.get("HF_HUB_OFFLINE") == "1":
        raise RuntimeError(
            "HF_HUB_OFFLINE=1 is set in the environment. Disable it to upload to the Hub."
        )

    dataset = LeRobotDataset(args.repo_id, root=str(root))

    kwargs = {
        "upload_large_folder": args.upload_large_folder,
        "commit_message": args.commit_message,
    }
    # These kwargs may not exist in all LeRobot versions; pass only if supported.
    if args.revision is not None:
        kwargs["revision"] = args.revision
    if args.private:
        kwargs["private"] = True

    try:
        dataset.push_to_hub(**kwargs)
    except TypeError:
        # Fall back to minimal signature if the underlying library is older.
        dataset.push_to_hub(upload_large_folder=args.upload_large_folder)

    print(f"[upload_dataset] Uploaded: {args.repo_id} from {root}")


if __name__ == "__main__":
    main()

