# Visualize your data
#
# Visualize your action based on the reconstructed simulation scene.
#
# The main simulation is replaying the action.
#
# The overlayed images on the top right and bottom right are from the dataset.

import sys
from pathlib import Path

# Get project root directory (parent of scripts/)
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from lerobot.common.datasets.lerobot_dataset import LeRobotDataset, LeRobotDatasetMetadata
import numpy as np
from lerobot.common.datasets.utils import write_json, serialize_dict

dataset = LeRobotDataset('so101_pnp', root=str(PROJECT_ROOT / 'data' / 'demo_data_so101'))  # if you want to use the example data provided, root = PROJECT_ROOT / 'data' / 'demo_data_example' instead!

# ## Load Dataset

import torch

class EpisodeSampler(torch.utils.data.Sampler):
    """
    Sampler for a single episode
    """
    def __init__(self, dataset: LeRobotDataset, episode_index: int):
        from_idx = dataset.episode_data_index["from"][episode_index].item()
        to_idx = dataset.episode_data_index["to"][episode_index].item()
        self.frame_ids = range(from_idx, to_idx)

    def __iter__(self):
        return iter(self.frame_ids)

    def __len__(self) -> int:
        return len(self.frame_ids)


# Select an episode index that you want to visualize
episode_index = 0

episode_sampler = EpisodeSampler(dataset, episode_index)
dataloader = torch.utils.data.DataLoader(
    dataset,
    num_workers=1,
    batch_size=1,
    sampler=episode_sampler,
)

# ## Visualize your Dataset on Simulation

from mujoco_env.y_env import SimpleEnv
xml_path = str(PROJECT_ROOT / 'asset' / 'scene_so101_y.xml')
PnPEnv = SimpleEnv(
    xml_path, 
    action_type='joint_angle',
    mug_body_name='body_obj_block_3',
    plate_body_name='body_obj_bin'
)

step = 0
iter_dataloader = iter(dataloader)
PnPEnv.reset()

while PnPEnv.env.is_viewer_alive():
    PnPEnv.step_env()
    if PnPEnv.env.loop_every(HZ=20):
        # Get the action from dataset
        data = next(iter_dataloader)
        if step == 0:
            # Reset the object pose based on the dataset
            PnPEnv.set_obj_pose(data['obj_init'][0,:3], data['obj_init'][0,3:])
        # Get the action from dataset
        action = data['action'].numpy()
        obs = PnPEnv.step(action[0])

        # Visualize the image from dataset to rgb_overlay
        PnPEnv.rgb_agent = data['observation.image'][0].numpy()*255
        PnPEnv.rgb_ego = data['observation.wrist_image'][0].numpy()*255
        PnPEnv.rgb_agent = PnPEnv.rgb_agent.astype(np.uint8)
        PnPEnv.rgb_ego = PnPEnv.rgb_ego.astype(np.uint8)
        # 3 256 256 -> 256 256 3
        PnPEnv.rgb_agent = np.transpose(PnPEnv.rgb_agent, (1,2,0))
        PnPEnv.rgb_ego = np.transpose(PnPEnv.rgb_ego, (1,2,0))
        PnPEnv.rgb_side = np.zeros((480, 640, 3), dtype=np.uint8)
        PnPEnv.render()
        step += 1

        if step == len(episode_sampler):
            # start from the beginning
            iter_dataloader = iter(dataloader)
            PnPEnv.reset()
            step = 0

PnPEnv.env.close_viewer()

# ### [Optional] Save Stats.json for other versions

stats = dataset.meta.stats
PATH = dataset.root / 'meta' / 'stats.json'
stats = serialize_dict(stats)

write_json(stats, PATH)
