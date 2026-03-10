# Collect Demonstration from Keyboard
#
# Collect demonstration data for the given environment.
# The task is to pick a mug and place it on the plate. The environment recognizes the success if the mug
# is on the plate, the gripper opened, and the end-effector positioned above the mug.
#
# Use WASD for the xy plane, RF for the z-axis, QE for tilt, and ARROWs for the rest of the rotations.
# SPACEBAR will change your gripper's state, and Z key will reset your environment with discarding the current episode data.
#
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

import random
import numpy as np
import os
from PIL import Image
from mujoco_env.y_env2 import SimpleEnv2
from lerobot.common.datasets.lerobot_dataset import LeRobotDataset

# If you want to randomize the object positions, set this to None
# If you fix the seed, the object positions will be the same every time
SEED = 0
# SEED = None  # Uncomment this line to randomize the object positions

REPO_NAME = 'omy_pnp_language'
NUM_DEMO = 20  # Number of demonstrations to collect
ROOT = str(PROJECT_ROOT / 'data' / 'demo_data_language')  # The root directory to save the demonstrations

# ### Object body names (for swapping objects)
# Update the body-name variables below when you change scene objects.
# Where to find valid names:
# - Open the object XML you included in the scene (e.g. asset/objaverse/mug_6/model_new.xml).
# - Use the <body name="..."> value (e.g. body_obj_mug_6).
# - Do the same for both mug body names and the plate body name.

xml_path = str(PROJECT_ROOT / 'asset' / 'example_scene_y2.xml')

# Choose object body names from the object XML files included in your scene.
# Examples:
# - mug_5 -> body_obj_mug_5
# - mug_6 -> body_obj_mug_6
MUG_RED_BODY_NAME = 'body_obj_mug_5'
MUG_BLUE_BODY_NAME = 'body_obj_mug_6'
PLATE_BODY_NAME = 'body_obj_plate_11'

# Define the environment
PnPEnv = SimpleEnv2(
    xml_path,
    seed=SEED,
    state_type='joint_angle',
    mug_red_body_name=MUG_RED_BODY_NAME,
    mug_blue_body_name=MUG_BLUE_BODY_NAME,
    plate_body_name=PLATE_BODY_NAME,
)

# ## Define Dataset Features and Create your dataset!

create_new = True
if os.path.exists(ROOT):
    print(f"Directory {ROOT} already exists.")
    ans = input("Do you want to delete it? (y/n) ")
    if ans == 'y':
        import shutil
        shutil.rmtree(ROOT)
    else:
        create_new = False

if create_new:
    dataset = LeRobotDataset.create(
                repo_id=REPO_NAME,
                root=ROOT,
                robot_type="omy",
                fps=20,  # 20 frames per second
                features={
                    "observation.image": {
                        "dtype": "image",
                        "shape": (256, 256, 3),
                        "names": ["height", "width", "channels"],
                    },
                    "observation.wrist_image": {
                        "dtype": "image",
                        "shape": (256, 256, 3),
                        "names": ["height", "width", "channel"],
                    },
                    "observation.state": {
                        "dtype": "float32",
                        "shape": (6,),
                        "names": ["state"],  # x, y, z, roll, pitch, yaw
                    },
                    "action": {
                        "dtype": "float32",
                        "shape": (7,),
                        "names": ["action"],  # 6 joint angles and 1 gripper
                    },
                    "obj_init": {
                        "dtype": "float32",
                        "shape": (9,),
                        "names": ["obj_init"],  # just the initial position of the object. Not used in training.
                    },
                },
                image_writer_threads=10,
                image_writer_processes=5,
        )
else:
    print("Load from previous dataset")
    dataset = LeRobotDataset(REPO_NAME, root=ROOT)

# ## Keyboard Control
# You can teleop your robot with keyboard and collect dataset.
# To receive the success signal, you have to release the gripper and move upwards above the mug!

action = np.zeros(7)
episode_id = 0
record_flag = False  # Start recording when the robot starts moving
while PnPEnv.env.is_viewer_alive() and episode_id < NUM_DEMO:
    PnPEnv.step_env()
    if PnPEnv.env.loop_every(HZ=20):
        # check if the episode is done
        done = PnPEnv.check_success()
        if done:
            # Save the episode data and reset the environment
            dataset.save_episode()
            PnPEnv.reset()
            episode_id += 1
        # Teleoperate the robot and get delta end-effector pose with gripper
        action, reset = PnPEnv.teleop_robot()
        if not record_flag and sum(action) != 0:
            record_flag = True
            print("Start recording")
        if reset:
            # Reset the environment and clear the episode buffer
            # This can be done by pressing 'z' key
            PnPEnv.reset()
            dataset.clear_episode_buffer()
            record_flag = False
        # Step the environment
        # Get the end-effector pose and images
        agent_image, wrist_image = PnPEnv.grab_image()
        # resize to 256x256
        agent_image = Image.fromarray(agent_image)
        wrist_image = Image.fromarray(wrist_image)
        agent_image = agent_image.resize((256, 256))
        wrist_image = wrist_image.resize((256, 256))
        agent_image = np.array(agent_image)
        wrist_image = np.array(wrist_image)
        joint_q = PnPEnv.step(action)
        action = PnPEnv.q[:7]  # 6 joint angles and 1 gripper
        action = action.astype(np.float32)
        if record_flag:
            # Add the frame to the dataset
            dataset.add_frame({
                    "observation.image": agent_image,
                    "observation.wrist_image": wrist_image,
                    "observation.state": joint_q[:6],
                    "action": action,
                    "obj_init": PnPEnv.obj_init_pose,
                }, task=PnPEnv.instruction
            )
        PnPEnv.render(teleop=True, idx=episode_id)

PnPEnv.env.close_viewer()

# Clean up the images folder
import shutil
shutil.rmtree(dataset.root / 'images')
