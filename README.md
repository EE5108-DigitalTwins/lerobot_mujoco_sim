# LeRobot Tutorial with MuJoCo

> **Original Work Credit:**  
> This repository is based on [lerobot-mujoco-tutorial](https://github.com/jeongeun980906/lerobot-mujoco-tutorial) by [Jeongeun Park](https://github.com/jeongeun980906).  
> All original work is credited to the original author. This fork includes modifications and extensions for educational purposes.

This repository contains minimal examples for collecting demonstration data and training (or fine-tuning) vision language action models on custom datasets. 

## Project Structure
```
lerobot-mujoco-tutorial/
├── README.md               # This file
├── requirements.txt        # Python dependencies
├── docker-compose.yml      # Docker configuration
├── Dockerfile              # Docker image definition
├── asset/                  # MuJoCo scene files and robot models
│   ├── objaverse/         # 3D object assets
│   ├── robotis_omy/       # OMY robot assets
│   ├── so_arm100/         # SO-ARM100/101 robot assets
│   └── tabletop/          # Scene objects
├── checkpoints/           # Trained model checkpoints
│   └── act_y/            # ACT model checkpoints
├── configs/              # YAML configuration files
│   ├── collect_data.yaml
│   ├── pi0_so101.yaml
│   └── smolvla_so101.yaml
├── data/                 # Demonstration datasets
│   ├── demo_data/
│   ├── demo_data_example/
│   ├── demo_data_so101/
│   └── demo_data_so100*/
├── media/                # Media files for documentation
├── mujoco_env/          # MuJoCo environment implementation
├── notebooks/           # Jupyter notebooks for tutorials
│   ├── 1.collect_data.ipynb
│   ├── 2.visualize_data.ipynb
│   ├── 3.train.ipynb
│   ├── 4.deploy.ipynb
│   ├── 5.language_env.ipynb
│   ├── 6.visualize_data.ipynb
│   ├── 7.pi0.ipynb
│   └── 8.smolvla.ipynb
├── scripts/             # Python scripts
│   ├── 1.collect_data.py
│   ├── 1.scripted_demo.py      # NEW: Automated demo generation
│   ├── 2.visualize_data.py
│   ├── 3.train.py
│   ├── 4.deploy.py
│   ├── 5.language_env.py
│   ├── 6.visualize_data_language.py
│   ├── 7.pi0_deploy.py
│   ├── 8.smolvla_deploy.py
│   └── train_model.py
└── third_party/        # Third-party code
    └── SO-ARM100/
```

## Table of Contents
- [Installation](#installation)
- [Updates and Plans](#updates--plans)
- [Recent Changes](#recent-changes)
- [1. Collect Demonstration Data](#1-collect-demonstration-data)
- [1b. Scripted Demonstration Generation](#1b-scripted-demonstration-generation)
- [2. Playback Your Data](#2-playback-your-data)
- [3. Train Action-Chunking-Transformer (ACT)](#3-train-action-chunking-transformer-act)
- [4. Deploy ACT](#4-deploy-your-policy)
- [5-6. Language conditioned Environment.](#5-6-collect-data-and-visualize-in-lanugage-conditioned-environment)
- [Models and Dataset](#models-and-dataset-)
- [7.Train and deploy pi_0](#7-train-and-deploy-pi_0)
- [8.Train and deploy smolvla](#8-train-and-deploy-smolvla)
- [Acknowledgements](#acknowledgements)
- [License](#license)

## Installation
We have tested our environment on python 3.10. 

I do **not** recommend installing lerobot package with `pip install lerobot`. This causes errors. 

Install mujoco package dependencies and lerobot
```
pip install -r requirements.txt
```
Make sure your mujoco version is **3.1.6**.

Unzip the asset
```
cd asset/objaverse
unzip plate_11.zip
```

## SO-100 / SO-101 Arm Assets

The SO arm definitions from [TheRobotStudio/SO-ARM100](https://github.com/TheRobotStudio/SO-ARM100) are now added to this workspace.

- Vendored sources: `third_party/SO-ARM100`
- Local simulation assets:
  - `asset/so_arm100/SO101/` (MuJoCo XML + meshes)
  - `asset/so_arm100/SO100/` (URDF + converted MuJoCo XML + meshes)

Verified loadable MuJoCo entrypoints:
- `asset/so_arm100/SO101/so101_new_calib.xml`
- `asset/so_arm100/SO100/so100.xml`

To use either robot in notebooks/scripts, set:

```python
xml_path = '../asset/so_arm100/SO101/so101_new_calib.xml'  # From scripts/
# or
xml_path = './asset/so_arm100/SO101/so101_new_calib.xml'   # From notebooks/
```

### Updates & Plans

:white_check_mark: Viewer Update.

:white_check_mark: Add different mugs, plates for different language instructions.

:white_check_mark: Add pi_0 training and inference. 

:white_check_mark: Add SmolVLA

:white_check_mark: Add scripted demonstration generation (FSM-based controller)

## Recent Changes

This workspace has undergone significant improvements to enhance simulation reliability and data collection:

### Environment Enhancements
- **Improved object spawning**: Added bin exclusion zones and reachability checks to prevent objects from spawning in unreachable locations or inside the target bin
- **Pick-and-place gating**: Success now requires the object to be lifted off the table before placement, preventing false positives
- **Configurable spawn bounds**: Updated default spawn ranges to match SO-101 arm reachability (`spawn_x_range: [0.25, 0.52]`, `spawn_y_range: [0.02, 0.22]`)
- **Enhanced success detection**: Added height-based checks relative to bin geometry for more reliable task completion verification

### New Features
- **Scripted demonstration generator** (`scripts/1.scripted_demo.py`): Fully automated FSM-based pick-and-place controller for generating large-scale demonstration datasets without manual teleoperation
  - IK-based waypoint planning with error checking
  - 8-phase motion sequence (approach → descend → grasp → lift → transport → place → release → retract)
  - Automatic retry on IK failures or task failures
  - Configurable motion parameters and spawn bounds
  - Output compatible with existing training pipeline

### Asset and Scene Updates
- **Updated MuJoCo scenes**: Refined block and bin geometries for more reliable grasping and placement
- **Improved collision handling**: Enhanced object meshes and collision geometries in `asset/tabletop/object/`
- **Calibrated SO-101 arm**: Updated `asset/so_arm100/SO101/so101_new_calib.xml` with improved joint limits and damping

### Configuration Changes
- **Updated spawn defaults**: All configs now use validated spawn bounds that match robot workspace
- **Dataset organization**: Default data paths moved to `./data/` subdirectory for better organization

### Code Quality
- **Mouse control improvements**: Enhanced viewer interaction in `mujoco_parser.py`
- **Robust state management**: Better tracking of gripper states and object positions
- **Error handling**: Improved IK convergence checking and fallback strategies

## Running Scripts vs Notebooks

## Arm Profiles and Defaults

The project supports multiple robot arm profiles:
- `so101` (default)
- `so100`
- `omy`

By default, data collection loads the SO101 scene/profile from [configs/collect_data.yaml](configs/collect_data.yaml):
- `env_robot_profile: so101`
- `xml_path: ./asset/scene_so101_y.xml`
- `root: ./data/demo_data_so101`
- Spawn bounds (calibrated for SO-101 reachability):
  - `spawn_x_range: [0.25, 0.52]` m
  - `spawn_y_range: [0.02, 0.22]` m
  - `spawn_z_range: [0.815, 0.815]` m

You can switch profiles at runtime with:
```bash
python 1.collect_data.py --env-robot-profile so100
python 1.collect_data.py --env-robot-profile omy
```

Or by editing [configs/collect_data.yaml](configs/collect_data.yaml).

You can use either the Python scripts in `scripts/` or the Jupyter notebooks in `notebooks/`:

**Python Scripts:**
```bash
cd scripts
python 1.collect_data.py
python 1.scripted_demo.py --num-demo 100  # Automated generation
python 2.visualize_data.py
python 3.train.py
python 4.deploy.py
```

**Jupyter Notebooks:**
```bash
jupyter notebook notebooks/
```

## 1. Collect Demonstration Data

Run [notebooks/1.collect_data.ipynb](notebooks/1.collect_data.ipynb) or use the script:
```bash
cd scripts
python 1.collect_data.py
```

Collect demonstration data for the given environment.
The task is to pick a mug and place it on the plate. The environment recognizes the success if the mug is on the plate, the gripper opened, and the end-effector positioned above the mug.

**Default Scene Configuration:**
The default scene `asset/scene_so101_y.xml` uses:
- Object to pick: `body_obj_block_3` (green block)
- Target location: `body_obj_bin` (bin)
- Task: "Put green block in the bin"

**Spawn Configuration:**
The spawn bounds have been calibrated for SO-101 reachability:
- X range: `[0.25, 0.52]` m (forward reach from robot base)
- Y range: `[0.02, 0.22]` m (lateral reach)
- Z range: `[0.815, 0.815]` m (table surface height)
- Minimum distance between objects: `0.2` m
- Automatic bin exclusion: Objects won't spawn inside or too close to the target bin

If using a custom scene, update the `mug_body_name` and `plate_body_name` parameters accordingly.

<img src="./media/teleop.gif" width="480" height="360">

Use WASD for the xy plane, RF for the z-axis, QE for tilt, and ARROWs for the rest of the rotations. 

SPACEBAR will change your gripper's state, and Z key will reset your environment with discarding the current episode data.

For overlayed images, 
- Top Right: Agent View 
- Bottom Right: Egocentric View
- Top Left: Left Side View
- Bottom Left: Top View

## 1b. Scripted Demonstration Generation

For generating large-scale demonstration datasets automatically, use the scripted FSM controller:

```bash
cd scripts
python 1.scripted_demo.py --num-demo 100
```

This script uses a Finite State Machine (FSM) controller with IK-based waypoint planning to generate demonstrations programmatically. The output format is identical to manual teleoperation data, so the same training pipeline works without modification.

**Key Features:**
- Fully automated pick-and-place execution
- 8-phase motion sequence: approach → descend → grasp → lift → transport → place → release → retract
- Automatic IK solving with error checking and retry logic
- Configurable motion parameters and spawn bounds
- Success verification before saving episodes
- Option to disable rendering for faster generation

**Common Usage Examples:**

```bash
# Generate 100 demonstrations with visualization
python 1.scripted_demo.py --num-demo 100

# Fast generation without rendering (for large datasets)
python 1.scripted_demo.py --num-demo 1000 --no-render --seed 42

# Adjust motion timing for smoother/faster trajectories
python 1.scripted_demo.py --num-demo 50 --steps-per-phase 30

# Customize height waypoints (all values in metres)
python 1.scripted_demo.py --num-demo 100 \
    --approach-height 0.15 \
    --grasp-height 0.01 \
    --lift-height 0.25 \
    --place-height 0.15 \
    --retract-height 0.30
```

**Configuration Parameters:**
- `--approach-height`: EEF height above block center for safe approach (default: 0.12 m)
- `--grasp-height`: EEF height above block center at grasp point (default: 0.01 m)
- `--lift-height`: EEF height above block center after grasping (default: 0.20 m)
- `--place-height`: EEF height above bin origin for release (default: 0.13 m)
- `--retract-height`: EEF height above bin origin after release (default: 0.25 m)
- `--steps-per-phase`: Control steps per motion segment (default: 20)
- `--sim-substeps`: Physics steps per control step (default: 25, matches 20 Hz recording rate)
- `--max-ik-err-skip`: Skip episode if IK error exceeds threshold (default: 0.05 m)

The scripted generator is particularly useful for:
- Creating baseline datasets for training
- Generating large-scale datasets (1000+ demonstrations) quickly
- Testing environment configurations and spawn bounds
- Pre-training models before fine-tuning with human demonstrations

**Output:** Saved to `data/demo_data_so101_scripted/` by default (configurable with `--root`).

The dataset is contained as follows:
```
fps = 20,
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
        "names": ["state"], # x, y, z, roll, pitch, yaw
    },
    "action": {
        "dtype": "float32",
        "shape": (7,),
        "names": ["action"], # 6 joint angles and 1 gripper
    },
    "obj_init": {
        "dtype": "float32",
        "shape": (6,),
        "names": ["obj_init"], # just the initial position of the object. Not used in training.
    },
},

```

This will make the dataset in the `data/demo_data_so101` folder (by default), which will look like this:
```
data/demo_data_so101/
├── data/
│   ├── chunk-000/
│   │   ├── episode_000000.parquet
│   │   └── ...
├── meta/
│   ├── episodes.jsonl
│   ├── info.json
│   ├── stats.json
│   └── tasks.jsonl
```

For convenience, we have added [Example Data](data/demo_data_example/) to the repository. 

## 2. Playback Your Data

Run [notebooks/2.visualize_data.ipynb](notebooks/2.visualize_data.ipynb) or:
```bash
cd scripts
python 2.visualize_data.py
```

<img src="./media/data.gif" width="480" height="360"></img>

Visualize your action based on the reconstructed simulation scene. 

The main simulation is replaying the action.

The overlayed images on the top right and bottom right are from the dataset.

**Note:** The visualization script now includes enhanced success checking that verifies:
- The target object was lifted off the table during the episode
- The object is positioned inside the target bin
- The gripper is open after placement
- The end-effector is positioned above the bin walls 

## 3. Train Action-Chunking-Transformer (ACT)

Run [notebooks/3.train.ipynb](notebooks/3.train.ipynb) or:
```bash
cd scripts
python 3.train.py
```

**This takes around 30~60 mins**.

Train the ACT model on your custom dataset. In this example, we set chunk_size as 10. 

The trained checkpoint will be saved in the `checkpoints/act_y` folder.

To evaluate the policy on the dataset, you can calculate the error between ground-truth actions from the dataset.

**Training Tips:**
- For scripted datasets (generated with `1.scripted_demo.py`), you may achieve faster convergence due to more consistent trajectories
- Combining scripted and human-teleoperated data can improve policy robustness
- The enhanced environment with spawn bounds and success checking helps produce cleaner training data

<image src="./media/inference.png"  width="480" height="360">

<details>
    <summary>PicklingError: Can't pickle <function <lambda> at 0x131d1bd00>: attribute lookup <lambda> on __main__ failed</summary>
If you have a pickling error, 
        
```
PicklingError: Can't pickle <function <lambda> at 0x131d1bd00>: attribute lookup <lambda> on __main__ failed
```

Please set your num_workers as 0, like, 

```
dataloader = torch.utils.data.DataLoader(
    dataset,
    num_workers=0, # 4
    batch_size=64,
    shuffle=True,
    pin_memory=device.type != "cpu",
    drop_last=True,
)
```
</details>

## 4. Deploy your Policy

Run [notebooks/4.deploy.ipynb](notebooks/4.deploy.ipynb) or:
```bash
cd scripts
python 4.deploy.py
```

You can download checkpoint from [google drive](https://drive.google.com/drive/folders/1UqxqUgGPKU04DkpQqSWNgfYMhlvaiZsp?usp=sharing) if you don't have gpu to train your model.

<img src="./media/rollout.gif" width="480" height="360" controls></img>

Deploy trained policy in simulation.

**Deployment Notes:**
- The enhanced environment ensures more reliable success detection during rollouts
- Object spawn positions now respect robot reachability constraints
- Success requires demonstrating a proper pick-and-place sequence (lift → transport → place)


## 5-6. Collect data and visualize in language conditioned environment

- [notebooks/5.language_env.ipynb](notebooks/5.language_env.ipynb): Collect Dataset with keyboard teleoperation. The command is same as first environment.
- [notebooks/6.visualize_data.ipynb](notebooks/6.visualize_data.ipynb): Visualize Collected Data

Or use the scripts:
```bash
cd scripts
python 5.language_env.py
python 6.visualize_data_language.py
```


### Environment
**Data**

<img src="./media/data_v2.gif" width="480" height="360" controls></img>


## Models and Dataset 🤗
<table>
  <tr>
    <th> Model 🤗 </th>
    <th> Dataset  🤗</th>
    </tr>
    <tr>
      <td> <a href="https://huggingface.co/Jeongeun/so101_pnp_pi0"> pi_0 finetuned </a></td>
      <td> <a href="https://huggingface.co/datasets/Jeongeun/so101_pnp_language"> dataset </a></td>
    </tr>
    <tr>
      <td> <a href="https://huggingface.co/Jeongeun/so101_pnp_smolvla"> smolvla finetuned </td>
        <td>  same dataset</td>
    </tr>
</table>

## 7. Train and Deploy pi_0
- [scripts/train_model.py](scripts/train_model.py): Training script
- [configs/pi0_so101.yaml](configs/pi0_so101.yaml): Training configuration file
- [notebooks/7.pi0.ipynb](notebooks/7.pi0.ipynb): Policy deployment
- [scripts/pi0_deploy.py](scripts/pi0_deploy.py): Policy deployment script



### Training Scripts
```bash
python train_model.py --config_path configs/pi0_so101.yaml
```



### Rollout of trained policy

<img src="./media/rollout2.gif" width="480" height="360" controls></img>


### Train logs

<image src="./media/wandb.png"  width="480" height="360">

### Configuration File
```
dataset:
  repo_id: so101_pnp_language # Repository ID
  root: ../data/demo_data_language # Your root for data file!
policy:
  type : pi0
  chunk_size: 5
  n_action_steps: 5
  
save_checkpoint: true
output_dir: ../checkpoints/pi0_so101 # Save directory
batch_size: 16
job_name : pi0_so101
resume: false 
seed : 42
num_workers: 8
steps: 20_000
eval_freq: -1 # No evaluation
log_freq: 50
save_checkpoint: true
save_freq: 10_000
use_policy_training_preset: true
  
wandb:
  enable: true
  project: pi0_so101
  entity: <your_wandb_entity>
  disable_artifact: true
```

## 8. Train and Deploy Smolvla

- [scripts/train_model.py](scripts/train_model.py): Training script
- [configs/smolvla_so101.yaml](configs/smolvla_so101.yaml): Training configuration file
- [notebooks/8.smolvla.ipynb](notebooks/8.smolvla.ipynb): Policy deployment
- [scripts/smolvla_deploy.py](scripts/smolvla_deploy.py): Policy deployment script



### Training Scripts
```bash
python train_model.py --config_path configs/smolvla_so101.yaml
```



### Rollout of trained policy

<img src="./media/rollout3.gif" width="480" height="360" controls></img>


### Train logs

<image src="./media/wandb2.png"  width="480" height="360">

### Configuration File
```
dataset:
  repo_id: so101_pnp_language # Repository ID
  root: ../data/demo_data_language # Your root for data file!
policy:
  type : smolvla
  chunk_size: 5
  n_action_steps: 5
  device: cuda
  
save_checkpoint: true
output_dir: ../checkpoints/smolvla_so101 # Save directory
batch_size: 16
job_name : smolvla_so101
resume: false 
seed : 42
num_workers: 8
steps: 20_000
eval_freq: -1 # No evaluation
log_freq: 50
save_checkpoint: true
save_freq: 10_000
use_policy_training_preset: true
  
wandb:
  enable: true
  project: smolvla_so101
  entity: <your_wandb_entity>
  disable_artifact: true
```


## Acknowledgements

This repository is a fork of the original [lerobot-mujoco-tutorial](https://github.com/jeongeun980906/lerobot-mujoco-tutorial) created by [Jeongeun Park](https://github.com/jeongeun980906). We are grateful for their excellent work in creating this comprehensive tutorial for robotics learning.

Additional acknowledgements:
- The asset for the robotis-omy manipulator is from [robotis_mujoco_menagerie](https://github.com/ROBOTIS-GIT/robotis_mujoco_menagerie/tree/main).
- The [MuJoco Parser Class](./mujoco_env/mujoco_parser.py) is modified from [yet-another-mujoco-tutorial](https://github.com/sjchoi86/yet-another-mujoco-tutorial-v3). 
- We refer to original tutorials from [lerobot examples](https://github.com/huggingface/lerobot/tree/main/examples).  
- The assets for plate and mug is from [Objaverse](https://objaverse.allenai.org/).

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

**Original Work:** This repository is based on [lerobot-mujoco-tutorial](https://github.com/jeongeun980906/lerobot-mujoco-tutorial) by Jeongeun Park.

**Third-party Components:** Note that some components may be subject to different licenses:
- SO-ARM100 assets: Apache License 2.0
- LeRobot framework: Apache License 2.0
- Other third-party assets: See respective repositories