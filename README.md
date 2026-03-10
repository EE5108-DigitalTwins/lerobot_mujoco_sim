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
│   ├── pi0_omy.yaml
│   └── smolvla_omy.yaml
├── data/                 # Demonstration datasets
│   ├── demo_data/
│   ├── demo_data_example/
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
│   ├── collect_data.py
│   ├── visualize_data.py
│   ├── train.py
│   ├── deploy.py
│   ├── language_env.py
│   ├── visualize_data_language.py
│   ├── pi0_deploy.py
│   └── smolvla_deploy.py
└── third_party/        # Third-party code
    └── SO-ARM100/
```

## Table of Contents
- [:pencil: Installation](#installation)
- [:mega: Updates and Plans](#updates--plans)
- [:video_game: 1. Collect Demonstration Data](#1-collect-demonstration-data)
- [:movie_camera: 2. Playback Your Data](#2-playback-your-data)
- [:fire: 3. Train Action-Chunking-Transformer (ACT)](#3-train-action-chunking-transformer-act)
- [:pushpin: 4. Deploy ACT](#4-deploy-your-policy)
- [:floppy_disk: 5-6. Language conditioned Environment.](#5-6-collect-data-and-visualize-in-lanugage-conditioned-environment)
- [🤗 Models and Dataset](#models-and-dataset-)
- [:zap:7.Train and deploy pi_0](#7-train-and-deploy-pi_0)
- [:bulb:8.Train and deploy smolvla](#8-train-and-deploy-smolvla)
- [:pencil: Acknowledgements](#acknowledgements)
- [:page_facing_up: License](#license)

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

## Running Scripts vs Notebooks

You can use either the Python scripts in `scripts/` or the Jupyter notebooks in `notebooks/`:

**Python Scripts:**
```bash
cd scripts
python collect_data.py
python train.py
python deploy.py
```

**Jupyter Notebooks:**
```bash
jupyter notebook notebooks/
```

## 1. Collect Demonstration Data

Run [notebooks/1.collect_data.ipynb](notebooks/1.collect_data.ipynb) or use the script:
```bash
cd scripts
python collect_data.py
```

Collect demonstration data for the given environment.
The task is to pick a mug and place it on the plate. The environment recognizes the success if the mug is on the plate, gthe ripper opened, and the end-effector positioned above the mug.

**Default Scene Configuration:**
The default scene `asset/example_scene_y.xml` uses:
- Object to pick: `body_obj_redbull` (Red Bull can)
- Target location: `body_obj_bin` (bin)
- Task: "Put red bull can in the bin"

If using a custom scene, update the `mug_body_name` and `plate_body_name` parameters accordingly.

<img src="./media/teleop.gif" width="480" height="360">

Use WASD for the xy plane, RF for the z-axis, QE for tilt, and ARROWs for the rest of rthe otations. 

SPACEBAR will change your gripper's state, and Z key will reset your environment with discarding the current episode data.

For overlayed images, 
- Top Right: Agent View 
- Bottom Right: Egocentric View
- Top Left: Left Side View
- Bottom Left: Top View

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

This will make the dataset in the `data/demo_data` folder, which will look like this:
```
data/demo_data/
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
python visualize_data.py
```

<img src="./media/data.gif" width="480" height="360"></img>

Visualize your action based on the reconstructed simulation scene. 

The main simulation is replaying the action.

The overlayed images on the top right and bottom right are from the dataset. 

## 3. Train Action-Chunking-Transformer (ACT)

Run [notebooks/3.train.ipynb](notebooks/3.train.ipynb) or:
```bash
cd scripts
python train.py
```

**This takes around 30~60 mins**.

Train the ACT model on your custom dataset. In this example, we set chunk_size as 10. 

The trained checkpoint will be saved in the `checkpoints/act_y` folder.

To evaluate the policy on the dataset, you can calculate the error between ground-truth actions from the dataset.

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
python deploy.py
```

You can download checkpoint from [google drive](https://drive.google.com/drive/folders/1UqxqUgGPKU04DkpQqSWNgfYMhlvaiZsp?usp=sharing) if you don't have gpu to train your model.

<img src="./media/rollout.gif" width="480" height="360" controls></img>

Deploy trained policy in simulation.


## 5-6. Collect data and visualize in language conditioned environment

- [notebooks/5.language_env.ipynb](notebooks/5.language_env.ipynb): Collect Dataset with keyboard teleoperation. The command is same as first environment.
- [notebooks/6.visualize_data.ipynb](notebooks/6.visualize_data.ipynb): Visualize Collected Data

Or use the scripts:
```bash
cd scripts
python language_env.py
python visualize_data_language.py
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
        <td> <a href="https://huggingface.co/Jeongeun/omy_pnp_pi0"> pi_0 finetuned </a></td>
        <td> <a href="https://huggingface.co/datasets/Jeongeun/omy_pnp_language"> dataset </a></td>
    </tr>
    <tr>
        <td> <a href="https://huggingface.co/Jeongeun/omy_pnp_smolvla"> smolvla finetuned </td>
        <td>  same dataset</td>
    </tr>
</table>

## 7. Train and Deploy pi_0
- [scripts/train_model.py](scripts/train_model.py): Training script
- [configs/pi0_omy.yaml](configs/pi0_omy.yaml): Training configuration file
- [notebooks/7.pi0.ipynb](notebooks/7.pi0.ipynb): Policy deployment
- [scripts/pi0_deploy.py](scripts/pi0_deploy.py): Policy deployment script



### Training Scripts
```bash
python train_model.py --config_path configs/pi0_omy.yaml
```



### Rollout of trained policy

<img src="./media/rollout2.gif" width="480" height="360" controls></img>


### Train logs

<image src="./media/wandb.png"  width="480" height="360">

### Configuration File
```
dataset:
  repo_id: omy_pnp_language # Repository ID
  root: ../data/demo_data_language # Your root for data file!
policy:
  type : pi0
  chunk_size: 5
  n_action_steps: 5
  
save_checkpoint: true
output_dir: ../checkpoints/pi0_omy # Save directory
batch_size: 16
job_name : pi0_omy
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
  project: pi0_omy
  entity: <your_wandb_entity>
  disable_artifact: true
```

## 8. Train and Deploy Smolvla

- [scripts/train_model.py](scripts/train_model.py): Training script
- [configs/smolvla_omy.yaml](configs/smolvla_omy.yaml): Training configuration file
- [notebooks/8.smolvla.ipynb](notebooks/8.smolvla.ipynb): Policy deployment
- [scripts/smolvla_deploy.py](scripts/smolvla_deploy.py): Policy deployment script



### Training Scripts
```bash
python train_model.py --config_path configs/smolvla_omy.yaml
```



### Rollout of trained policy

<img src="./media/rollout3.gif" width="480" height="360" controls></img>


### Train logs

<image src="./media/wandb2.png"  width="480" height="360">

### Configuration File
```
dataset:
  repo_id: omy_pnp_language # Repository ID
  root: ../data/demo_data_language # Your root for data file!
policy:
  type : smolvla
  chunk_size: 5
  n_action_steps: 5
  device: cuda
  
save_checkpoint: true
output_dir: ../checkpoints/smolvla_omy # Save directory
batch_size: 16
job_name : smolvla_omy
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
  project: smolvla_omy
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


### temp
launch jupyter lab:
jupyter lab --ip=0.0.0.0 --port=8888 --no-browser --allow-root
