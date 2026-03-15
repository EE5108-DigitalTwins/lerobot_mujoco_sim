import numpy as np
from so101_forward_kinematics import get_g45, get_g5t


'''Inverse kinematics for SO101 robot arm.'''


def get_inverse_kinematics(target_position, target_orientation):
    "Geometric appraoch specific to the so-101 arms"
    
    # Initialize the joint configuration dictionary
    joint_config = {
        'shoulder_pan': 0.0,
        'shoulder_lift': 0.0,
        'elbow_flex': 0.0,
        'wrist_flex': 0.0,
        'wrist_roll': 0.0,
        'gripper': 0.0
    }

    return joint_config

def get_wrist_flex_position(target_position):
    p = np.array(target_position)

    # Build desired tool frame (vertical grasp, z pointing down)
    R_wt = np.array([
        [1,  0,  0],
        [0, -1,  0],
        [0,  0, -1]
    ])
    g_wt = np.eye(4)
    g_wt[:3, :3] = R_wt
    g_wt[:3,  3] = p

    # Build g_4t from FK helpers (joint 4 → tool)
    g_45 = get_g45(0.0, translation_dict['g45_displacement'])
    g_5t = get_g5t(translation_dict['g5t_displacement'])
    g_4t = g_45 @ g_5t

    # Wrist position in world frame
    g_w4 = g_wt @ np.linalg.inv(g_4t)

    wrist_flex_position    = g_w4[:3, 3]
    wrist_flex_orientation = g_w4[:3, :3]

    return wrist_flex_position, wrist_flex_orientation

