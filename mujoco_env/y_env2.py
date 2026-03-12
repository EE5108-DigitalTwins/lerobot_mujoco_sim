import sys
import random
import numpy as np
import xml.etree.ElementTree as ET
from mujoco_env.mujoco_parser import MuJoCoParserClass
from mujoco_env.utils import prettify, rotation_matrix, add_title_to_img
from mujoco_env.ik import solve_ik
from mujoco_env.transforms import rpy2r, r2rpy
import os
import copy
import glfw

class SimpleEnv2:
    def __init__(self, 
                 xml_path,
                action_type='eef_pose', 
                state_type='joint_angle',
                seed = None,
                mug_red_body_name='body_obj_mug_5',
                mug_blue_body_name='body_obj_mug_6',
                plate_body_name='body_obj_plate_11',
                spawn_x_range=(0.25, 0.52),
                spawn_y_range=(0.02, 0.22),
                spawn_z_range=(0.815, 0.815),
                spawn_min_dist=0.2,
                spawn_xy_margin=0.0,
                spawn_fallback_min_dist=0.1):
        """
        args:
            xml_path: str, path to the xml file
            action_type: str, type of action space, 'eef_pose','delta_joint_angle' or 'joint_angle'
            state_type: str, type of state space, 'joint_angle' or 'ee_pose'
            seed: int, seed for random number generator
            mug_red_body_name: str, body name for the red mug object in the MuJoCo model
            mug_blue_body_name: str, body name for the blue mug object in the MuJoCo model
            plate_body_name: str, body name for the plate object in the MuJoCo model
        """
        # Load the xml file
        self.env = MuJoCoParserClass(name='Tabletop',rel_xml_path=xml_path)
        self.action_type = action_type
        self.state_type = state_type
        self.mug_red_body_name = mug_red_body_name
        self.mug_blue_body_name = mug_blue_body_name
        self.plate_body_name = plate_body_name
        self.spawn_x_range = spawn_x_range
        self.spawn_y_range = spawn_y_range
        self.spawn_z_range = spawn_z_range
        self.spawn_min_dist = spawn_min_dist
        self.spawn_xy_margin = spawn_xy_margin
        self.spawn_fallback_min_dist = spawn_fallback_min_dist
        self.spawn_bin_exclusion_margin = 0.015
        self.spawn_reach_radius = 0.50

        self.joint_names = ['joint1',
                    'joint2',
                    'joint3',
                    'joint4',
                    'joint5',
                    'joint6',]
        self._prev_key_state = {}
        self.init_viewer()
        self.reset(seed)

    def _key_is_down(self, key):
        if self.env.is_key_pressed_repeat(key=key):
            return True

        viewer = getattr(self.env, 'viewer', None)
        window = getattr(viewer, 'window', None)
        if window is None:
            return False

        try:
            return glfw.get_key(window, key) == glfw.PRESS
        except Exception:
            return False

    def _key_is_pressed_once(self, key):
        is_down = self._key_is_down(key)
        was_down = self._prev_key_state.get(key, False)
        self._prev_key_state[key] = is_down
        return is_down and not was_down

    def _get_bin_exclusion_bounds(self):
        """
        Return a rectangular XY exclusion zone for the fixed bin.
        """
        try:
            body_names = self.env.get_body_names(prefix='body_obj_')
            if 'body_obj_bin' not in body_names:
                return None
            p_bin = self.env.get_p_body('body_obj_bin')
        except Exception:
            return None

        half_extent = 0.06 + self.spawn_bin_exclusion_margin
        return (
            p_bin[0] - half_extent,
            p_bin[0] + half_extent,
            p_bin[1] - half_extent,
            p_bin[1] + half_extent,
        )

    def _is_within_reach_xy(self, x, y):
        """
        Check whether an XY point is inside a conservative SO101 reachable area.
        """
        try:
            p_base = self.env.get_p_body('base')
        except Exception:
            return True
        d_xy = np.linalg.norm(np.array([x - p_base[0], y - p_base[1]], dtype=np.float32))
        return bool(d_xy <= self.spawn_reach_radius)

    def _sample_object_xyzs(self, n_obj, min_dist):
        """
        Sample object positions while respecting reachability and bin exclusion.
        """
        x_min, x_max = self.spawn_x_range
        y_min, y_max = self.spawn_y_range
        z_min, z_max = self.spawn_z_range
        m = self.spawn_xy_margin
        x_lo, x_hi = x_min + m, x_max - m
        y_lo, y_hi = y_min + m, y_max - m
        if x_lo > x_hi or y_lo > y_hi:
            raise ValueError("spawn_xy_margin is too large for the configured spawn range.")

        bin_bounds = self._get_bin_exclusion_bounds()
        xyzs = []
        max_attempts = 5000
        attempts = 0
        while len(xyzs) < n_obj and attempts < max_attempts:
            attempts += 1
            x = np.random.uniform(x_lo, x_hi)
            y = np.random.uniform(y_lo, y_hi)
            z = np.random.uniform(z_min, z_max)
            cand = np.array([x, y, z], dtype=np.float32)

            if not self._is_within_reach_xy(x, y):
                continue

            if bin_bounds is not None:
                bx0, bx1, by0, by1 = bin_bounds
                if (bx0 <= x <= bx1) and (by0 <= y <= by1):
                    continue

            if any(np.linalg.norm(cand - other) < min_dist for other in xyzs):
                continue

            xyzs.append(cand)

        if len(xyzs) != n_obj:
            raise ValueError(
                f"Could not sample {n_obj} objects within constrained spawn region "
                f"after {max_attempts} attempts."
            )
        return np.stack(xyzs, axis=0)

    def _teleop_debug_status(self):
        key_map = {
            'W': glfw.KEY_W,
            'A': glfw.KEY_A,
            'S': glfw.KEY_S,
            'D': glfw.KEY_D,
            'R': glfw.KEY_R,
            'F': glfw.KEY_F,
            'Q': glfw.KEY_Q,
            'E': glfw.KEY_E,
            'LEFT': glfw.KEY_LEFT,
            'RIGHT': glfw.KEY_RIGHT,
            'UP': glfw.KEY_UP,
            'DOWN': glfw.KEY_DOWN,
            'SPACE': glfw.KEY_SPACE,
            'Z': glfw.KEY_Z,
        }
        active_keys = [name for name, key in key_map.items() if self._key_is_down(key)]

        viewer = getattr(self.env, 'viewer', None)
        window = getattr(viewer, 'window', None)
        focused = False
        if window is not None:
            try:
                focused = bool(glfw.get_window_attrib(window, glfw.FOCUSED))
            except Exception:
                focused = False

        return focused, active_keys

    def init_viewer(self):
        '''
        Initialize the viewer
        '''
        self.env.reset()
        try:
            p_base = self.env.get_p_body('base')
            p_tcp = self.env.get_p_body('tcp_link')
            lookat = 0.5 * (p_base + p_tcp)
        except Exception:
            lookat = np.array([0.25, 0.0, 0.95], dtype=np.float32)
        self.env.init_viewer(
            distance=0.5,
            azimuth=180,
            elevation=-20,
            lookat=lookat,
            transparent=False,
            black_sky=True,
            use_rgb_overlay=False,
            loc_rgb_overlay='top right',
        )
    def reset(self, seed = None):
        '''
        Reset the environment
        Move the robot to a initial position, set the object positions based on the seed
        '''
        if seed is not None:
            np.random.seed(seed=seed)
        q_init = np.deg2rad([0,0,0,0,0,0])
        q_zero,ik_err_stack,ik_info = solve_ik(
            env = self.env,
            joint_names_for_ik = self.joint_names,
            body_name_trgt     = 'tcp_link',
            q_init       = q_init, # ik from zero pose
            p_trgt       = np.array([0.3,0.0,1.0]),
            R_trgt       = rpy2r(np.deg2rad([90,-0.,90 ])),
        )
        self.env.forward(q=q_zero,joint_names=self.joint_names,increase_tick=False)
        
        # set plate position
        plate_xyz = np.array([0.3, -0.25, 0.82])
        self.env.set_p_base_body(body_name=self.plate_body_name,p=plate_xyz)
        self.env.set_R_base_body(body_name=self.plate_body_name,R=np.eye(3,3))
        # Set object positions with constrained spawn sampling.
        try:
            obj_xyzs = self._sample_object_xyzs(n_obj=2, min_dist=self.spawn_min_dist)
        except ValueError as exc:
            fallback_min_dist = self.spawn_fallback_min_dist
            print(f"[SimpleEnv2.reset] object sampling failed: {exc}")
            print(f"[SimpleEnv2.reset] retrying with min_dist={fallback_min_dist}")
            obj_xyzs = self._sample_object_xyzs(n_obj=2, min_dist=fallback_min_dist)

        self.env.set_p_base_body(body_name=self.mug_red_body_name, p=obj_xyzs[0, :])
        self.env.set_R_base_body(body_name=self.mug_red_body_name, R=np.eye(3, 3))
        self.env.set_p_base_body(body_name=self.mug_blue_body_name, p=obj_xyzs[1, :])
        self.env.set_R_base_body(body_name=self.mug_blue_body_name, R=np.eye(3, 3))
        self.env.forward(increase_tick=False)

        # Set the initial pose of the robot
        self.last_q = copy.deepcopy(q_zero)
        self.q = np.concatenate([q_zero, np.array([0.0]*4)])
        self.p0, self.R0 = self.env.get_pR_body(body_name='tcp_link')
        mug_red_init_pose, mug_blue_init_pose, plate_init_pose = self.get_obj_pose()
        self.obj_init_pose = np.concatenate([mug_red_init_pose, mug_blue_init_pose, plate_init_pose],dtype=np.float32)
        for _ in range(100):
            self.step_env()
        self.set_instruction()
        print("DONE INITIALIZATION")
        self.gripper_state = False
        self.past_chars = []

    def set_instruction(self, given = None):
        """
        Set the instruction for the task
        """
        if given is None:
            obj_candidates = ['red', 'blue']
            obj1 = random.choice(obj_candidates)
            self.instruction = f'Place the {obj1} mug on the plate.'
            if obj1 == 'red':
                self.obj_target = self.mug_red_body_name
            else:
                self.obj_target = self.mug_blue_body_name
        else:
            self.instruction = given
            if 'red' in self.instruction:
                self.obj_target = self.mug_red_body_name
            elif 'blue' in self.instruction:
                self.obj_target = self.mug_blue_body_name
            else:
                raise ValueError('Instruction does not contain a valid object color (red or blue).')

    def step(self, action):
        '''
        Take a step in the environment
        args:
            action: np.array of shape (7,), action to take
        returns:
            state: np.array, state of the environment after taking the action
                - ee_pose: [px,py,pz,r,p,y]
                - joint_angle: [j1,j2,j3,j4,j5,j6]

        '''
        if self.action_type == 'eef_pose':
            q = self.env.get_qpos_joints(joint_names=self.joint_names)
            self.p0 += action[:3]
            self.R0 = self.R0.dot(rpy2r(action[3:6]))
            q ,ik_err_stack,ik_info = solve_ik(
                env                = self.env,
                joint_names_for_ik = self.joint_names,
                body_name_trgt     = 'tcp_link',
                q_init             = q,
                p_trgt             = self.p0,
                R_trgt             = self.R0,
                max_ik_tick        = 50,
                ik_stepsize        = 1.0,
                ik_eps             = 1e-2,
                ik_th              = np.radians(5.0),
                render             = False,
                verbose_warning    = False,
                restore_state      = False,
            )
        elif self.action_type == 'delta_joint_angle':
            q = action[:-1] + self.last_q
        elif self.action_type == 'joint_angle':
            q = action[:-1]
        else:
            raise ValueError('action_type not recognized')
        
        gripper_cmd = np.array([action[-1]]*4)
        gripper_cmd[[1,3]] *= 0.8
        self.compute_q = q
        q = np.concatenate([q, gripper_cmd])

        self.q = q
        if self.state_type == 'joint_angle':
            return self.get_joint_state()
        elif self.state_type == 'ee_pose':
            return self.get_ee_pose()
        elif self.state_type == 'delta_q' or self.action_type == 'delta_joint_angle':
            dq =  self.get_delta_q()
            return dq
        else:
            raise ValueError('state_type not recognized')

    def step_env(self):
        self.env.step(self.q)

    def grab_image(self):
        '''
        grab images from the environment
        returns:
            rgb_agent: np.array, rgb image from the top view
            rgb_aux: np.array, compatibility secondary image (same as top view)
        '''
        self.rgb_agent = self.env.get_fixed_cam_rgb(
            cam_name='topview')
        # Egocentric camera was removed from the SO101 scene. Keep a secondary
        # return image for backward compatibility with existing scripts.
        self.rgb_ego = self.rgb_agent.copy()
        # self.rgb_top = self.env.get_fixed_cam_rgbd_pcd(
        #     cam_name='topview')
        self.rgb_side = self.env.get_fixed_cam_rgb(
            cam_name='sideview')
        return self.rgb_agent, self.rgb_ego
        

    def render(self, teleop=False, idx = 0):
        '''
        Render the environment
        '''
        self.env.plot_time()
        p_current, R_current = self.env.get_pR_body(body_name='tcp_link')
        R_current = R_current @ np.array([[1,0,0],[0,0,1],[0,1,0 ]])
        self.env.plot_sphere(p=p_current, r=0.02, rgba=[0.95,0.05,0.05,0.5])
        self.env.plot_capsule(p=p_current, R=R_current, r=0.01, h=0.2, rgba=[0.05,0.95,0.05,0.5])
        rgb_agent_view = add_title_to_img(self.rgb_agent,text='Top View',shape=(640,480))
        self.env.plot_T(p = np.array([0.1,0.0,1.0]), label=f"Episode {idx}", plot_axis=False, plot_sphere=False)
        self.env.viewer_rgb_overlay(rgb_agent_view,loc='top right')
        if teleop:
            rgb_side_view = add_title_to_img(self.rgb_side,text='Side View',shape=(640,480))
            self.env.viewer_rgb_overlay(rgb_side_view, loc='top left')
            self.env.viewer_text_overlay(text1='Key Pressed',text2='%s'%(self.env.get_key_pressed_list(as_text=True)))
            self.env.viewer_text_overlay(text1='Key Repeated',text2='%s'%(self.env.get_key_repeated_list(as_text=True)))
            focused, active_keys = self._teleop_debug_status()
            self.env.viewer_text_overlay(text1='Window Focus', text2='YES' if focused else 'NO (click viewer)')
            self.env.viewer_text_overlay(text1='Active Keys (raw)', text2='%s' % active_keys)
        if getattr(self, 'instruction', None) is not None:
            language_instructions = self.instruction
            self.env.viewer_text_overlay(text1='Language Instructions',text2=language_instructions)
        self.env.render()

    def get_joint_state(self):
        '''
        Get the joint state of the robot
        returns:
            q: np.array, joint angles of the robot + gripper state (0 for open, 1 for closed)
            [j1,j2,j3,j4,j5,j6,gripper]
        '''
        qpos = self.env.get_qpos_joints(joint_names=self.joint_names)
        gripper = self.env.get_qpos_joint('rh_r1')
        gripper_cmd = 1.0 if gripper[0] > 0.5 else 0.0
        return np.concatenate([qpos, [gripper_cmd]],dtype=np.float32)
    
    def teleop_robot(self):
        '''
        Teleoperate the robot using keyboard
        returns:
            action: np.array, action to take
            done: bool, True if the user wants to reset the teleoperation
        
        Keys:
            ---------     -----------------------
               w       ->        backward
            s  a  d        left   forward   right
            ---------      -----------------------
            In x, y plane

            ---------
            R: Moving Up
            F: Moving Down
            ---------
            In z axis

            ---------
            Q: Tilt left
            E: Tilt right
            UP: Look Upward
            Down: Look Donward
            Right: Turn right
            Left: Turn left
            ---------
            For rotation

            ---------
            z: reset
            SPACEBAR: gripper open/close
            ---------   


        '''
        # char = self.env.get_key_pressed()
        dpos = np.zeros(3)
        drot = np.eye(3)
        if self._key_is_down(glfw.KEY_S):
            dpos += np.array([0.007,0.0,0.0])
        if self._key_is_down(glfw.KEY_W):
            dpos += np.array([-0.007,0.0,0.0])
        if self._key_is_down(glfw.KEY_A):
            dpos += np.array([0.0,-0.007,0.0])
        if self._key_is_down(glfw.KEY_D):
            dpos += np.array([0.0,0.007,0.0])
        if self._key_is_down(glfw.KEY_R):
            dpos += np.array([0.0,0.0,0.007])
        if self._key_is_down(glfw.KEY_F):
            dpos += np.array([0.0,0.0,-0.007])
        if  self._key_is_down(glfw.KEY_LEFT):
            drot = rotation_matrix(angle=0.1 * 0.3, direction=[0.0, 1.0, 0.0])[:3, :3]
        if  self._key_is_down(glfw.KEY_RIGHT):
            drot = rotation_matrix(angle=-0.1 * 0.3, direction=[0.0, 1.0, 0.0])[:3, :3]
        if self._key_is_down(glfw.KEY_DOWN):
            drot = rotation_matrix(angle=0.1 * 0.3, direction=[1.0, 0.0, 0.0])[:3, :3]
        if self._key_is_down(glfw.KEY_UP):
            drot = rotation_matrix(angle=-0.1 * 0.3, direction=[1.0, 0.0, 0.0])[:3, :3]
        if self._key_is_down(glfw.KEY_Q):
            drot = rotation_matrix(angle=0.1 * 0.3, direction=[0.0, 0.0, 1.0])[:3, :3]
        if self._key_is_down(glfw.KEY_E):
            drot = rotation_matrix(angle=-0.1 * 0.3, direction=[0.0, 0.0, 1.0])[:3, :3]
        if self._key_is_pressed_once(glfw.KEY_Z):
            return np.zeros(7, dtype=np.float32), True
        if self._key_is_pressed_once(glfw.KEY_SPACE):
            self.gripper_state =  not  self.gripper_state
        drot = r2rpy(drot)
        action = np.concatenate([dpos, drot, np.array([self.gripper_state],dtype=np.float32)],dtype=np.float32)
        return action, False
    
    def get_delta_q(self):
        '''
        Get the delta joint angles of the robot
        returns:
            delta: np.array, delta joint angles of the robot + gripper state (0 for open, 1 for closed)
            [dj1,dj2,dj3,dj4,dj5,dj6,gripper]
        '''
        delta = self.compute_q - self.last_q
        self.last_q = copy.deepcopy(self.compute_q)
        gripper = self.env.get_qpos_joint('rh_r1')
        gripper_cmd = 1.0 if gripper[0] > 0.5 else 0.0
        return np.concatenate([delta, [gripper_cmd]],dtype=np.float32)

    def check_success(self):
        '''
        Check task completion for language-conditioned episodes.

        If the target body is a bin (name contains "bin"), apply the same
        strict SO101 bin logic used in y_env.py:
          1) object center must be inside bin inner XY bounds
          2) object center Z must be inside bin cavity bounds
          3) gripper must be open

        Otherwise (legacy plate task), require close XY/Z alignment to plate,
        gripper open, and arm retracted upward.
        '''
        p_obj = self.env.get_p_body(self.obj_target)
        p_tgt = self.env.get_p_body(self.plate_body_name)
        gripper_open = float(self.env.get_qpos_joint('rh_r1')[0]) < 0.1

        # Support bin tasks in language env with strict "drop-in-bin only" criteria.
        if 'bin' in self.plate_body_name.lower():
            inner_half_extent = 0.048
            block_half_size = 0.0125
            floor_top = p_tgt[2] + 0.012
            wall_top = p_tgt[2] + 0.076

            dx = abs(p_obj[0] - p_tgt[0])
            dy = abs(p_obj[1] - p_tgt[1])
            xy_ok = (dx < (inner_half_extent - 0.003)) and (dy < (inner_half_extent - 0.003))

            z_min = floor_top + block_half_size - 0.003
            z_max = wall_top + 0.005
            z_ok = (p_obj[2] > z_min) and (p_obj[2] < z_max)

            contact_pairs = self.env.get_contact_body_names()

            def _has_contact(body_a, body_b):
                for c1, c2 in contact_pairs:
                    if (c1 == body_a and c2 == body_b) or (c1 == body_b and c2 == body_a):
                        return True
                return False

            block_bin_contact = _has_contact(self.obj_target, self.plate_body_name)
            # OMY gripper finger bodies in this env:
            # - rh_p12_rn_r1 / rh_p12_rn_r2
            # - rh_p12_rn_l1 / rh_p12_rn_l2
            block_gripper_contact = any(
                _has_contact(self.obj_target, body_name)
                for body_name in ['rh_p12_rn_r1', 'rh_p12_rn_r2', 'rh_p12_rn_l1', 'rh_p12_rn_l2']
            )

            return bool(xy_ok and z_ok and gripper_open and block_bin_contact and not block_gripper_contact)

        # Legacy plate-placement logic for mug tasks.
        xy_ok = np.linalg.norm(p_obj[:2] - p_tgt[:2]) < 0.10
        z_ok = abs(p_obj[2] - p_tgt[2]) < 0.08
        if not (xy_ok and z_ok and gripper_open):
            return False
        return bool(self.env.get_p_body('tcp_link')[2] > 0.9)
    
    def get_obj_pose(self):
        '''
        returns: 
            p_mug_red: np.array, position of the red mug
            p_mug_blue: np.array, position of the blue mug
            p_plate: np.array, position of the plate
        '''
        p_mug_red = self.env.get_p_body(self.mug_red_body_name)
        p_mug_blue = self.env.get_p_body(self.mug_blue_body_name)
        p_plate = self.env.get_p_body(self.plate_body_name)

        return p_mug_red, p_mug_blue, p_plate
    
    def set_obj_pose(self, p_mug_red, p_mug_blue, p_plate):
        '''
        Set the object poses
        args:
            p_mug_red: np.array, position of the red mug
            p_mug_blue: np.array, position of the blue mug
            p_plate: np.array, position of the plate
        '''
        self.env.set_p_base_body(body_name=self.mug_red_body_name,p=p_mug_red)
        self.env.set_R_base_body(body_name=self.mug_red_body_name,R=np.eye(3,3))
        self.env.set_p_base_body(body_name=self.mug_blue_body_name,p=p_mug_blue)
        self.env.set_R_base_body(body_name=self.mug_blue_body_name,R=np.eye(3,3))
        self.env.set_p_base_body(body_name=self.plate_body_name,p=p_plate)
        self.env.set_R_base_body(body_name=self.plate_body_name,R=np.eye(3,3))
        self.step_env()


    def get_ee_pose(self):
        '''
        get the end effector pose of the robot + gripper state
        '''
        p, R = self.env.get_pR_body(body_name='tcp_link')
        rpy = r2rpy(R)
        return np.concatenate([p, rpy],dtype=np.float32)