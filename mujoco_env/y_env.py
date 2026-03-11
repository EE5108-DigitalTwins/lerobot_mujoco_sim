import sys
import random
import numpy as np
import xml.etree.ElementTree as ET
from mujoco_env.mujoco_parser import MuJoCoParserClass
from mujoco_env.utils import prettify, sample_xyzs, rotation_matrix, add_title_to_img
from mujoco_env.ik import solve_ik
from mujoco_env.transforms import rpy2r, r2rpy
import os
import copy
import glfw

class SimpleEnv:
    def __init__(self, 
                 xml_path,
                action_type='eef_pose', 
                state_type='joint_angle',
                robot_profile='omy',
                seed = None,
                mug_body_name='body_obj_mug_5',
                plate_body_name='body_obj_plate_11',
                spawn_x_range=(0.30, 0.46),
                spawn_y_range=(-0.2, 0.2),
                spawn_z_range=(0.82, 0.82),
                spawn_min_dist=0.2,
                spawn_xy_margin=0.0,
                spawn_fallback_min_dist=0.1):
        """
        args:
            xml_path: str, path to the xml file
            action_type: str, type of action space, 'eef_pose','delta_joint_angle' or 'joint_angle'
            state_type: str, type of state space, 'joint_angle' or 'ee_pose'
            robot_profile: str, robot kinematic profile ('omy', 'so100', 'so101')
            seed: int, seed for random number generator
            mug_body_name: str, body name for the mug object in the MuJoCo model
            plate_body_name: str, body name for the plate object in the MuJoCo model
            spawn_x_range: tuple[float, float], x sampling range for objects
            spawn_y_range: tuple[float, float], y sampling range for objects
            spawn_z_range: tuple[float, float], z sampling range for objects
            spawn_min_dist: float, minimum distance between sampled objects
            spawn_xy_margin: float, xy margin applied to ranges during sampling
            spawn_fallback_min_dist: float, retry distance when initial sampling fails
        """
        # Load the xml file
        self.env = MuJoCoParserClass(name='Tabletop',rel_xml_path=xml_path)
        self.action_type = action_type
        self.state_type = state_type
        self.mug_body_name = mug_body_name
        self.plate_body_name = plate_body_name
        self.spawn_x_range = spawn_x_range
        self.spawn_y_range = spawn_y_range
        self.spawn_z_range = spawn_z_range
        self.spawn_min_dist = spawn_min_dist
        self.spawn_xy_margin = spawn_xy_margin
        self.spawn_fallback_min_dist = spawn_fallback_min_dist
        self.robot_profile = robot_profile

        if robot_profile == 'omy':
            self.joint_names = ['joint1',
                        'joint2',
                        'joint3',
                        'joint4',
                        'joint5',
                        'joint6',]
            self.tcp_body_name = 'tcp_link'
            self.gripper_joint_name = 'rh_r1'
            self.gripper_actuator_scales = np.array([1.0, 0.8, 1.0, 0.8], dtype=np.float32)
        elif robot_profile in ['so100', 'so101']:
            self.joint_names = [
                'shoulder_pan',
                'shoulder_lift',
                'elbow_flex',
                'wrist_flex',
                'wrist_roll',
            ]
            self.tcp_body_name = 'gripper'
            self.gripper_joint_name = 'gripper'
            self.gripper_actuator_scales = np.array([1.0], dtype=np.float32)
        else:
            raise ValueError(f"Unsupported robot_profile: {robot_profile}")

        self.n_arm_joints = len(self.joint_names)
        self.n_gripper_actuators = len(self.gripper_actuator_scales)
        self._prev_key_state = {}
        self._teleop_debug_counter = 0
        self.joint_mins = np.array([self.env.model.joint(name).range[0] for name in self.joint_names], dtype=np.float32)
        self.joint_maxs = np.array([self.env.model.joint(name).range[1] for name in self.joint_names], dtype=np.float32)

        available_joint_names = set(self.env.joint_names)
        missing_joints = [jname for jname in self.joint_names if jname not in available_joint_names]
        if missing_joints:
            raise ValueError(
                f"robot_profile='{self.robot_profile}' is incompatible with xml_path='{xml_path}'. "
                f"Missing joints: {missing_joints}. "
                f"Available joints: {self.env.joint_names}"
            )

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
        self.env.init_viewer(
            distance          = 2.0,
            elevation         = -30, 
            transparent       = False,
            black_sky         = True,
            use_rgb_overlay = False,
            loc_rgb_overlay = 'top right',
        )
    def reset(self, seed = None):
        '''
        Reset the environment
        Move the robot to a initial position, set the object positions based on the seed
        '''
        if seed is not None:
            np.random.seed(seed=seed)
        
        # Set robot-specific initial configurations and home positions
        # SO101 uses joint control, skip IK and use direct joint positions
        if self.robot_profile == 'so101' and self.action_type == 'delta_joint_angle':
            # Direct joint configuration for SCARA-like robot
            q_zero = np.array([0.0, 0.8, -0.8, 0.0, 0.0], dtype=np.float32)
            self.env.forward(q=q_zero, joint_names=self.joint_names, increase_tick=False)
        else:
            # Use IK for Cartesian-controlled robots
            if self.robot_profile == 'so100':
                q_init = np.zeros(self.n_arm_joints, dtype=np.float32)
                p_trgt = np.array([0.25, -0.35, 0.95])
                R_trgt = rpy2r(np.deg2rad([90, 0, 90]))
            else:  # omy
                q_init = np.zeros(self.n_arm_joints, dtype=np.float32)
                p_trgt = np.array([0.3, 0.0, 1.0])
                R_trgt = rpy2r(np.deg2rad([90, 0, 90]))
                
            q_zero,ik_err_stack,ik_info = solve_ik(
                env = self.env,
                joint_names_for_ik = self.joint_names,
                body_name_trgt     = self.tcp_body_name,
                q_init       = q_init,
                p_trgt       = p_trgt,
                R_trgt       = R_trgt,
            )
            self.env.forward(q=q_zero,joint_names=self.joint_names,increase_tick=False)

        # Set object positions
        obj_names = self.env.get_body_names(prefix='body_obj_')
        n_obj = len(obj_names)
        x_range = self.spawn_x_range
        y_range = self.spawn_y_range
        z_range = self.spawn_z_range
        min_dist = self.spawn_min_dist
        xy_margin = self.spawn_xy_margin
        try:
            obj_xyzs = sample_xyzs(
                n_obj,
                x_range=x_range,
                y_range=y_range,
                z_range=z_range,
                min_dist=min_dist,
                xy_margin=xy_margin,
            )
        except ValueError as exc:
            fallback_min_dist = self.spawn_fallback_min_dist
            print(f"[SimpleEnv.reset] object sampling failed: {exc}")
            print(f"[SimpleEnv.reset] retrying with min_dist={fallback_min_dist}")
            obj_xyzs = sample_xyzs(
                n_obj,
                x_range=x_range,
                y_range=y_range,
                z_range=z_range,
                min_dist=fallback_min_dist,
                xy_margin=xy_margin,
            )
        for obj_idx in range(n_obj):
            self.env.set_p_base_body(body_name=obj_names[obj_idx],p=obj_xyzs[obj_idx,:])
            self.env.set_R_base_body(body_name=obj_names[obj_idx],R=np.eye(3,3))

        # Clear residual dynamics so newly spawned objects start at rest
        self.env.data.qvel[:] = 0.0
        self.env.data.qacc[:] = 0.0
        if self.env.data.ctrl is not None and self.env.data.ctrl.size > 0:
            self.env.data.ctrl[:] = 0.0
        self.env.forward(increase_tick=False)

        # Let objects settle under gravity before starting teleoperation
        if self.env.data.ctrl is not None and self.env.data.ctrl.size > 0:
            self.env.data.ctrl[:] = 0.0
        for _ in range(20):
            self.env.step(np.zeros(self.env.n_ctrl))

        # Set the initial pose of the robot
        self.last_q = copy.deepcopy(q_zero)
        self.prev_q = copy.deepcopy(q_zero)
        self.q = np.concatenate([q_zero, np.zeros(self.n_gripper_actuators, dtype=np.float32)])
        self.p0, self.R0 = self.env.get_pR_body(body_name=self.tcp_body_name)
        mug_init_pose, plate_init_pose = self.get_obj_pose()
        self.obj_init_pose = np.concatenate([mug_init_pose, plate_init_pose],dtype=np.float32)
        for _ in range(100):
            self.step_env()
        print("DONE INITIALIZATION")
        self.gripper_state = False
        self.past_chars = []

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
        prev_q = copy.deepcopy(self.last_q)
        if self.action_type == 'eef_pose':
            q = self.env.get_qpos_joints(joint_names=self.joint_names)
            self.p0 += action[:3]
            self.R0 = self.R0.dot(rpy2r(action[3:6]))
            
            # Adjust IK parameters based on robot profile
            if self.robot_profile == 'so100':
                # SO100 needs moderate damping
                ik_stepsize = 0.3
                ik_eps = 0.5
                max_ik_tick = 50
            else:  # omy
                ik_stepsize = 1.0
                ik_eps = 1e-2
                max_ik_tick = 50
            
            q ,ik_err_stack,ik_info = solve_ik(
                env                = self.env,
                joint_names_for_ik = self.joint_names,
                body_name_trgt     = self.tcp_body_name,
                q_init             = q,
                p_trgt             = self.p0,
                R_trgt             = self.R0,
                max_ik_tick        = max_ik_tick,
                ik_stepsize        = ik_stepsize,
                ik_eps             = ik_eps,
                ik_th              = np.radians(5.0),
                render             = False,
                verbose_warning    = False,
            )
        elif self.action_type == 'delta_joint_angle':
            q = action[:self.n_arm_joints] + prev_q
        elif self.action_type == 'joint_angle':
            q = action[:self.n_arm_joints]
        else:
            raise ValueError('action_type not recognized')

        q = np.clip(q, self.joint_mins, self.joint_maxs)
        self.prev_q = prev_q
        self.last_q = copy.deepcopy(q)
        
        gripper_cmd = np.array([action[-1]] * self.n_gripper_actuators, dtype=np.float32)
        gripper_cmd *= self.gripper_actuator_scales
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
            rgb_agent: np.array, rgb image from the agent's view
            rgb_ego: np.array, rgb image from the egocentric
        '''
        self.rgb_agent = self.env.get_fixed_cam_rgb(
            cam_name='agentview')
        self.rgb_ego = self.env.get_fixed_cam_rgb(
            cam_name='egocentric')
        # self.rgb_top = self.env.get_fixed_cam_rgbd_pcd(
        #     cam_name='topview')
        self.rgb_side = self.env.get_fixed_cam_rgb(
            cam_name='sideview')
        return self.rgb_agent, self.rgb_ego
        

    def render(self, teleop=False):
        '''
        Render the environment
        '''
        self.env.plot_time()
        p_current, R_current = self.env.get_pR_body(body_name=self.tcp_body_name)
        R_current = R_current @ np.array([[1,0,0],[0,0,1],[0,1,0 ]])
        self.env.plot_sphere(p=p_current, r=0.02, rgba=[0.95,0.05,0.05,0.5])
        self.env.plot_capsule(p=p_current, R=R_current, r=0.01, h=0.2, rgba=[0.05,0.95,0.05,0.5])
        rgb_egocentric_view = add_title_to_img(self.rgb_ego,text='Egocentric View',shape=(640,480))
        rgb_agent_view = add_title_to_img(self.rgb_agent,text='Agent View',shape=(640,480))
        
        self.env.viewer_rgb_overlay(rgb_agent_view,loc='top right')
        self.env.viewer_rgb_overlay(rgb_egocentric_view,loc='bottom right')
        if teleop:
            rgb_side_view = add_title_to_img(self.rgb_side,text='Side View',shape=(640,480))
            self.env.viewer_rgb_overlay(rgb_side_view, loc='top left')
            self.env.viewer_text_overlay(text1='Key Pressed',text2='%s'%(self.env.get_key_pressed_list()))
            self.env.viewer_text_overlay(text1='Key Repeated',text2='%s'%(self.env.get_key_repeated_list()))
            focused, active_keys = self._teleop_debug_status()
            self.env.viewer_text_overlay(text1='Window Focus', text2='YES' if focused else 'NO (click viewer)')
            self.env.viewer_text_overlay(text1='Active Keys (raw)', text2='%s' % active_keys)
        self.env.render()

    def get_joint_state(self):
        '''
        Get the joint state of the robot
        returns:
            q: np.array, joint angles of the robot + gripper state (0 for open, 1 for closed)
            [j1,j2,j3,j4,j5,j6,gripper]
        '''
        qpos = self.env.get_qpos_joints(joint_names=self.joint_names)
        gripper = self.env.get_qpos_joint(self.gripper_joint_name)
        gripper_cmd = 1.0 if gripper[0] > 0.5 else 0.0
        return np.concatenate([qpos, [gripper_cmd]],dtype=np.float32)
    
    def teleop_robot(self):
        '''
        Teleoperate the robot using keyboard
        returns:
            action: np.array, action to take
            done: bool, True if the user wants to reset the teleoperation
        
        Keys:
            Movement:
            W: Forward
            S: Backward
            A: Left
            D: Right
            R: Up
            F: Down

            Rotation:
            Q: Yaw left
            E: Yaw right
            UP: Pitch up
            DOWN: Pitch down
            LEFT: Roll left
            RIGHT: Roll right

            ---------
            z: reset
            SPACEBAR: gripper open/close
            ---------   


        '''
        # For SCARA-like robots (SO101), use joint-space control
        if self.action_type == 'delta_joint_angle':
            return self._teleop_joint_space()
        
        # Standard Cartesian control for other robots
        dpos = np.zeros(3)
        drot = np.eye(3)
        
        # Collect all WASD and RF inputs
        # These deltas are in world frame, which is what the IK solver expects
        # WASD: Horizontal plane movement (X-Y)
        # R/F: Vertical movement (Z)
        # Q/E: Yaw rotation
        # Arrow keys: Pitch/Roll rotation
        if self._key_is_down(glfw.KEY_W):
            dpos += np.array([0.007,0.0,0.0])  # Forward (+X)
        if self._key_is_down(glfw.KEY_S):
            dpos += np.array([-0.007,0.0,0.0])  # Backward (-X)
        if self._key_is_down(glfw.KEY_A):
            dpos += np.array([0.0,-0.007,0.0])  # Left (-Y)
        if self._key_is_down(glfw.KEY_D):
            dpos += np.array([0.0,0.007,0.0])  # Right (+Y)
        if self._key_is_down(glfw.KEY_R):
            dpos += np.array([0.0,0.0,0.007])  # Up (+Z)
        if self._key_is_down(glfw.KEY_F):
            dpos += np.array([0.0,0.0,-0.007])  # Down (-Z)
        if self._key_is_down(glfw.KEY_Q):
            drot = rotation_matrix(angle=0.1 * 0.3, direction=[0.0, 0.0, 1.0])[:3, :3]  # Yaw left
        if self._key_is_down(glfw.KEY_E):
            drot = rotation_matrix(angle=-0.1 * 0.3, direction=[0.0, 0.0, 1.0])[:3, :3]  # Yaw right
        if  self._key_is_down(glfw.KEY_LEFT):
            drot = rotation_matrix(angle=0.1 * 0.3, direction=[0.0, 1.0, 0.0])[:3, :3]  # Roll left
        if  self._key_is_down(glfw.KEY_RIGHT):
            drot = rotation_matrix(angle=-0.1 * 0.3, direction=[0.0, 1.0, 0.0])[:3, :3]  # Roll right
        if self._key_is_down(glfw.KEY_UP):
            drot = rotation_matrix(angle=0.1 * 0.3, direction=[1.0, 0.0, 0.0])[:3, :3]  # Pitch up
        if self._key_is_down(glfw.KEY_DOWN):
            drot = rotation_matrix(angle=-0.1 * 0.3, direction=[1.0, 0.0, 0.0])[:3, :3]  # Pitch down
        if self._key_is_pressed_once(glfw.KEY_Z):
            return np.zeros(7, dtype=np.float32), True
        if self._key_is_pressed_once(glfw.KEY_SPACE):
            self.gripper_state =  not  self.gripper_state
        drot = r2rpy(drot)
        action = np.concatenate([dpos, drot, np.array([self.gripper_state],dtype=np.float32)],dtype=np.float32)
        self._teleop_debug_counter += 1
        if np.linalg.norm(action[:6]) > 1e-8 or self._teleop_debug_counter % 200 == 0:
            focused, active_keys = self._teleop_debug_status()
            print(f"[teleop-debug] focused={focused} keys={active_keys} action={np.round(action, 4)}")
        return action, False
    
    def _teleop_joint_space(self):
        '''
        Joint-space teleoperation for SCARA-like robots (SO101)
        Maps WASD/RF to joint velocities directly
        '''
        dq = np.zeros(self.n_arm_joints, dtype=np.float32)
        joint_vel = 0.03  # Joint velocity magnitude
        
        # W/S: shoulder lift
        if self._key_is_down(glfw.KEY_W):
            dq[1] += joint_vel  # Shoulder lift up
        if self._key_is_down(glfw.KEY_S):
            dq[1] -= joint_vel  # Shoulder lift down

        # A/D: left-right (shoulder pan)
        if self._key_is_down(glfw.KEY_A):
            dq[0] -= joint_vel  # Shoulder pan left
        if self._key_is_down(glfw.KEY_D):
            dq[0] += joint_vel  # Shoulder pan right

        # UP/DOWN: up-down (wrist flex)
        if self._key_is_down(glfw.KEY_UP):
            dq[3] += joint_vel  # Wrist flex up
        if self._key_is_down(glfw.KEY_DOWN):
            dq[3] -= joint_vel  # Wrist flex down
        
        # RF: Control elbow
        if self._key_is_down(glfw.KEY_R):
            dq[2] += joint_vel  # Elbow flex up
        if self._key_is_down(glfw.KEY_F):
            dq[2] -= joint_vel  # Elbow flex down
        
        # LEFT/RIGHT: Control wrist roll
        if self._key_is_down(glfw.KEY_LEFT):
            dq[4] += joint_vel  # Wrist roll
        if self._key_is_down(glfw.KEY_RIGHT):
            dq[4] -= joint_vel  # Wrist roll
        
        # UP/DOWN reserved for future use or finer control
        
        # Reset and gripper
        if self._key_is_pressed_once(glfw.KEY_Z):
            return np.zeros(self.n_arm_joints + 1, dtype=np.float32), True
        if self._key_is_pressed_once(glfw.KEY_SPACE):
            self.gripper_state = not self.gripper_state
        
        # Combine joint deltas with gripper state
        action = np.concatenate([dq, np.array([self.gripper_state], dtype=np.float32)])
        self._teleop_debug_counter += 1
        if np.linalg.norm(dq) > 1e-8 or self._teleop_debug_counter % 200 == 0:
            focused, active_keys = self._teleop_debug_status()
            print(f"[teleop-debug] focused={focused} keys={active_keys} dq={np.round(dq, 4)} gripper={self.gripper_state}")
        return action, False
    
    def get_delta_q(self):
        '''
        Get the delta joint angles of the robot
        returns:
            delta: np.array, delta joint angles of the robot + gripper state (0 for open, 1 for closed)
            [dj1,dj2,dj3,dj4,dj5,dj6,gripper]
        '''
        delta = self.compute_q - self.prev_q
        gripper = self.env.get_qpos_joint(self.gripper_joint_name)
        gripper_cmd = 1.0 if gripper[0] > 0.5 else 0.0
        return np.concatenate([delta, [gripper_cmd]],dtype=np.float32)

    def check_success(self):
        '''
        ['body_obj_mug_5', 'body_obj_plate_11']
        Check if the mug is placed on the plate
        + Gripper should be open and move upward above 0.9
        '''
        p_mug = self.env.get_p_body(self.mug_body_name)
        p_plate = self.env.get_p_body(self.plate_body_name)
        if np.linalg.norm(p_mug[:2] - p_plate[:2]) < 0.1 and np.linalg.norm(p_mug[2] - p_plate[2]) < 0.6 and self.env.get_qpos_joint(self.gripper_joint_name) < 0.1:
            p = self.env.get_p_body(self.tcp_body_name)[2]
            if p > 0.9:
                return True
        return False
    
    def get_obj_pose(self):
        '''
        returns: 
            p_mug: np.array, position of the mug
            p_plate: np.array, position of the plate
        '''
        p_mug = self.env.get_p_body(self.mug_body_name)
        p_plate = self.env.get_p_body(self.plate_body_name)
        return p_mug, p_plate
    
    def set_obj_pose(self, p_mug, p_plate):
        '''
        Set the object poses
        args:
            p_mug: np.array, position of the mug
            p_plate: np.array, position of the plate
        '''
        self.env.set_p_base_body(body_name=self.mug_body_name,p=p_mug)
        self.env.set_R_base_body(body_name=self.mug_body_name,R=np.eye(3,3))
        self.env.set_p_base_body(body_name=self.plate_body_name,p=p_plate)
        self.env.set_R_base_body(body_name=self.plate_body_name,R=np.eye(3,3))
        self.step_env()


    def get_ee_pose(self):
        '''
        get the end effector pose of the robot + gripper state
        '''
        p, R = self.env.get_pR_body(body_name=self.tcp_body_name)
        rpy = r2rpy(R)
        return np.concatenate([p, rpy],dtype=np.float32)