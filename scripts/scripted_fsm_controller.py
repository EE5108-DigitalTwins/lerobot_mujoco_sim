# scripted_fsm_controller.py
#
# Robust scripted controller for MuJoCo dataset generation.
# Designed for long unattended demo collection (10k+ episodes).
#
# Features
# - declarative phase table
# - IK retry logic
# - timeout protection
# - grasp validation
# - clean state handling

import numpy as np

# -------------------------------------------------------------
# GRASP CALIBRATION
# -------------------------------------------------------------

# How far below the block center the gripper should descend
GRASP_Z_OFFSET = -0.01     # meters (tune: -0.015 to -0.03)

# XY offset between TCP frame and finger centre
GRASP_X_OFFSET = -0.005      # tune if grasp is consistently off
GRASP_Y_OFFSET = 0.0

# Slight approach above grasp
APPROACH_Z_OFFSET = 0.075

# slight noise to add to target positions for robustness (tune: 0.003 to 0.01)
NOISE = 0.003

# Refresh noisy target occasionally (not every tick) to avoid moving-goal slowdown.
TARGET_REFRESH_EVERY = 10
REFRESH_PHASES = {"descend"}

# -------------------------------------------------------------
# PHASE TABLE
# -------------------------------------------------------------

PHASES = [

    dict(
        name="approach_cube",
        target=lambda c, b: [
            c[0] + GRASP_X_OFFSET + np.random.uniform(-NOISE, NOISE),
            c[1] + GRASP_Y_OFFSET + np.random.uniform(-NOISE, NOISE),
            c[2] + APPROACH_Z_OFFSET
        ],
        gripper=0.0,
        tol=0.022,
        timeout=120
    ),

    dict(
        name="descend",
        target=lambda c, b: [
            c[0] + GRASP_X_OFFSET + np.random.uniform(-NOISE, NOISE),
            c[1] + GRASP_Y_OFFSET + np.random.uniform(-NOISE, NOISE),
            c[2] + GRASP_Z_OFFSET
        ],
        gripper=0.0,
        tol=0.017,
        timeout=120
    ),

    dict(
        name="close_gripper",
        target=None,
        gripper=1.0,
        tol=None,
        timeout=20
    ),

    dict(
        name="lift",
        target=lambda c, b: [c[0], c[1], c[2] + 0.10],
        gripper=1.0,
        tol=0.025,
        timeout=120
    ),

    dict(
        name="move_to_bin",
        target=lambda c, b: [b[0], b[1] + 0.02, b[2] + 0.07],
        gripper=1.0,
        tol=0.02,
        timeout=150
    ),

    dict(
        name="place",
        target=lambda c, b: [b[0], b[1] + 0.02, b[2] + 0.055],
        gripper=1.0,
        tol=0.02,
        timeout=120
    ),

    dict(
        name="open",
        target=None,
        gripper=0.0,
        tol=None,
        timeout=20
    ),

    dict(
        name="retract",
        target=lambda c, b: [b[0], b[1] + 0.02, b[2] + 0.09],
        gripper=0.0,
        tol=0.025,
        timeout=100
    ),
]

# FSM STATE
# -------------------------------------------------------------

def make_fsm():
    return dict(
        phase=0,
        tick=0,
        planned_phase=None,
        phase_target_xyz=None,
        q_target=None,
        pick_xyz=None,
        bin_xyz=None,
        ik_failures=0,
        lift_fail_streak=0,

        # trajectory state
        traj_start=None,
        traj_target=None,
        traj_progress=0.0,
    )

# -------------------------------------------------------------
# smoothing curve
# -------------------------------------------------------------
def smoothstep(x):
    # More linear smoothing for faster convergence, less oscillation
    return x

# -------------------------------------------------------------
# smoothing curve
# -------------------------------------------------------------

def next_ee_waypoint(env, fsm, target_xyz, get_ee_xyz_fn):
    phase_name = PHASES[fsm["phase"]]["name"]
    if phase_name == "approach_cube":
        traj_speed = 0.03
    elif phase_name == "descend":
        traj_speed = 0.02
    else:
        traj_speed = 0.02

    ee_xyz = get_ee_xyz_fn(env)

    if fsm["traj_start"] is None:
        fsm["traj_start"] = ee_xyz.copy()
        fsm["traj_target"] = target_xyz.copy()
        fsm["traj_progress"] = 0.0

    dist = np.linalg.norm(fsm["traj_target"] - fsm["traj_start"])

    if dist < 1e-5:
        return fsm["traj_target"]

    fsm["traj_progress"] += traj_speed / dist
    fsm["traj_progress"] = min(fsm["traj_progress"], 1.0)

    s = smoothstep(fsm["traj_progress"])

    waypoint = (
        (1 - s) * fsm["traj_start"]
        + s * fsm["traj_target"]
    )

    return waypoint

# -------------------------------------------------------------
# IK PLANNING
# -------------------------------------------------------------

def plan_ik(env, target_xyz, ik_fn):

    for dz in [0.0, -0.01, -0.02, -0.03, 0.01]:

        trial = target_xyz.copy()
        trial[2] += dz

        q = ik_fn(trial)

        if q is None:
            continue

        if not np.any(np.isnan(q)):
            return q

    return None


# -------------------------------------------------------------
# GRASP VALIDATION
# -------------------------------------------------------------

def cube_lifted(env, cube_xyz_initial, cube_fn, min_z_delta=0.01):
    cube_now = cube_fn(env)
    return cube_now[2] > cube_xyz_initial[2] + min_z_delta


# -------------------------------------------------------------
# MAIN FSM STEP
# -------------------------------------------------------------

def fsm_step(
    env,
    fsm,
    get_ee_xyz_fn,
    cube_pos_fn,
    ik_fn,
    fallback_ik_fn,
):

    ee_xyz = get_ee_xyz_fn(env)

    cube_xyz_live = cube_pos_fn(env)
    _, bin_xyz_live = env.get_obj_pose()

    if fsm["pick_xyz"] is None:
        fsm["pick_xyz"] = cube_xyz_live.copy()

    if fsm["bin_xyz"] is None:
        fsm["bin_xyz"] = np.asarray(bin_xyz_live, dtype=np.float32)

    cube_xyz = fsm["pick_xyz"]
    bin_xyz = fsm["bin_xyz"]

    phase_cfg = PHASES[fsm["phase"]]

    target_fn = phase_cfg["target"]
    gripper = phase_cfg["gripper"]

    q_current = env.get_joint_state()[:env.n_arm_joints]

    # -------------------------------------------------
    # target
    # -------------------------------------------------

    should_refresh_target = False
    if fsm["planned_phase"] != fsm["phase"]:
        should_refresh_target = True
    elif phase_cfg["name"] in REFRESH_PHASES and fsm["tick"] > 0:
        should_refresh_target = (fsm["tick"] % TARGET_REFRESH_EVERY) == 0

    if should_refresh_target:
        if target_fn is None:
            fsm["phase_target_xyz"] = None
        else:
            fsm["phase_target_xyz"] = np.asarray(
                target_fn(cube_xyz, bin_xyz),
                dtype=np.float32
            )
        fsm["planned_phase"] = fsm["phase"]

    if target_fn is None:
        target_xyz = ee_xyz.copy()
    else:
        target_xyz = fsm["phase_target_xyz"]

    # -------------------------------------------------
    # plan once per phase
    # -------------------------------------------------

    # if fsm["planned_phase"] != fsm["phase"]:

    #     if q_target is None:

    #         print("[FSM] IK failed, using fallback")

    #         q_target = fallback_ik_fn(env, target_xyz)
    #         fsm["ik_failures"] += 1

    #     else:
    #         fsm["ik_failures"] = 0

    #     fsm["q_target"] = q_target
    #     fsm["planned_phase"] = fsm["phase"]

    waypoint = next_ee_waypoint(env, fsm, target_xyz, get_ee_xyz_fn)

    q_target = plan_ik(env, waypoint, ik_fn)

    if q_target is None:
        # fallback IK solver
        q_target = fallback_ik_fn(env, waypoint)

        if q_target is None:
            fsm["traj_start"] = None
            fsm["traj_progress"] = 0.0
            q_target = q_current.copy()

    # -------------------------------------------------
    # action
    # -------------------------------------------------

    MAX_DQ = 0.04  # radians per control step (balance speed and stability)
    dq = np.clip(q_target - q_current, -MAX_DQ, MAX_DQ)

    action = np.zeros(env.n_arm_joints + 1, dtype=np.float32)
    action[:env.n_arm_joints] = dq
    action[-1] = gripper

    # -------------------------------------------------
    # completion check
    # -------------------------------------------------


    done = False
    if phase_cfg["tol"] is not None:
        dist = np.linalg.norm(ee_xyz - target_xyz)
        done = dist < phase_cfg["tol"]
        # For approach_cube, require also that ee is above block (not just close in 3D)
        if phase_cfg["name"] == "approach_cube":
            # Must be above block by at least 1.5cm
            above_margin = 0.015
            done = done and (ee_xyz[2] > cube_xyz[2] + above_margin)

    if fsm["tick"] > phase_cfg["timeout"]:

        print(f"[FSM] timeout phase {phase_cfg['name']}")
        done = True

    # -------------------------------------------------
    # grasp validation
    # -------------------------------------------------

    if phase_cfg["name"] == "lift":

        LIFT_CHECK_DELAY = 12
        LIFT_MIN_Z_DELTA = 0.004
        LIFT_FAIL_STREAK_MAX = 10

        if fsm["tick"] > LIFT_CHECK_DELAY:
            if cube_lifted(env, cube_xyz, cube_pos_fn, min_z_delta=LIFT_MIN_Z_DELTA):
                fsm["lift_fail_streak"] = 0
            else:
                fsm["lift_fail_streak"] += 1

            if fsm["lift_fail_streak"] >= LIFT_FAIL_STREAK_MAX:
                print("[FSM] grasp failed, restarting")
                fsm["phase"] = 0
                fsm["planned_phase"] = None
                fsm["phase_target_xyz"] = None
                fsm["tick"] = 0
                fsm["traj_start"] = None
                fsm["traj_progress"] = 0.0
                fsm["lift_fail_streak"] = 0
                return action

    # -------------------------------------------------
    # phase transition
    # -------------------------------------------------

    if done and fsm["phase"] < len(PHASES) - 1:

        fsm["phase"] += 1
        fsm["planned_phase"] = None
        fsm["phase_target_xyz"] = None
        fsm["lift_fail_streak"] = 0
        fsm["tick"] = 0
        fsm["traj_start"] = None
        fsm["traj_progress"] = 0.0

    else:

        fsm["tick"] += 1

    return action