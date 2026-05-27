import os
from pathlib import Path
from typing import Union

from params_proto import Meta

from globe_walking.go1_gym.envs.base.legged_robot_config import Cfg


def _go2_urdf_path() -> str:
    env_path = os.getenv("GO2_DESCRIPTION_URDF")
    if env_path:
        return env_path
    workspace_path = Path("/workspace/eureka-workspace/thirdparties/go2_description/urdf/go2_description.urdf")
    if workspace_path.exists():
        return str(workspace_path)
    root_path = Path(__file__).resolve().parents[6] / "thirdparties" / "go2_description" / "urdf" / "go2_description.urdf"
    return str(root_path)


def config_go2(Cnfg: Union[Cfg, Meta]):
    _ = Cnfg.init_state

    # Unitree MuJoCo Go2 home keyframe:
    # qpos joints are [hip, thigh, calf] * 4 with hip=0, thigh=0.9, calf=-1.8.
    _.default_joint_angles = {
        "FL_hip_joint": 0.0,
        "FL_thigh_joint": 0.9,
        "FL_calf_joint": -1.8,
        "FR_hip_joint": 0.0,
        "FR_thigh_joint": 0.9,
        "FR_calf_joint": -1.8,
        "RL_hip_joint": 0.0,
        "RL_thigh_joint": 0.9,
        "RL_calf_joint": -1.8,
        "RR_hip_joint": 0.0,
        "RR_thigh_joint": 0.9,
        "RR_calf_joint": -1.8,
    }

    _ = Cnfg.control
    _.control_type = "P"
    _.stiffness = {"hip": 20.0, "thigh": 20.0, "calf": 40.0}
    _.damping = {"hip": 1.0, "thigh": 1.0, "calf": 2.0}
    _.action_scale = 0.25
    _.hip_scale_reduction = 0.5
    _.decimation = 4

    _ = Cnfg.asset
    _.file = _go2_urdf_path()
    _.foot_name = "foot"
    _.penalize_contacts_on = ["thigh", "calf"]
    _.terminate_after_contacts_on = ["base"]
    _.self_collisions = 0
    _.flip_visual_attachments = False
    _.fix_base_link = False

    _ = Cnfg.rewards
    _.soft_dof_pos_limit = 0.9

    _ = Cnfg.terrain
    _.mesh_type = "trimesh"
    _.measure_heights = False
    _.terrain_noise_magnitude = 0.0
    _.teleport_robots = True
    _.border_size = 50
    _.terrain_proportions = [0, 0, 0, 0, 0, 0, 0, 0, 1.0]
    _.curriculum = False

    _ = Cnfg.env
    _.num_observations = 42

    _ = Cnfg.commands
    _.heading_command = False
    _.resampling_time = 10.0
    _.command_curriculum = False
    _.num_lin_vel_bins = 30
    _.num_ang_vel_bins = 30
    _.lin_vel_x = [-0.6, 0.6]
    _.lin_vel_y = [-0.6, 0.6]
    _.ang_vel_yaw = [-1, 1]
