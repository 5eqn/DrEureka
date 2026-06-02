import torch
import numpy as np
from globe_walking.go1_gym.utils.math_utils import quat_apply_yaw, wrap_to_pi, get_scale_shift
from isaacgym.torch_utils import *

class EurekaReward():
    def __init__(self, env):
        self.env = env

    def load_env(self, env):
        self.env = env
    
    def _reward_height(self):
        env = self.env
        target_height = 2.0 * env.ball_radius
        base_z = env.base_pos[:, 2]
        diff = base_z - target_height
        reward = torch.where(diff >= 0.0, torch.ones_like(base_z),
                             torch.exp(-10.0 * torch.square(diff)))
        return 1.0 * reward
    
    def _reward_orientation(self):
        env = self.env
        horizontal_sq = torch.sum(torch.square(env.projected_gravity[:, :2]), dim=1)
        reward = torch.exp(-horizontal_sq / 0.05)
        return 1.0 * reward
    
    def _reward_contact(self):
        env = self.env
        feet_contact_z = env.contact_forces[:, env.feet_indices, 2]
        in_contact = (feet_contact_z > 0.1).float()
        contact_fraction = torch.mean(in_contact, dim=1)
        return 1.0 * contact_fraction
    
    def _reward_centering(self):
        env = self.env
        # Horizontal distance between robot base and ball center
        base_xy = env.base_pos[:, :2]
        ball_xy = env.object_pos_world_frame[:, :2]
        dist_sq = torch.sum(torch.square(base_xy - ball_xy), dim=1)
        reward = torch.exp(-dist_sq / 0.15)
        return 0.5 * reward
    
    def _reward_action_rate(self):
        env = self.env
        diff = env.actions - env.last_actions
        rate = torch.sum(torch.square(diff), dim=1)
        return -0.005 * rate
    
    def _reward_joint_velocity(self):
        env = self.env
        vel_sq = torch.sum(torch.square(env.dof_vel), dim=1)
        return -0.0005 * vel_sq
    
    def _reward_torque(self):
        env = self.env
        torque_sq = torch.sum(torch.square(env.torques), dim=1)
        return -0.0001 * torque_sq
    
    def _reward_base_velocity(self):
        env = self.env
        lin_vel_sq = torch.sum(torch.square(env.base_lin_vel), dim=1)
        return -0.01 * lin_vel_sq
    
    def _reward_base_angular_velocity(self):
        env = self.env
        ang_vel_sq = torch.sum(torch.square(env.base_ang_vel), dim=1)
        return -0.005 * ang_vel_sq

    # Success criteria as episode length
    def compute_success(self):
        return torch.ones_like(self.env.base_pos[:, 2])

