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
        target_min = 2.0 * env.ball_radius
        # Use sigmoid with steep slope to reward being at or above target height.
        # For alpha=20, reward ~0.5 at exactly target_min, ~1.0 above.
        alpha = 20.0
        reward = torch.sigmoid(alpha * (env.base_pos[:, 2] - target_min))
        return 1.0 * reward
    
    
    def _reward_orientation(self):
        env = self.env
        # Penalize horizontal components of projected gravity (should be 0 when upright)
        horizontal_sq = torch.sum(torch.square(env.projected_gravity[:, :2]), dim=1)
        sigma_sq = 0.05
        reward = torch.exp(-horizontal_sq / sigma_sq)
        return 1.0 * reward
    
    
    def _reward_contact(self):
        env = self.env
        # Encourage all feet to be in contact with the ball.
        # A foot is considered in contact if the z-force exceeds a small threshold.
        feet_contact_z = env.contact_forces[:, env.feet_indices, 2]  # (num_envs, num_feet)
        in_contact = (feet_contact_z > 0.1).float()
        # Penalize missing contacts: for each foot not in contact, subtract 0.05.
        penalty = 0.05 * torch.sum(1.0 - in_contact, dim=1)
        return -penalty
    
    
    def _reward_action_rate(self):
        env = self.env
        # Penalize large differences between consecutive joint position targets.
        diff = env.actions - env.last_actions
        rate = torch.sum(torch.square(diff), dim=1)
        return -0.001 * rate
    
    
    def _reward_joint_velocity(self):
        env = self.env
        vel_sq = torch.sum(torch.square(env.dof_vel), dim=1)
        return -0.0001 * vel_sq
    
    
    def _reward_torque(self):
        env = self.env
        torque_sq = torch.sum(torch.square(env.torques), dim=1)
        return -0.00005 * torque_sq
    
    
    def _reward_base_velocity(self):
        env = self.env
        lin_vel_sq = torch.sum(torch.square(env.base_lin_vel), dim=1)
        return -0.01 * lin_vel_sq
    
    
    def _reward_base_angular_velocity(self):
        env = self.env
        ang_vel_sq = torch.sum(torch.square(env.base_ang_vel), dim=1)
        return -0.01 * ang_vel_sq

    # Success criteria as episode length
    def compute_success(self):
        return torch.ones_like(self.env.base_pos[:, 2])
