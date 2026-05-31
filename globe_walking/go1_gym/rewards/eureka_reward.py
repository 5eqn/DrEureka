import torch
import numpy as np
from globe_walking.go1_gym.utils.math_utils import quat_apply_yaw, wrap_to_pi, get_scale_shift
from isaacgym.torch_utils import *

class EurekaReward():
    def __init__(self, env):
        self.env = env

    def load_env(self, env):
        self.env = env
    
    def _reward_z_position(self):
        """Reward for maintaining robot height at or above 2×ball_radius"""
        env = self.env
        # Get robot base height (z-component)
        robot_z = env.base_pos[:, 2]
        # Minimum desired height (2×ball_radius)
        min_height = 2.0 * env.ball_radius
        # Height tracking reward
        height_diff = torch.clamp(min_height - robot_z, min=0.0)
        # Exponential reward: higher when closer to target height, penalize being too low
        height_reward = torch.exp(-height_diff * 10.0)
        return 1.0 * height_reward
    
    def _reward_ball_centering(self):
        """Reward for staying centered above the ball"""
        env = self.env
        # Object local position in robot frame (x,y components)
        object_local_xy = env.object_local_pos[:, :2]
        # Penalize horizontal distance from ball center
        horizontal_dist = torch.norm(object_local_xy, dim=1)
        # Exponential penalty for being off-center
        centering_reward = torch.exp(-horizontal_dist * 5.0)
        return 0.5 * centering_reward
    
    def _reward_angular_stability(self):
        """Penalize excessive roll and pitch angles"""
        env = self.env
        # Projected gravity vector in robot frame (indicates orientation)
        projected_gravity = env.projected_gravity
        # We want gravity vector pointing down (-z direction)
        # Penalize deviation from vertical
        roll_pitch_error = torch.norm(projected_gravity[:, :2], dim=1)
        # Quadratic penalty for angular instability
        stability_penalty = -roll_pitch_error ** 2
        return 0.2 * stability_penalty
    
    def _reward_action_smoothness(self):
        """Penalize large or sudden action changes for motor safety"""
        env = self.env
        # Action rate penalty (difference between current and last actions)
        action_rate = torch.norm(env.actions - env.last_actions, dim=1)
        # Also penalize absolute action magnitude
        action_magnitude = torch.norm(env.actions, dim=1)
        # Combined penalty (weighted sum)
        smoothness_penalty = -0.1 * action_rate - 0.05 * action_magnitude
        return 0.1 * smoothness_penalty
    
    def _reward_torque_regularization(self):
        """Penalize large torques to prevent motor burnout"""
        env = self.env
        # Torques tensor shape: (num_envs, num_dof)
        torque_norms = torch.norm(env.torques, dim=1)
        # Quadratic penalty (scaled down since torques can be large)
        torque_penalty = -torque_norms ** 2
        # Very small scaling factor as suggested
        return 0.00001 * torque_penalty
    
    def _reward_joint_velocity_penalty(self):
        """Penalize excessive joint velocities for safety"""
        env = self.env
        # Joint velocity norms
        joint_vel_norms = torch.norm(env.dof_vel, dim=1)
        # Quadratic penalty
        velocity_penalty = -joint_vel_norms ** 2
        return 0.01 * velocity_penalty
    
    def _reward_foot_contact(self):
        """Encourage maintaining contact with the ball"""
        env = self.env
        # Number of feet in contact (binary contact state)
        contacts = env.last_contacts
        # Sum of contacts across all feet (higher is better)
        contact_count = contacts.sum(dim=1).float()
        # Reward proportional to number of contacts
        contact_reward = contact_count / env.feet_indices.shape[0]
        # Additional penalty for feet air time
        air_time_penalty = -torch.sum(env.feet_air_time, dim=1) * 0.1
        contact_reward += air_time_penalty
        return 0.3 * contact_reward

    # Success criteria as episode length
    def compute_success(self):
        return torch.ones_like(self.env.base_pos[:, 2])
