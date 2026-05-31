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
        
        # Calculate desired height: 2 * ball_radius above ball
        ball_z = env.object_pos_world_frame[:, 2]
        desired_height = ball_z + 2 * env.ball_radius
        
        # Get current robot base height
        current_height = env.base_pos[:, 2]
        
        # Quadratic penalty for being below desired height
        height_diff = desired_height - current_height
        height_penalty = torch.clamp(height_diff, min=0.0) ** 2
        
        # Small reward for being near desired height
        height_reward = torch.exp(-((current_height - desired_height) ** 2) / (2 * 0.1 ** 2))
        
        # Combine: penalize being too low, reward being near target
        reward = height_reward - 0.5 * height_penalty
        
        return 1.0 * reward
    
    def _reward_foot_contact(self):
        env = self.env
        
        # Get contact forces on feet
        feet_contact_forces = env.contact_forces[:, env.feet_indices, :]
        feet_contact_mag = torch.norm(feet_contact_forces, dim=-1)
        
        # Binary contact detection (threshold for reliable contact)
        contact_threshold = 1.0  # Newtons
        feet_in_contact = (feet_contact_mag > contact_threshold).float()
        
        # Reward for having 3-4 feet in contact (optimal for stability)
        contact_count = torch.sum(feet_in_contact, dim=-1)
        optimal_contacts = torch.clamp(contact_count, 2, 4)
        
        # Penalty for no contact (falling) or 1 foot contact (unstable)
        contact_reward = torch.where(
            contact_count >= 2,
            1.0 - 0.1 * torch.abs(contact_count - 3),  # Peak at 3 contacts
            -1.0  # Penalty for unsafe contact situation
        )
        
        return 0.3 * contact_reward
    
    def _reward_orientation(self):
        env = self.env
        
        # Get base orientation using projected gravity
        projected_gravity = env.projected_gravity
        
        # Ideal gravity vector is straight down (0, 0, -1)
        # Penalize deviation from vertical orientation
        # projected_gravity[:, 0:2] should be close to 0 for upright orientation
        horizontal_gravity = projected_gravity[:, 0:2]
        orientation_penalty = torch.sum(horizontal_gravity ** 2, dim=-1)
        
        # Also penalize excessive roll and pitch
        # Use quaternion to get roll/pitch if needed
        # For simplicity, we use gravity projection
        
        return -0.5 * orientation_penalty
    
    def _reward_smooth_motion(self):
        env = self.env
        
        # Penalize jerky motion using base acceleration approximation
        # Use difference in base velocity (linear and angular)
        if hasattr(env, 'last_root_vel'):
            current_vel = env.root_states[env.robot_actor_idxs, 7:13]
            vel_diff = current_vel - env.last_root_vel
            acceleration_penalty = torch.sum(vel_diff ** 2, dim=-1)
        else:
            acceleration_penalty = torch.zeros(env.num_envs, device=env.device)
        
        # Also penalize large changes in joint positions
        if hasattr(env, 'last_dof_pos'):
            dof_pos_diff = env.dof_pos - env.last_dof_pos
            joint_jerk_penalty = torch.sum(dof_pos_diff ** 2, dim=-1)
        else:
            joint_jerk_penalty = torch.zeros(env.num_envs, device=env.device)
        
        smoothness_penalty = acceleration_penalty + 0.1 * joint_jerk_penalty
        
        return -0.05 * smoothness_penalty
    
    def _reward_joint_velocity(self):
        env = self.env
        
        # Penalize fast joint velocities (safety for motors)
        joint_vel = env.dof_vel
        joint_vel_penalty = torch.sum(joint_vel ** 2, dim=-1)
        
        return -0.01 * joint_vel_penalty
    
    def _reward_action_rate(self):
        env = self.env
        
        # Penalize rapid changes in actions
        action_diff = env.actions - env.last_actions
        action_rate_penalty = torch.sum(action_diff ** 2, dim=-1)
        
        return -0.1 * action_rate_penalty
    
    def _reward_torque(self):
        env = self.env
        
        # Penalize large torques (motor protection)
        # Scale appropriately since torques can be large
        torque_penalty = torch.sum(env.torques ** 2, dim=-1)
        
        return -0.00001 * torque_penalty
    
    def _reward_foot_slippage(self):
        env = self.env
        
        # Penalize foot sliding on ball surface
        # Approximate by penalizing horizontal foot velocities when in contact
        feet_velocities = env.foot_velocities  # Shape: (num_envs, num_feet, 3)
        
        # Get contact states
        feet_contact_forces = env.contact_forces[:, env.feet_indices, :]
        feet_contact_mag = torch.norm(feet_contact_forces, dim=-1)
        feet_in_contact = (feet_contact_mag > 1.0).float()
        
        # Horizontal velocity (x, y components)
        horizontal_vel = feet_velocities[:, :, 0:2]
        horizontal_speed = torch.norm(horizontal_vel, dim=-1)
        
        # Slippage penalty: only when in contact
        slippage_penalty = torch.sum(horizontal_speed * feet_in_contact, dim=-1)
        
        return -0.2 * slippage_penalty
    
    def _reward_ball_height_safety(self):
        env = self.env
        
        # Safety penalty for being too low (risk of falling off ball)
        ball_z = env.object_pos_world_frame[:, 2]
        current_height = env.base_pos[:, 2]
        
        # Critical threshold: 1.5 * ball_radius (safety margin)
        critical_height = ball_z + 1.5 * env.ball_radius
        
        # Exponential penalty for being below critical height
        height_diff = critical_height - current_height
        safety_penalty = torch.exp(torch.clamp(height_diff, min=0.0)) - 1.0
        
        return -0.3 * safety_penalty

    # Success criteria as episode length
    def compute_success(self):
        return torch.ones_like(self.env.base_pos[:, 2])
