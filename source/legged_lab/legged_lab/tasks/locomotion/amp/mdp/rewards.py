from __future__ import annotations

import torch
from typing import TYPE_CHECKING

from isaaclab.envs import mdp
from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors import ContactSensor
from isaaclab.assets import RigidObject
import isaaclab.utils.math as math_utils

from isaaclab.assets import Articulation

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv
    
def feet_orientation_l2(env: ManagerBasedRLEnv, 
                          sensor_cfg: SceneEntityCfg, 
                          asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """Penalize feet orientation not parallel to the ground when in contact.

    This is computed by penalizing the xy-components of the projected gravity vector.
    """
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    asset:RigidObject = env.scene[asset_cfg.name]
    
    in_contact = contact_sensor.data.net_forces_w_history[:, :, sensor_cfg.body_ids, :].norm(dim=-1).max(dim=1)[0] > 1.0
    # shape: (N, M)
    
    num_feet = len(sensor_cfg.body_ids)
    
    feet_quat = asset.data.body_quat_w[:, sensor_cfg.body_ids, :]   # shape: (N, M, 4)
    feet_proj_g = math_utils.quat_apply_inverse(
        feet_quat, 
        asset.data.GRAVITY_VEC_W.unsqueeze(1).expand(-1, num_feet, -1)  # shape: (N, M, 3)
    )
    feet_proj_g_xy_square = torch.sum(torch.square(feet_proj_g[:, :, :2]), dim=-1)  # shape: (N, M)
    
    return torch.sum(feet_proj_g_xy_square * in_contact, dim=-1)  # shape: (N, )
    
def stand_still_joint_deviation_l1(
    env: ManagerBasedRLEnv, command_name: str, command_threshold: float = 0.06, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    """Penalize offsets from the default joint positions when the command is very small."""
    command = env.command_manager.get_command(command_name)
    # Penalize motion when command is nearly zero.
    return mdp.joint_deviation_l1(env, asset_cfg) * (torch.norm(command[:, :2], dim=1) < command_threshold)

def joint_deviation(env: ManagerBasedRLEnv, command_name: str, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    asset: Articulation = env.scene[asset_cfg.name]
    angle = asset.data.joint_pos[:, asset_cfg.joint_ids] - asset.data.default_joint_pos[:, asset_cfg.joint_ids]
    zero_flag = (
        torch.norm(env.command_manager.get_command(command_name)[:, :2], dim=1) + torch.abs(env.command_manager.get_command(command_name)[:, 2])
    ) < 0.1
    neg_x_flag = (env.command_manager.get_command(command_name)[:, 0] >= 0)
    return torch.sum(torch.abs(angle), dim=1) * zero_flag * neg_x_flag

def flat_orientation_l2_custom(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    body_cfg: SceneEntityCfg = SceneEntityCfg("robot", body_names="p_waist_yaw"),
) -> torch.Tensor:
    """Penalize non-flat orientation of a specified body using L2 squared kernel.

    Useful when the IMU is mounted on a body other than base_link (e.g. p_waist_yaw).
    """
    robot: Articulation = env.scene[asset_cfg.name]

    body_quat = robot.data.body_quat_w[:, body_cfg.body_ids[0], :]  # (N, 4)
    gravity_w = robot.data.GRAVITY_VEC_W  # (N, 3)
    proj_g = math_utils.quat_apply_inverse(body_quat, gravity_w)  # (N, 3)

    return torch.sum(torch.square(proj_g[:, :2]), dim=1)


def bad_orientation_custom(
    env: ManagerBasedRLEnv,
    limit_angle: float,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    body_cfg: SceneEntityCfg = SceneEntityCfg("robot", body_names="p_waist_yaw"),
) -> torch.Tensor:
    """Terminate when a specified body's orientation is too far from upright.

    Useful when the IMU is mounted on a body other than base_link (e.g. p_waist_yaw).
    """
    robot: Articulation = env.scene[asset_cfg.name]

    body_quat = robot.data.body_quat_w[:, body_cfg.body_ids[0], :]  # (N, 4)
    gravity_w = robot.data.GRAVITY_VEC_W  # (N, 3)
    proj_g = math_utils.quat_apply_inverse(body_quat, gravity_w)  # (N, 3)

    return torch.acos(-proj_g[:, 2]).abs() > limit_angle


def velocity_direction_penalty(env: ManagerBasedRLEnv, command_name: str, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """Penalize deviation between actual velocity direction and commanded velocity direction.

    Only considers direction, not magnitude. The larger the angular difference, the higher the penalty.
    """
    asset: Articulation = env.scene[asset_cfg.name]
    command = env.command_manager.get_command(command_name)

    # Get actual base velocity in world frame (xy plane)
    actual_vel = asset.data.root_lin_vel_w[:, :2]  # shape: (N, 2)
    # Get commanded velocity (xy plane)
    command_vel = command[:, :2]  # shape: (N, 2)

    # Normalize velocities to get direction vectors
    actual_vel_norm = torch.nn.functional.normalize(actual_vel, dim=1, eps=1e-6)
    command_vel_norm = torch.nn.functional.normalize(command_vel, dim=1, eps=1e-6)

    # Compute cosine similarity (dot product of normalized vectors)
    # cos_sim = 1 means same direction, -1 means opposite direction
    cos_sim = torch.sum(actual_vel_norm * command_vel_norm, dim=1)

    # Convert to penalty: (1 - cos_sim) ranges from 0 (same direction) to 2 (opposite direction)
    # Only penalize when command velocity is non-negligible
    command_magnitude = torch.norm(command_vel, dim=1)
    # penalty = (1.0 - cos_sim) * (command_magnitude > 0.1)
    penalty = 1.0 - cos_sim  # 始终惩罚

    return penalty 

def feet_slide(env, sensor_cfg: SceneEntityCfg, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """Penalize feet sliding.

    This function penalizes the agent for sliding its feet on the ground. The reward is computed as the
    norm of the linear velocity of the feet multiplied by a binary contact sensor. This ensures that the
    agent is penalized only when the feet are in contact with the ground.
    """
    # Penalize feet sliding
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    contacts = contact_sensor.data.net_forces_w_history[:, :, sensor_cfg.body_ids, :].norm(dim=-1).max(dim=1)[0] > 1.0
    asset = env.scene[asset_cfg.name]

    body_vel = asset.data.body_lin_vel_w[:, asset_cfg.body_ids, :2]
    reward = torch.sum(body_vel.norm(dim=-1) * contacts, dim=1)
    return reward

def feet_air_time_positive_biped(env, command_name: str, threshold: float, sensor_cfg: SceneEntityCfg) -> torch.Tensor:
    """Reward long steps taken by the feet for bipeds.

    This function rewards the agent for taking steps up to a specified threshold and also keep one foot at
    a time in the air.

    If the commands are small (i.e. the agent is not supposed to take a step), then the reward is zero.
    """
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    # compute the reward
    air_time = contact_sensor.data.current_air_time[:, sensor_cfg.body_ids]
    contact_time = contact_sensor.data.current_contact_time[:, sensor_cfg.body_ids]
    in_contact = contact_time > 0.0
    in_mode_time = torch.where(in_contact, contact_time, air_time)
    single_stance = torch.sum(in_contact.int(), dim=1) == 1
    if env.common_step_counter % 200 == 0:
        print(f"[feet] in_contact(env0): left={in_contact[0,0].item()}, right={in_contact[0,1].item()} | single_stance(env0): {single_stance[0].item()}")
    reward = torch.min(torch.where(single_stance.unsqueeze(-1), in_mode_time, 0.0), dim=1)[0]
    reward = torch.clamp(reward, max=threshold)
    # no reward for zero command
    reward *= torch.norm(env.command_manager.get_command(command_name)[:, :2], dim=1) > 0.1
    return reward

def both_feet_contact(
    env,
    sensor_cfg: SceneEntityCfg,
    force_threshold: float = 1.0,
) -> torch.Tensor:
    """Reward both feet being in contact with the ground.

    Returns 1.0 when both feet have contact force above threshold, 0.0 otherwise.
    """
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    forces = contact_sensor.data.net_forces_w_history[:, :, sensor_cfg.body_ids, :].norm(dim=-1).max(dim=1)[0]
    in_contact = forces > force_threshold  # (num_envs, num_feet)
    both_contact = torch.all(in_contact, dim=1)  # both feet touching
    return both_contact.float()


def idle_when_commanded(
    env: ManagerBasedRLEnv,
    cmd_threshold: float = 0.2,
    vel_threshold: float = 0.1,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Penalize being idle when a velocity command is given.
    
    This reward function detects "lazy standing" behavior where the robot receives
    a movement command but remains stationary. It returns 1.0 when the robot should
    be moving but is not, enabling a negative weight penalty.
    
    Args:
        env: Environment instance.
        cmd_threshold: Minimum command magnitude to be considered "commanded to move".
            Commands below this threshold are ignored (robot is allowed to stand).
        vel_threshold: Maximum velocity magnitude to be considered "idle/stationary".
            If actual velocity is below this, the robot is considered not moving.
        asset_cfg: Robot configuration.
    
    Returns:
        Tensor of shape (num_envs,) with values:
        - 1.0 if commanded to move but idle (should be penalized)
        - 0.0 otherwise (no penalty)
    
    Example:
        idle_penalty = RewTerm(
            func=mdp.idle_when_commanded,
            weight=-2.0,
            params={"cmd_threshold": 0.2, "vel_threshold": 0.1}
        )
    """
    asset: Articulation = env.scene[asset_cfg.name]
    
    # Get velocity command (xy components)
    cmd_xy = env.command_manager.get_command("base_velocity")[:, :2]
    cmd_magnitude = torch.linalg.norm(cmd_xy, dim=-1)
    
    # Get actual root velocity in yaw frame (same as track_lin_vel_xy uses)
    vel_yaw = math_utils.quat_apply_inverse(
        math_utils.yaw_quat(asset.data.root_quat_w), asset.data.root_lin_vel_w[:, :3]
    )
    vel_magnitude = torch.linalg.norm(vel_yaw[:, :2], dim=-1)
    
    # Detect "commanded but idle" condition
    is_commanded = cmd_magnitude > cmd_threshold  # Should be moving
    is_idle = vel_magnitude < vel_threshold       # But not moving
    
    return (is_commanded & is_idle).float()
