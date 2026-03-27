from __future__ import annotations

import torch
from typing import TYPE_CHECKING

import isaaclab.utils.math as math_utils
import isaaclab.utils.string as string_utils
from isaaclab.assets import Articulation, RigidObject
from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors import FrameTransformer

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv
    from legged_lab.envs import ManagerBasedAnimationEnv
    from legged_lab.managers import AnimationTerm
    


def root_local_rot_tan_norm(
    env: ManagerBasedEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    robot: Articulation = env.scene[asset_cfg.name]
    
    root_quat = robot.data.root_quat_w
    yaw_quat = math_utils.yaw_quat(root_quat)
    
    root_quat_local = math_utils.quat_mul(math_utils.quat_conjugate(yaw_quat), root_quat)
    
    root_rotm_local = math_utils.matrix_from_quat(root_quat_local)
    # use the first and last column of the rotation matrix as the tangent and normal vectors
    tan_vec = root_rotm_local[:, :, 0]  # (N, 3)
    norm_vec = root_rotm_local[:, :, 2]  # (N, 3)
    obs = torch.cat([tan_vec, norm_vec], dim=-1)  # (N, 6)

    return obs


def root_local_rot_tan_norm_custom(
    env: ManagerBasedEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    body_cfg: SceneEntityCfg = SceneEntityCfg("robot", body_names="p_waist_yaw"),
) -> torch.Tensor:
    """Same as root_local_rot_tan_norm but uses a specified body's quaternion instead of root.

    Useful when the IMU is mounted on a body other than base_link (e.g. p_waist_yaw).
    """
    robot: Articulation = env.scene[asset_cfg.name]

    body_quat = robot.data.body_quat_w[:, body_cfg.body_ids[0], :]  # (N, 4)
    yaw_quat = math_utils.yaw_quat(body_quat)

    body_quat_local = math_utils.quat_mul(math_utils.quat_conjugate(yaw_quat), body_quat)

    body_rotm_local = math_utils.matrix_from_quat(body_quat_local)
    tan_vec = body_rotm_local[:, :, 0]  # (N, 3)
    norm_vec = body_rotm_local[:, :, 2]  # (N, 3)
    obs = torch.cat([tan_vec, norm_vec], dim=-1)  # (N, 6)

    return obs


def base_ang_vel_custom(
    env: ManagerBasedEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    body_cfg: SceneEntityCfg = SceneEntityCfg("robot", body_names="p_waist_yaw"),
) -> torch.Tensor:
    """Angular velocity of a specified body in that body's frame.

    Useful when the IMU is mounted on a body other than base_link (e.g. p_waist_yaw).
    """
    robot: Articulation = env.scene[asset_cfg.name]

    # Get body angular velocity in world frame
    body_ang_vel_w = robot.data.body_ang_vel_w[:, body_cfg.body_ids[0], :]  # (N, 3)

    # Transform to body frame using body quaternion
    body_quat = robot.data.body_quat_w[:, body_cfg.body_ids[0], :]  # (N, 4)
    body_ang_vel_b = math_utils.quat_apply_inverse(body_quat, body_ang_vel_w)

    return body_ang_vel_b


def base_lin_vel_custom(
    env: ManagerBasedEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    body_cfg: SceneEntityCfg = SceneEntityCfg("robot", body_names="p_waist_yaw"),
) -> torch.Tensor:
    """Linear velocity of a specified body in that body's frame."""
    robot: Articulation = env.scene[asset_cfg.name]

    body_lin_vel_w = robot.data.body_lin_vel_w[:, body_cfg.body_ids[0], :]
    body_quat = robot.data.body_quat_w[:, body_cfg.body_ids[0], :]
    body_lin_vel_b = math_utils.quat_apply_inverse(body_quat, body_lin_vel_w)

    return body_lin_vel_b


def ref_root_local_rot_tan_norm(
    env: ManagerBasedAnimationEnv,
    animation: str,
    flatten_steps_dim: bool = True,
) -> torch.Tensor:

    animation_term: AnimationTerm = env.animation_manager.get_term(animation)
    num_envs = env.num_envs

    ref_root_quat = animation_term.get_root_quat() # shape: (num_envs, num_steps, 4)
    ref_yaw_quat = math_utils.yaw_quat(ref_root_quat)
    ref_root_quat_local = math_utils.quat_mul(
        math_utils.quat_conjugate(ref_yaw_quat), ref_root_quat
    )  # shape: (num_envs, num_steps, 4)
    ref_root_rotm_local = math_utils.matrix_from_quat(ref_root_quat_local) # shape: (num_envs, num_steps, 3, 3)

    tan_vec = ref_root_rotm_local[:, :, :, 0]  # (num_envs, num_steps, 3)
    norm_vec = ref_root_rotm_local[:, :, :, 2]  # (num_envs, num_steps, 3)
    obs = torch.cat([tan_vec, norm_vec], dim=-1)  # (num_envs, num_steps, 6)

    if flatten_steps_dim:
        return obs.reshape(num_envs, -1)
    else:
        return obs
