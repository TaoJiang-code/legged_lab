from __future__ import annotations

from typing import TYPE_CHECKING

import torch

import isaaclab.utils.math as math_utils
from isaaclab.envs.mdp.commands.commands_cfg import UniformVelocityCommandCfg as IsaacUniformVelocityCommandCfg
from isaaclab.envs.mdp.commands.velocity_command import UniformVelocityCommand as IsaacUniformVelocityCommand
from isaaclab.utils import configclass

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv


class UniformVelocityCommand(IsaacUniformVelocityCommand):
    """Velocity command expressed in a selected body frame instead of the articulation root."""

    cfg: "UniformVelocityCommandCfg"

    def __init__(self, cfg: "UniformVelocityCommandCfg", env: "ManagerBasedEnv"):
        super().__init__(cfg, env)

        self.body_idx, body_names = self.robot.find_bodies(cfg.body_name)
        if len(self.body_idx) != 1:
            raise ValueError(
                f"Expected one body match for velocity command body '{cfg.body_name}', found {len(self.body_idx)}: {body_names}."
            )
        self.body_idx = self.body_idx[0]
        self.body_name = body_names[0]

    def _get_body_heading_w(self) -> torch.Tensor:
        body_quat_w = self.robot.data.body_quat_w[:, self.body_idx]
        forward_w = math_utils.quat_apply(body_quat_w, self.robot.data.FORWARD_VEC_B)
        return torch.atan2(forward_w[:, 1], forward_w[:, 0])

    def _get_body_lin_vel_b(self) -> torch.Tensor:
        body_quat_w = self.robot.data.body_quat_w[:, self.body_idx]
        body_lin_vel_w = self.robot.data.body_lin_vel_w[:, self.body_idx]
        return math_utils.quat_apply_inverse(body_quat_w, body_lin_vel_w)

    def _get_body_ang_vel_b(self) -> torch.Tensor:
        body_quat_w = self.robot.data.body_quat_w[:, self.body_idx]
        body_ang_vel_w = self.robot.data.body_ang_vel_w[:, self.body_idx]
        return math_utils.quat_apply_inverse(body_quat_w, body_ang_vel_w)

    def _update_metrics(self):
        max_command_time = self.cfg.resampling_time_range[1]
        max_command_step = max_command_time / self._env.step_dt

        body_lin_vel_b = self._get_body_lin_vel_b()
        body_ang_vel_b = self._get_body_ang_vel_b()

        self.metrics["error_vel_xy"] += (
            torch.norm(self.vel_command_b[:, :2] - body_lin_vel_b[:, :2], dim=-1) / max_command_step
        )
        self.metrics["error_vel_yaw"] += torch.abs(self.vel_command_b[:, 2] - body_ang_vel_b[:, 2]) / max_command_step

    def _update_command(self):
        if self.cfg.heading_command:
            env_ids = self.is_heading_env.nonzero(as_tuple=False).flatten()
            heading_error = math_utils.wrap_to_pi(self.heading_target[env_ids] - self._get_body_heading_w()[env_ids])
            self.vel_command_b[env_ids, 2] = torch.clip(
                self.cfg.heading_control_stiffness * heading_error,
                min=self.cfg.ranges.ang_vel_z[0],
                max=self.cfg.ranges.ang_vel_z[1],
            )

        standing_env_ids = self.is_standing_env.nonzero(as_tuple=False).flatten()
        self.vel_command_b[standing_env_ids, :] = 0.0

    def _debug_vis_callback(self, event):
        if not self.robot.is_initialized:
            return

        body_pos_w = self.robot.data.body_pos_w[:, self.body_idx].clone()
        body_pos_w[:, 2] += 0.5

        vel_des_arrow_scale, vel_des_arrow_quat = self._resolve_xy_velocity_to_arrow(self.command[:, :2])
        vel_arrow_scale, vel_arrow_quat = self._resolve_xy_velocity_to_arrow(self._get_body_lin_vel_b()[:, :2])

        self.goal_vel_visualizer.visualize(body_pos_w, vel_des_arrow_quat, vel_des_arrow_scale)
        self.current_vel_visualizer.visualize(body_pos_w, vel_arrow_quat, vel_arrow_scale)

    def _resolve_xy_velocity_to_arrow(self, xy_velocity: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        default_scale = self.goal_vel_visualizer.cfg.markers["arrow"].scale
        arrow_scale = torch.tensor(default_scale, device=self.device).repeat(xy_velocity.shape[0], 1)
        arrow_scale[:, 0] *= torch.linalg.norm(xy_velocity, dim=1) * 3.0

        heading_angle = torch.atan2(xy_velocity[:, 1], xy_velocity[:, 0])
        zeros = torch.zeros_like(heading_angle)
        arrow_quat = math_utils.quat_from_euler_xyz(zeros, zeros, heading_angle)
        body_quat_w = self.robot.data.body_quat_w[:, self.body_idx]
        arrow_quat = math_utils.quat_mul(body_quat_w, arrow_quat)

        return arrow_scale, arrow_quat


@configclass
class UniformVelocityCommandCfg(IsaacUniformVelocityCommandCfg):
    """Configuration for a velocity command expressed in a selected body frame."""

    class_type: type = UniformVelocityCommand
    body_name: str = "base_link"
