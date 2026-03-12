import os
import math
from dataclasses import MISSING

import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg, AssetBaseCfg
from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.managers import CurriculumTermCfg as CurrTerm
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors import ContactSensorCfg, RayCasterCfg, patterns
from isaaclab.terrains import TerrainImporterCfg
from isaaclab.utils import configclass
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR, ISAACLAB_NUCLEUS_DIR
from isaaclab.utils.noise import AdditiveUniformNoiseCfg as Unoise

import legged_lab.tasks.locomotion.amp.mdp as mdp
from legged_lab.tasks.locomotion.amp.amp_env_cfg import LocomotionAmpEnvCfg
from legged_lab import LEGGED_LAB_ROOT_DIR

##
# Pre-defined configs
##
from legged_lab.assets.qingyun_z1_A import qingyun_z1_A_CFG
# The order must align with the retarget config file scripts/tools/retarget/config/g1_29dof.yaml
KEY_BODY_NAMES = [
    "lrp_foot_roll",
    "rp_foot_roll",
    
    "lp_forearm_yaw",
    "rp_forearm_yaw",
    
    "lp_arm_roll",
    "rp_arm_roll",
] # if changed here and symmetry is enabled, remember to update amp.mdp.symmetry.g1 as well!
ANIMATION_TERM_NAME = "animation"
AMP_NUM_STEPS = 4

@configclass
class qingyun_z1_A_AmpRewards():
    """Reward terms for the MDP."""
    # -- task
    track_lin_vel_xy_exp = RewTerm(
        func=mdp.track_lin_vel_xy_exp, weight=1.0, params={"command_name": "base_velocity", "std": math.sqrt(0.25)}
    )
    track_ang_vel_z_exp = RewTerm(
        func=mdp.track_ang_vel_z_exp, weight=1.0, params={"command_name": "base_velocity", "std": math.sqrt(0.25)}
    )

    # -- penalties
    flat_orientation_l2 = RewTerm(func=mdp.flat_orientation_l2, weight=-1.0)
    # lin_vel_z_l2 = RewTerm(func=mdp.lin_vel_z_l2, weight=-0.3)
    # lin_vel_z_l2 = RewTerm(func=mdp.lin_vel_z_l2, weight=-0.4)
    # lin_vel_z_l2 = RewTerm(func=mdp.lin_vel_z_l2, weight=-0.7)
    ang_vel_xy_l2 = RewTerm(func=mdp.ang_vel_xy_l2, weight=-0.05)
    dof_torques_l2 = RewTerm(func=mdp.joint_torques_l2, weight=-2.0e-6)
    dof_acc_l2 = RewTerm(func=mdp.joint_acc_l2, weight=-1.0e-7)
    action_rate_l2 = RewTerm(func=mdp.action_rate_l2, weight=-0.005)
    dof_pos_limits = RewTerm(
        func=mdp.joint_pos_limits,
        weight=-1.0,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=[".*_foot_pitch", ".*_foot_roll"])},
    )
    
#=========================================================================================#
# 原生
    # joint_deviation_hip = RewTerm(
    #     func=mdp.joint_deviation_l1,
    #     weight=-0.1,
    #     params={"asset_cfg": SceneEntityCfg("robot", joint_names=[".*_hip_yaw_joint", ".*_hip_roll_joint"])},
    # )
    # joint_deviation_arms = RewTerm(
    #     func=mdp.joint_deviation_l1,
    #     weight=-0.05,
    #     params={
    #         "asset_cfg": SceneEntityCfg(
    #             "robot",
    #             joint_names=[
    #                 ".*_shoulder_.*_joint",
    #                 ".*_elbow_joint",
    #                 ".*_wrist_.*_joint",
    #             ],
    #         )
    #     },
    # )
    # joint_deviation_waist = RewTerm(
    #     func=mdp.joint_deviation_l1,
    #     weight=-0.1,
    #     params={"asset_cfg": SceneEntityCfg("robot", joint_names="waist_.*_joint")},
    # )
#=========================================================================================#
 #修改
    # joint_deviation_hip = RewTerm(
    #     func=mdp.joint_deviation_l1,
    #     # weight=-0.05,
    #     weight=-0.01,
    #     params={
    #         # "command_name": "base_velocity",
    #         "asset_cfg": SceneEntityCfg("robot", joint_names=[".*_hip_roll", ".*_hip_yaw"])},
    # )
    joint_deviation_arms = RewTerm(
        func=mdp.joint_deviation_l1,
        weight=-0.05,
        params={
            # "command_name": "base_velocity",
            "asset_cfg": SceneEntityCfg(
                "robot",
                joint_names=[
                    ".*_shoulder_pitch",
                    ".*_arm_roll",
                    ".*_forearm_yaw",
                ],
            )
        },
    )
    joint_deviation_waist = RewTerm(
        func=mdp.joint_deviation_l1,
        weight=-0.1,
        params={
            # "command_name": "base_velocity",
            "asset_cfg": SceneEntityCfg("robot", joint_names="w_waist_yaw")},
    )

# # 原地不动
#     joint_stationary_waist = RewTerm(
#         func=mdp.joint_deviation,
#         weight=-0.3,
#         params={
#             "command_name": "base_velocity",
#             "asset_cfg": SceneEntityCfg("robot", joint_names="loin_yaw_joint")},
#     )
#     joint_stationary_legs = RewTerm(
#         func=mdp.joint_deviation,
#         weight=-0.02,
#         params={
#             "command_name": "base_velocity",
#             "asset_cfg": SceneEntityCfg(
#                 "robot",
#                 joint_names=[
#                     "leg_.*1_joint",
#                     "leg_.*2_joint",
#                     "leg_.*3_joint",
#                     "leg_.*4_joint",
#                     "leg_.*5_joint",
#                 ],
#             )
#         },
#     )

#=========================================================================================#
    # velocity_direction_penalty = RewTerm(
    #     func=mdp.velocity_direction_penalty,
    #     weight=-0.4,
    #     params={
    #         "command_name": "base_velocity"
    #     },
    # )

    # idle_penalty = RewTerm(
    #     func=mdp.idle_when_commanded,
    #     weight=-2.0,
    #     params={"cmd_threshold": 0.2, "vel_threshold": 0.1},
    # )

    lin_vel_magnitude_l2 = RewTerm(
        func=mdp.lin_vel_magnitude_l2,
        weight=-1.0,
        params={"command_name": "base_velocity"},
    )

    # feet_height_diff = RewTerm(
    #     func=mdp.feet_height_diff,
    #     weight=2.0,
    #     params={
    #         "command_name": "base_velocity",
    #         "asset_cfg": SceneEntityCfg("robot", body_names=".*_foot_roll"),
    #         "cmd_threshold": 0.1,
    #     },
    # )

    feet_air_time = RewTerm(
        func=mdp.feet_air_time_positive_biped,
        # weight=0.5,
        weight=1.5,
        params={
            "command_name": "base_velocity",
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*_foot_roll"),
            "threshold": 0.4,
        },
    )

    feet_slide = RewTerm(
        func=mdp.feet_slide,
        weight=-0.1,
        params={
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*_foot_roll"),
            "asset_cfg": SceneEntityCfg("robot", body_names=".*_foot_roll"),
        },
    )
    
    termination_penalty = RewTerm(func=mdp.is_terminated, weight=-200.0)

@configclass
class qingyun_z1_A_AmpEnvCfg(LocomotionAmpEnvCfg):
    """Configuration for the qingyun_z1 AMP environment."""
    
    rewards: qingyun_z1_A_AmpRewards = qingyun_z1_A_AmpRewards()
    
    def __post_init__(self):
        super().__post_init__()
        
        self.scene.robot = qingyun_z1_A_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
        
        # self.scene.contact_forces = ContactSensorCfg(prim_path="{ENV_REGEX_NS}/Robot/.*", history_length=3, track_air_time=True)

        # ------------------------------------------------------
        # motion data
        # ------------------------------------------------------
        self.motion_data.motion_dataset.motion_data_dir = os.path.join(
            LEGGED_LAB_ROOT_DIR, "data", "MotionData", "qingyun_z1_A", "amp", "walk_and_run"
        )
        self.motion_data.motion_dataset.motion_data_weights = {
            "1_walk1_subject1": 1.0,
            "1_walk1_subject2": 1.0,
            "1_walk1_subject3": 1.0,
        }

        # ------------------------------------------------------
        # animation
        # ------------------------------------------------------
        self.animation.animation.num_steps_to_use = AMP_NUM_STEPS

        # -----------------------------------------------------
        # Observations
        # -----------------------------------------------------
        
        # policy observations

        self.observations.policy.base_ang_vel.func = mdp.base_ang_vel_custom
        self.observations.policy.base_ang_vel.params = {
            "body_cfg": SceneEntityCfg("robot", body_names="p_waist_yaw")
        }
        self.observations.policy.root_local_rot_tan_norm.func = mdp.root_local_rot_tan_norm_custom
        self.observations.policy.root_local_rot_tan_norm.params = {
            "body_cfg": SceneEntityCfg("robot", body_names="p_waist_yaw")
        }
        self.observations.policy.key_body_pos_b.params = {
            "asset_cfg": SceneEntityCfg(
                name="robot",
                body_names=KEY_BODY_NAMES,
                preserve_order=True
            )
        }

        # critic observations

        self.observations.critic.base_ang_vel.func = mdp.base_ang_vel_custom
        self.observations.critic.base_ang_vel.params = {
            "body_cfg": SceneEntityCfg("robot", body_names="p_waist_yaw")
        }
        self.observations.critic.root_local_rot_tan_norm.func = mdp.root_local_rot_tan_norm_custom
        self.observations.critic.root_local_rot_tan_norm.params = {
            "body_cfg": SceneEntityCfg("robot", body_names="p_waist_yaw")
        }
        self.observations.critic.key_body_pos_b.params = {
            "asset_cfg": SceneEntityCfg(
                name="robot",
                body_names=KEY_BODY_NAMES,
                preserve_order=True
            )
        }

        # discriminator observations

        self.observations.disc.base_ang_vel.func = mdp.base_ang_vel_custom
        self.observations.disc.base_ang_vel.params = {
            "body_cfg": SceneEntityCfg("robot", body_names="p_waist_yaw")
        }
        self.observations.disc.root_local_rot_tan_norm.func = mdp.root_local_rot_tan_norm_custom
        self.observations.disc.root_local_rot_tan_norm.params = {
            "body_cfg": SceneEntityCfg("robot", body_names="p_waist_yaw")
        }
        self.observations.disc.key_body_pos_b.params = {
            "asset_cfg": SceneEntityCfg(
                name="robot",
                body_names=KEY_BODY_NAMES,
                preserve_order=True
            )
        }
        self.observations.disc.history_length = AMP_NUM_STEPS
        
        # discriminator demostration observations
        
        self.observations.disc_demo.ref_root_local_rot_tan_norm.params["animation"] = ANIMATION_TERM_NAME
        self.observations.disc_demo.ref_root_ang_vel_b.params["animation"] = ANIMATION_TERM_NAME
        self.observations.disc_demo.ref_joint_pos.params["animation"] = ANIMATION_TERM_NAME
        self.observations.disc_demo.ref_joint_vel.params["animation"] = ANIMATION_TERM_NAME
        self.observations.disc_demo.ref_key_body_pos_b.params["animation"] = ANIMATION_TERM_NAME

        # ------------------------------------------------------
        # Events
        # ------------------------------------------------------
        self.events.add_base_mass.params["asset_cfg"].body_names = "base_link"
        self.events.base_external_force_torque.params["asset_cfg"].body_names = ["base_link"]
        self.events.reset_from_ref.params = {
            "animation": ANIMATION_TERM_NAME,
            "height_offset": 0.1
        }
        
        # ------------------------------------------------------
        # Rewards
        # ------------------------------------------------------
        self.rewards.flat_orientation_l2.func = mdp.flat_orientation_l2_custom
        self.rewards.flat_orientation_l2.params = {
            "body_cfg": SceneEntityCfg("robot", body_names="p_waist_yaw")
        }

        # ------------------------------------------------------
        # Commands
        # ------------------------------------------------------
        self.commands.base_velocity.ranges.lin_vel_x = (-2.0, 3.0)
        self.commands.base_velocity.ranges.lin_vel_y = (-1.0, 1.0)
        self.commands.base_velocity.ranges.ang_vel_z = (-1.0, 1.0)
        self.commands.base_velocity.ranges.heading = (-math.pi, math.pi)
        
        # ------------------------------------------------------
        # Curriculum
        # ------------------------------------------------------
        self.curriculum.lin_vel_cmd_levels = None
        self.curriculum.ang_vel_cmd_levels = None
        
        # ------------------------------------------------------
        # terminations
        # ------------------------------------------------------
        self.terminations.base_contact = None
        self.terminations.bad_orientation.func = mdp.bad_orientation_custom
        self.terminations.bad_orientation.params = {
            "limit_angle": math.radians(60.0),
            "body_cfg": SceneEntityCfg("robot", body_names="p_waist_yaw"),
        }


@configclass
class qingyun_z1_A_AmpEnvCfg_PLAY(qingyun_z1_A_AmpEnvCfg):
    
    def __post_init__(self):
        super().__post_init__()
        
        self.scene.num_envs = 48 
        self.scene.env_spacing = 2.5
        
        self.commands.base_velocity.ranges.lin_vel_x = (-2.0, 3.0)
        self.commands.base_velocity.ranges.lin_vel_y = (-1.0, 1.0)
        self.commands.base_velocity.ranges.ang_vel_z = (-1.0, 1.0)
        self.commands.base_velocity.ranges.heading = (0.0, 0.0)
        
        self.events.reset_from_ref = None

        
