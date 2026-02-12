
import isaaclab.sim as sim_utils
from isaaclab.actuators import ActuatorNetMLPCfg, DCMotorCfg, ImplicitActuatorCfg
from isaaclab.assets.articulation import ArticulationCfg
from isaaclab.utils import configclass

from legged_lab import LEGGED_LAB_ROOT_DIR
from legged_lab.assets import unitree_actuators


@configclass
class qingyun_z1_ArticulationCfg(ArticulationCfg):

    joint_sdk_names: list[str] = None

    soft_joint_pos_limit_factor = 0.9

qingyun_z1_CFG = qingyun_z1_ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=f"{LEGGED_LAB_ROOT_DIR}/data/Robots/qingyun_z1/usd/qingyun_z1.usd",
        activate_contact_sensors=True,
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=False,
            retain_accelerations=False,
            linear_damping=0.0,
            angular_damping=0.0,
            max_linear_velocity=1000.0,
            max_angular_velocity=1000.0,
            max_depenetration_velocity=1.0,
        ),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=True, solver_position_iteration_count=8, solver_velocity_iteration_count=4
        ),
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.693),
        joint_pos={
            "loin_yaw_joint": 0.0,

            "leg_l1_joint": 0.0,
            "leg_l2_joint": 0.0,
            "leg_l3_joint": 0.0,
            "leg_l4_joint": 0.0,
            "leg_l5_joint": 0.0,

            "leg_r1_joint": 0.0,
            "leg_r2_joint": 0.0,
            "leg_r3_joint": 0.0,
            "leg_r4_joint": 0.0,
            "leg_r5_joint": 0.0,

            "l_shoulder_pitch_joint": 0.0,
            "l_shoulder_roll_joint": 0.0,
            "l_shoulder_yaw_joint": 0.0,
            "l_arm_pitch_joint": 0.0,

            "r_shoulder_pitch_joint": 0.0,
            "r_shoulder_roll_joint": 0.0,
            "r_shoulder_yaw_joint": 0.0,
            "r_arm_pitch_joint": 0.0,
        },
        joint_vel={".*": 0.0},
    ),
    actuators={
        "arm": ImplicitActuatorCfg(
            joint_names_expr=[
                ".*_shoulder_pitch_joint", 
                ".*_shoulder_roll_joint", 
                ".*_shoulder_yaw_joint",
                ".*_arm_pitch_joint"
                ],
            effort_limit_sim=7.0,
            velocity_limit_sim=20.0,
            stiffness={
                ".*_shoulder_pitch_joint": 10.0,
                ".*_shoulder_roll_joint": 10.0,
                ".*_shoulder_yaw_joint": 10.0,
                ".*_arm_pitch_joint": 10.0,
            },
            damping={
                ".*_shoulder_pitch_joint": 0.7,
                ".*_shoulder_roll_joint": 0.7,
                ".*_shoulder_yaw_joint": 0.7,
                ".*_arm_pitch_joint": 0.7,
            },
            armature=0.01,
        ),
        "leg": ImplicitActuatorCfg(
            joint_names_expr=[
                "leg_.*1_joint", 
                "leg_.*2_joint",
                "leg_.*3_joint",
                "leg_.*4_joint",
                "leg_.*5_joint"
                ],
            effort_limit_sim=28.0,
            velocity_limit_sim=20.0,
            stiffness={
                "leg_.*1_joint": 30.0,
                "leg_.*2_joint": 30.0,
                "leg_.*3_joint": 30.0,
                "leg_.*4_joint": 30.0,
                "leg_.*5_joint": 30.0,
            },
            damping={
                "leg_.*1_joint": 3.0,
                "leg_.*2_joint": 3.0,
                "leg_.*3_joint": 3.0,
                "leg_.*4_joint": 3.0,
                "leg_.*5_joint": 3.0,
            },
            armature=0.01,
        ),
        "waist": ImplicitActuatorCfg(
            joint_names_expr=[
                "loin_yaw_joint"
            ],
            effort_limit_sim=10.0,
            velocity_limit_sim=20.0,
            stiffness={
                "loin_yaw_joint": 15.0,
            },
            damping={
                "loin_yaw_joint": 1.0,
            },
            armature=0.01,
        ),
    },
    joint_sdk_names=[
            "loin_yaw_joint",

            "leg_l1_joint",
            "leg_l2_joint",
            "leg_l3_joint",
            "leg_l4_joint",
            "leg_l5_joint",

            "leg_r1_joint",
            "leg_r2_joint",
            "leg_r3_joint",
            "leg_r4_joint",
            "leg_r5_joint",

            "l_shoulder_pitch_joint",
            "l_shoulder_roll_joint",
            "l_shoulder_yaw_joint",
            "l_arm_pitch_joint",

            "r_shoulder_pitch_joint",
            "r_shoulder_roll_joint",
            "r_shoulder_yaw_joint",
            "r_arm_pitch_joint",
    ],
)
