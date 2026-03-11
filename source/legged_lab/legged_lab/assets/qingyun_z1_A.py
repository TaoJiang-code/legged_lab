
import isaaclab.sim as sim_utils
from isaaclab.actuators import ActuatorNetMLPCfg, DCMotorCfg, ImplicitActuatorCfg
from isaaclab.assets.articulation import ArticulationCfg
from isaaclab.utils import configclass

from legged_lab import LEGGED_LAB_ROOT_DIR
from legged_lab.assets import unitree_actuators


@configclass
class qingyun_z1_A_ArticulationCfg(ArticulationCfg):

    joint_sdk_names: list[str] = None

    soft_joint_pos_limit_factor = 0.9

qingyun_z1_A_CFG = qingyun_z1_A_ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=f"{LEGGED_LAB_ROOT_DIR}/data/Robots/qingyun_z1_A/usd/qingyun_z1_A.usd",
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
        pos=(0.0, 0.0, 0.8),
        joint_pos={
            "w_waist_yaw": 0.0,

            "lrw_hip_pitch": 0.0,
            "lrw_hip_roll": 0.0,
            "lrw_hip_yaw": 0.0,
            "lrw_knee_pitch": 0.0,
            "lrw_foot_pitch": 0.0,
            "lrw_foot_roll": 0.0,

            "rw_hip_pitch": 0.0,
            "rw_hip_roll": 0.0,
            "rw_hip_yaw": 0.0,
            "rw_knee_pitch": 0.0,
            "rw_foot_pitch": 0.0,
            "rw_foot_roll": 0.0,

            "lw_shoulder_pitch": 0.0,
            "lw_arm_roll": 0.0,
            "lw_forearm_yaw": 0.0,

            "rw_shoulder_pitch": 0.0,
            "rw_arm_roll": 0.0,
            "rw_forearm_yaw": 0.0,
        },
        joint_vel={".*": 0.0},
    ),
    actuators={
        "arm": ImplicitActuatorCfg(
            joint_names_expr=[
                "lw_shoulder_pitch",
                "rw_shoulder_pitch",
                "lw_arm_roll",
                "rw_arm_roll",
                "lw_forearm_yaw",
                "rw_forearm_yaw",
            ],
            effort_limit_sim={
                "lw_shoulder_pitch": 7.0, 
                "rw_shoulder_pitch": 7.0, 
                "lw_arm_roll": 7.0, 
                "rw_arm_roll": 7.0, 
                "lw_forearm_yaw": 3.5, 
                "rw_forearm_yaw": 3.5,
            },
            velocity_limit_sim={
                "lw_shoulder_pitch": 40.0, 
                "rw_shoulder_pitch": 40.0, 
                "lw_arm_roll": 40.0, 
                "rw_arm_roll": 40.0, 
                "lw_forearm_yaw": 30.0, 
                "rw_forearm_yaw": 30.0,
            },
            stiffness=10.0,
            damping=0.7,
            armature=0.01,
        ),
        "leg": ImplicitActuatorCfg(
            joint_names_expr=[
                "lrw_hip_pitch",
                "lrw_hip_roll",
                "lrw_hip_yaw",
                "lrw_knee_pitch",
                "lrw_foot_pitch",
                "lrw_foot_roll",
                "rw_hip_pitch",
                "rw_hip_roll",
                "rw_hip_yaw",
                "rw_knee_pitch",
                "rw_foot_pitch",
                "rw_foot_roll",
            ],
            effort_limit_sim={
                "lrw_hip_pitch": 27.0,
                "lrw_hip_roll": 27.0,
                "lrw_hip_yaw": 27.0,
                "lrw_knee_pitch": 27.0,
                "lrw_foot_pitch": 27.0,
                "lrw_foot_roll": 27.0,
                "rw_hip_pitch": 27.0,
                "rw_hip_roll": 27.0,
                "rw_hip_yaw": 27.0,
                "rw_knee_pitch": 27.0,
                "rw_foot_pitch": 27.0,
                "rw_foot_roll": 27.0,
            },
            velocity_limit_sim={
                "lrw_hip_pitch": 5.45,
                "lrw_hip_roll": 5.45,
                "lrw_hip_yaw": 5.45,
                "lrw_knee_pitch": 5.45,
                "lrw_foot_pitch": 5.45,
                "lrw_foot_roll": 5.45,
                "rw_hip_pitch": 5.45,
                "rw_hip_roll": 5.45,
                "rw_hip_yaw": 5.45,
                "rw_knee_pitch": 5.45,
                "rw_foot_pitch": 5.45,
                "rw_foot_roll": 5.45,
            },
            stiffness=30.0,
            damping=3.0,
            armature=0.01,
        ),
        "waist": ImplicitActuatorCfg(
            joint_names_expr=["w_waist_yaw"],
            effort_limit_sim=17.0,
            velocity_limit_sim=28.0,
            stiffness=15.0,
            damping=1.0,
            armature=0.01,
        ),
    },
    joint_sdk_names=[
        "rw_shoulder_pitch",
        "rw_arm_roll",
        "rw_forearm_yaw",
        "lw_shoulder_pitch",
        "lw_arm_roll",
        "lw_forearm_yaw",

        "w_waist_yaw",

        "rw_hip_pitch",
        "rw_hip_roll",
        "rw_hip_yaw",
        "rw_knee_pitch",
        "rw_foot_pitch",
        "rw_foot_roll",

        "lrw_hip_pitch",
        "lrw_hip_roll",
        "lrw_hip_yaw",
        "lrw_knee_pitch",
        "lrw_foot_pitch",
        "lrw_foot_roll",
    ],
)
