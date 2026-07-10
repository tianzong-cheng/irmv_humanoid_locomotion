import math

import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets import ArticulationCfg

from irmv_humanoid_locomotion.assets import ASSET_DIR

# TODO: MEASURE AND CALIBRATE — these are placeholder values from G1, not measured for IRMV V3.
#       Replace with measured armature constants once available.
ARMATURE_HIP = 0.010177520  # placeholder, derived from G1's 7520-14 actuator
ARMATURE_KNEE = 0.025101925  # placeholder, derived from G1's 7520-22 actuator
ARMATURE_ANKLE = 0.003609725  # placeholder, derived from G1's 5020 actuator

NATURAL_FREQ = 10 * 2.0 * math.pi  # 10 Hz
DAMPING_RATIO = 2.0

STIFFNESS_HIP = ARMATURE_HIP * NATURAL_FREQ**2
STIFFNESS_KNEE = ARMATURE_KNEE * NATURAL_FREQ**2
STIFFNESS_ANKLE = ARMATURE_ANKLE * NATURAL_FREQ**2

DAMPING_HIP = 2.0 * DAMPING_RATIO * ARMATURE_HIP * NATURAL_FREQ
DAMPING_KNEE = 2.0 * DAMPING_RATIO * ARMATURE_KNEE * NATURAL_FREQ
DAMPING_ANKLE = 2.0 * DAMPING_RATIO * ARMATURE_ANKLE * NATURAL_FREQ

IRMV_V3_CFG = ArticulationCfg(
    spawn=sim_utils.UrdfFileCfg(
        fix_base=False,
        replace_cylinders_with_capsules=True,
        asset_path=f"{ASSET_DIR}/irmv_v3_description/urdf/irmv_v3.urdf",
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
        joint_drive=sim_utils.UrdfConverterCfg.JointDriveCfg(
            gains=sim_utils.UrdfConverterCfg.JointDriveCfg.PDGainsCfg(stiffness=0, damping=0)
        ),
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.75),
        joint_pos={
            ".*_hip_pitch_joint": -0.3,
            ".*_knee_joint": 0.6,
            ".*_ankle_pitch_joint": -0.3,
        },
        joint_vel={".*": 0.0},
    ),
    soft_joint_pos_limit_factor=0.9,
    actuators={
        "legs": ImplicitActuatorCfg(
            joint_names_expr=[
                ".*_hip_yaw_joint",
                ".*_hip_roll_joint",
                ".*_hip_pitch_joint",
                ".*_knee_joint",
            ],
            effort_limit_sim=50.0,
            velocity_limit_sim=30.0,
            stiffness={
                ".*_hip_pitch_joint": STIFFNESS_HIP,
                ".*_hip_roll_joint": STIFFNESS_KNEE,
                ".*_hip_yaw_joint": STIFFNESS_HIP,
                ".*_knee_joint": STIFFNESS_KNEE,
            },
            damping={
                ".*_hip_pitch_joint": DAMPING_HIP,
                ".*_hip_roll_joint": DAMPING_KNEE,
                ".*_hip_yaw_joint": DAMPING_HIP,
                ".*_knee_joint": DAMPING_KNEE,
            },
            armature={
                ".*_hip_pitch_joint": ARMATURE_HIP,
                ".*_hip_roll_joint": ARMATURE_KNEE,
                ".*_hip_yaw_joint": ARMATURE_HIP,
                ".*_knee_joint": ARMATURE_KNEE,
            },
        ),
        "feet": ImplicitActuatorCfg(
            joint_names_expr=[".*_ankle_pitch_joint", ".*_ankle_roll_joint"],
            effort_limit_sim=50.0,
            velocity_limit_sim=30.0,
            stiffness=2.0 * STIFFNESS_ANKLE,
            damping=2.0 * DAMPING_ANKLE,
            armature=2.0 * ARMATURE_ANKLE,
        ),
    },
)
