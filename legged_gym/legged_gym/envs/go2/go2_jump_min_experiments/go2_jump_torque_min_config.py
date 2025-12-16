from legged_gym.envs.go2.go2_jump.go2_jump_config import GO2JumpCfg, GO2JumpCfgPPO
from legged_gym.envs.go2.go2_jump_min_experiments.go2_jump_min_env import GO2JumpMinEnv

class GO2JumpTorqueMinCfg(GO2JumpCfg):
    class env(GO2JumpCfg.env):
        episode_length_s = 24
        num_envs = 2048 

    class asset(GO2JumpCfg.asset):
        # Extremely strict termination: Any contact EXCEPT feet results in termination.
        terminate_after_contacts_on = ["base", "FL_hip", "FL_thigh", "FL_calf", "FR_hip", "FR_thigh", "FR_calf", "Head_upper", "Head_lower", "RL_hip", "RL_thigh", "RL_calf", "RR_hip", "RR_thigh", "RR_calf"]

    class init_state(GO2JumpCfg.init_state):
        # Using jump default angles
        default_joint_angles = {
            'FL_hip_joint': 0.1,   'RL_hip_joint': 0.1,
            'FR_hip_joint': -0.1 , 'RR_hip_joint': -0.1,
            'FL_thigh_joint': 0.8, 'RL_thigh_joint': 1.0,
            'FR_thigh_joint': 0.8, 'RR_thigh_joint': 1.0,
            'FL_calf_joint': -1.5, 'RL_calf_joint': -1.5,
            'FR_calf_joint': -1.5, 'RR_calf_joint': -1.5,
        }

    class control(GO2JumpCfg.control):
        control_type = 'P_Residual' # Torque Control (Residual)
        stiffness = {'joint': 20.} # Consistent Kp
        damping = {'joint': 0.5} # Consistent Kd
        action_scale = 5.0 # Torque action scale (equivalent to 0.25 rad * 20 Kp)

    class commands(GO2JumpCfg.commands):
        class ranges:
            lin_vel_x = [0.0, 0.0] # Pure vertical jump, no forward movement
            lin_vel_y = [0.0, 0.0] 
            ang_vel_yaw = [0.0, 0.0] 
            heading = [0.0, 0.0]

    class rewards(GO2JumpCfg.rewards):
        class scales:
            termination = -200.0
            tracking_lin_vel = 1.0 
            tracking_ang_vel = 1.0
            lin_vel_z = 0.0 
            
            # --- STABILITY & SAFETY ---
            ang_vel_xy = -0.2 # NEW: Penalize spinning/flipping in air
            orientation = 20.0 # Boosted: Staying upright is priority #1
            
            base_height = 5.0 
            feet_air_time = 2.0 
            
            # --- VERTICAL DRIVE ---
            jump_velocity = 10.0 
            forward_velocity = 0.0 
            
            # --- STABILITY PENALTIES ---
            action_rate = -0.01 
            dof_vel = -0.01
            torques = -0.0001
            
            # --- Constraints REMOVED ---
            dof_acc = -0.0 
            feet_contact_forces = -0.0 
            feet_clearance = 0.0 
            collision = -0.0
            feet_stumble = -0.0
            jump = 0.0 
            
            # --- Convergence Helpers KEPT ---
            default_pos = -0.5 
            default_hip_pos = -0.5
            
            # All other specific SATA torque rewards remain disabled as in base cfg
            
        cycle_time = 1.5 
        base_height_target = 0.40 # Restored to normal standing height
        only_positive_rewards = False

class GO2JumpTorqueMinCfgPPO(GO2JumpCfgPPO):
    seed = 1
    class runner(GO2JumpCfgPPO.runner):
        experiment_name = 'go2_jump_torque_min' # Unique experiment name
        run_name = ''
