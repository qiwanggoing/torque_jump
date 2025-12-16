from legged_gym.envs.go2.go2_torque.go2_torque_config import GO2TorqueCfg, GO2TorqueCfgPPO

class GO2JumpCfg(GO2TorqueCfg):
    class env(GO2TorqueCfg.env):
        # SATA Torque (60) + Phase (2) = 62
        num_observations = 62
        num_actions = 12
        episode_length_s = 24
        # No frame stack for now
        
    class init_state(GO2TorqueCfg.init_state):
        pos = [0.0, 0.0, 0.42] # x,y,z [m]
        # Using jump default angles
        default_joint_angles = {
            'FL_hip_joint': 0.1,   'RL_hip_joint': 0.1,
            'FR_hip_joint': -0.1 , 'RR_hip_joint': -0.1,
            'FL_thigh_joint': 0.8, 'RL_thigh_joint': 1.0,
            'FR_thigh_joint': 0.8, 'RR_thigh_joint': 1.0,
            'FL_calf_joint': -1.5, 'RL_calf_joint': -1.5,
            'FR_calf_joint': -1.5, 'RR_calf_joint': -1.5,
        }

    class control(GO2TorqueCfg.control):
        control_type = 'P_Residual' # We will handle this in Env
        # Disable SATA complex torque features
        activation_process = False
        hill_model = False
        motor_fatigue = False
        
        
        # PD Params
        stiffness = {'joint': 20.} 
        damping = {'joint': 0.5}
        
        action_scale = 10.0 # Reverted from 20.0 to 10.0
        decimation = 4

    class growth(GO2TorqueCfg.growth):
        # Disable growth curriculum effectively
        max_torque_scale = 1.0
        start_torque_scale = 1.0
        max_rear_torque_scale = 1.0
        start_rear_torque_scale = 1.0
        max_freq = 200
        start_freq = 200

    class rewards(GO2TorqueCfg.rewards):
        class scales:
            termination = -10.0
            tracking_lin_vel = 2.0
            tracking_ang_vel = 2.0
            lin_vel_z = 0.0 
            ang_vel_xy = 0.5
            orientation = 2.0
            torques = -0.0001
            dof_vel = -0.0
            dof_acc = -2.5e-7 
            base_height = 3.0 # Restored from 0.0 to 3.0 to fix low stance
            feet_air_time = 1.0 
            collision = -1.0
            feet_stumble = -0.0
            action_rate = -0.05
            default_pos = -0.5 
            default_hip_pos = 1.0 
            feet_contact_forces = -0.01
            jump = 5.0
            feet_clearance = 0.0
            
            # Disable SATA Torque rewards
            dof_vel = -0.005 # Enable dof_vel penalty to suppress erratic leg movements
            dof_pos_limits = 0.0 # Disable parent config reward
            forward = 0.
            head_height = 0.
            moving_y = 0.
            moving_yaw = 0.
            soft_dof_pos_limits = 0.
            motor_fatigue = 0.
            roll = 0.

        cycle_time = 1.5
        base_height_target = 0.45
        max_contact_force = 100.
        target_feet_height = 0.05
        only_positive_rewards = False

class GO2JumpCfgPPO(GO2TorqueCfgPPO):
    seed = 1
    class algorithm(GO2TorqueCfgPPO.algorithm):
        entropy_coef = 0.01
        learning_rate = 1.e-4
        num_learning_epochs = 5
        sym_loss = True
        sym_coef = 1.0
        # SATA Jump 62-dim obs structure:
        # 0-2: lin_vel (flip y) -> 0, -1, 2
        # 3-5: ang_vel (flip x, z) -> -3, 4, -5
        # 6-8: gravity (flip y) -> 6, -7, 8
        # 9-20: dof_pos (FL<->FR, RL<->RR, flip Hip) -> -12, 13, 14, -9, 10, 11, -18, 19, 20, -15, 16, 17
        # 21-32: dof_vel (same) -> -24, 25, 26, -21, 22, 23, -30, 31, 32, -27, 28, 29
        # 33-35: commands (vx, vy, yaw -> vx, -vy, -yaw) -> 33, -34, -35
        # 36-47: torques (same as dof) -> -39, 40, 41, -36, 37, 38, -45, 46, 47, -42, 43, 44
        # 48-59: fatigue (swap only, scalar) -> 51, 52, 53, 48, 49, 50, 57, 58, 59, 54, 55, 56
        # 60-61: phase (no change) -> 60, 61
        obs_permutation = [
            0, -1, 2, 
            -3, 4, -5, 
            6, -7, 8, 
            -12, 13, 14, -9, 10, 11, -18, 19, 20, -15, 16, 17,
            -24, 25, 26, -21, 22, 23, -30, 31, 32, -27, 28, 29,
            33, -34, -35,
            -39, 40, 41, -36, 37, 38, -45, 46, 47, -42, 43, 44,
            51, 52, 53, 48, 49, 50, 57, 58, 59, 54, 55, 56,
            60, 61
        ]
        act_permutation = [-3, 4, 5, -0, 1, 2, -9, 10, 11, -6, 7, 8]
    
    class runner(GO2TorqueCfgPPO.runner):
        run_name = ''
        experiment_name = 'go2_jump_sata_torque'
        max_iterations = 5000
