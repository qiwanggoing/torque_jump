from legged_gym.envs.go2.go2_jump_min_experiments.go2_jump_torque_min_config import GO2JumpTorqueMinCfg, GO2JumpTorqueMinCfgPPO

class GO2JumpControlCfg(GO2JumpTorqueMinCfg):
    class env(GO2JumpTorqueMinCfg.env):
        # 保持参数不变
        episode_length_s = 24
        num_envs = 4096

    class commands(GO2JumpTorqueMinCfg.commands):
        class ranges:
            lin_vel_x = [-0.5, 1.0]   # 允许前后移动
            lin_vel_y = [-0.3, 0.3]   # 允许左右移动
            ang_vel_yaw = [-0.5, 0.5] # 允许少量转向
            heading = [0.0, 0.0]      # 不使用 heading
            
            # 新增：跳跃频率范围 (Hz)
            # 周期 = 1.0 / freq. 
            # 0.5Hz -> 2.0s 周期 (慢跳)
            # 1.5Hz -> 0.66s 周期 (快跳)
            jump_freq = [0.5, 1.25] 

    class asset(GO2JumpTorqueMinCfg.asset):
        # 显式覆盖终止接触列表，加入 URDF 中发现的 calflower 等细节 Link
        # 以防止小腿前侧或膝盖护板部位触地时不触发终止
        terminate_after_contacts_on = [
            "base", "Head_upper", "Head_lower",
            "FL_hip", "FL_thigh", "FL_calf", "FL_calflower", "FL_calflower1",
            "FR_hip", "FR_thigh", "FR_calf", "FR_calflower", "FR_calflower1",
            "RL_hip", "RL_thigh", "RL_calf", "RL_calflower", "RL_calflower1",
            "RR_hip", "RR_thigh", "RR_calf", "RR_calflower", "RR_calflower1",
        ]

    class rewards(GO2JumpTorqueMinCfg.rewards):
        class scales(GO2JumpTorqueMinCfg.rewards.scales):
            # 增加对速度追踪的奖励权重，让它听从方向指令
            tracking_lin_vel = 5.0 
            tracking_ang_vel = 2.0
            
            # --- 安全与防撞 ---
            # 根据用户要求，不启用 collision 惩罚，只依赖 termination
            collision = 0.0
            # 根据用户要求，不启用 dof_pos_limits 惩罚
            dof_pos_limits = 0.0
            
            # 保持原有的稳定性奖励
            termination = -200.0
            lin_vel_z = 0.0 # z轴速度仍然由 jump_velocity 奖励处理
            
            # 根据用户要求，将 orientation 奖励设置为 12.0
            orientation = 10.0 
            
            # 保持跳跃相关的奖励
            base_height = 5.0 
            feet_air_time = 2.0 
            jump_velocity = 10.0 
            
            # 增加默认姿态惩罚，避免不自然或导致触地的关节位置
            default_pos = -1.0 
            
            # 惩罚
            action_rate = -0.01 
            dof_vel = -0.01
            torques = -0.0001

class GO2JumpControlCfgPPO(GO2JumpTorqueMinCfgPPO):
    seed = 1
    class runner(GO2JumpTorqueMinCfgPPO.runner):
        experiment_name = 'go2_jump_control'
        run_name = ''
