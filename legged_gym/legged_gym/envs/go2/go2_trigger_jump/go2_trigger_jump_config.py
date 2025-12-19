from legged_gym.envs.go2.go2_jump_control.go2_jump_control_config import GO2JumpControlCfg, GO2JumpControlCfgPPO

class GO2TriggerJumpCfg(GO2JumpControlCfg):
    class env(GO2JumpControlCfg.env):
        # 保持参数不变
        episode_length_s = 24
        num_envs = 4096

    class commands(GO2JumpControlCfg.commands):
        class ranges:
            lin_vel_x = [-0.5, 1.5]   # 最大速度限制在 1.5 m/s   
            lin_vel_y = [-0.3, 0.3]   
            ang_vel_yaw = [-0.5, 0.5] 
            heading = [0.0, 0.0]      
            
            # 这里的 commands[3] 不再是频率，而是 Trigger 信号 (0 or 1)
            # 我们在 Env 中会特殊处理采样，这里写 [0, 1] 只是占位
            jump_freq = [0.0, 1.0] 

    class asset(GO2JumpControlCfg.asset):
        # 显式重写，确保继承无误，双重保险
        terminate_after_contacts_on = [
            "base", "Head_upper", "Head_lower",
            "FL_hip", "FL_thigh", "FL_calf", "FL_calflower", "FL_calflower1",
            "FR_hip", "FR_thigh", "FR_calf", "FR_calflower", "FR_calflower1",
            "RL_hip", "RL_thigh", "RL_calf", "RL_calflower", "RL_calflower1",
            "RR_hip", "RR_thigh", "RR_calf", "RR_calflower", "RR_calflower1",
        ]

    class rewards(GO2JumpControlCfg.rewards):
        class scales(GO2JumpControlCfg.rewards.scales):
            # 恢复均衡的奖励设置
            tracking_lin_vel = 5.0 
            tracking_ang_vel = 2.0
            
            # 适度的姿态限制
            orientation = 5.0 
            
            # 保持默认姿态惩罚较低
            default_pos = -0.1 
            
            # 禁用显式站立奖励，靠 dof_vel/torques/action_rate 惩罚来实现静止
            stand_still = 0.0 
            
            # 其他保持不变
            
class GO2TriggerJumpCfgPPO(GO2JumpControlCfgPPO):
    seed = 1
    class runner(GO2JumpControlCfgPPO.runner):
        experiment_name = 'go2_trigger_jump'
        run_name = ''
