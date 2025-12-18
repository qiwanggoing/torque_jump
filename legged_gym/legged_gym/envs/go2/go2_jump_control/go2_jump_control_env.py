from legged_gym.envs.go2.go2_jump_min_experiments.go2_jump_min_env import GO2JumpMinEnv
import torch
from isaacgym.torch_utils import torch_rand_float

class GO2JumpControlEnv(GO2JumpMinEnv):
    """
    GO2JumpControlEnv
    
    继承自 GO2JumpMinEnv (Min Torque 实现)。
    增加了对跳跃方向 (lin_vel_x, lin_vel_y) 和跳跃频率 (通过 command[:, 3]) 的控制。
    """

    def _resample_commands(self, env_ids):
        """
        重写命令采样函数。
        index 0: lin_vel_x
        index 1: lin_vel_y
        index 2: ang_vel_yaw
        index 3: jump_freq (Hz) -> 用来控制跳跃周期
        """
        # 1. 采样标准速度命令 (利用父类逻辑，但我们需要手动覆盖 index 3)
        # 我们可以直接调用父类的 _resample_commands 来处理前三个命令，或者全部重写以防万一。
        # 为了清晰，这里全部重写。
        
        self.commands[env_ids, 0] = torch_rand_float(self.command_ranges["lin_vel_x"][0],
                                                     self.command_ranges["lin_vel_x"][1], (len(env_ids), 1),
                                                     device=self.device).squeeze(1)
        self.commands[env_ids, 1] = torch_rand_float(self.command_ranges["lin_vel_y"][0],
                                                     self.command_ranges["lin_vel_y"][1], (len(env_ids), 1),
                                                     device=self.device).squeeze(1)
        self.commands[env_ids, 2] = torch_rand_float(self.command_ranges["ang_vel_yaw"][0],
                                                     self.command_ranges["ang_vel_yaw"][1], (len(env_ids), 1),
                                                     device=self.device).squeeze(1)

        # 2. 采样跳跃频率 (利用 Config 中新增的 ranges.jump_freq)
        # 注意：我们需要确保 config 中存在 jump_freq，否则回退到默认值
        freq_range = getattr(self.cfg.commands.ranges, 'jump_freq', [0.66, 0.66]) # 默认 1.5s 周期 -> 0.66Hz
        
        self.commands[env_ids, 3] = torch_rand_float(freq_range[0],
                                                     freq_range[1], (len(env_ids), 1),
                                                     device=self.device).squeeze(1)

        # 处理小命令归零 (Deadband)，保持与原版逻辑一致
        self.commands[env_ids, :2] *= (torch.norm(self.commands[env_ids, :2], dim=1) > 0.2).unsqueeze(1)

    def _get_phase(self):
        """
        计算当前的相位。
        Phase = time * frequency
        这里 frequency 来自 commands[:, 3]
        """
        # commands[:, 3] 是频率 (Hz)
        frequencies = self.commands[:, 3]
        
        # 保护机制：防止频率为 0 或负数
        frequencies = torch.clip(frequencies, min=0.1)
        
        # 相位 = 时间 * 频率
        phase = self.episode_length_buf * self.dt * frequencies
        return phase

    def _get_gait_phase(self):
        """
        覆盖 gait phase 计算，使用动态的 phase。
        """
        phase = self._get_phase()
        phase_mod = phase % 1.0
        stance_mask = torch.zeros((self.num_envs, 2), device=self.device)
        
        # 保持原本的 60% 占空比 (Duty Cycle)
        # 0.6 表示 60% 的时间是腾空准备/站立，40% 是其他？ 
        # 原代码：stance_mask[:, 0] = phase_mod < 0.6
        # 需要确认原代码语义。在 GO2JumpEnv 中：
        # stance_mask[:, 0] = phase_mod < 0.6
        # stance_mask[:, 1] = phase_mod > 0.6 
        # 并在 _reward_jump 中使用了 stance_mask[:, 0] 作为目标相位。
        # 我们保持这个逻辑不变，只是 phase 来源变了。
        
        stance_mask[:, 0] = phase_mod < 0.6
        stance_mask[:, 1] = phase_mod > 0.6 
        return stance_mask
