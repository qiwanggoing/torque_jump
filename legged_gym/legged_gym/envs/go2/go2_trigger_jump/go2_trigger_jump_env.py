from legged_gym.envs.go2.go2_jump_control.go2_jump_control_env import GO2JumpControlEnv
import torch
from isaacgym.torch_utils import torch_rand_float

class GO2TriggerJumpEnv(GO2JumpControlEnv):
    """
    GO2TriggerJumpEnv (Updated with Pulse Training Logic)
    
    实现“单次触发跳跃” (One-shot Trigger Jump)。
    
    训练关键逻辑：
    - 为了让 Policy 学会“按一下跳一下”以及“跳完这一下再停”，我们必须在训练中模拟 Trigger 的消失。
    - 使用 command_timers 来控制 Trigger 在一段时间后自动归零，强迫 Policy 在 Trigger=0 的情况下完成剩余的 Phase。
    """
    
    def __init__(self, cfg, sim_params, physics_engine, sim_device, headless):
        super().__init__(cfg, sim_params, physics_engine, sim_device, headless)
        
        # 状态标志：是否正在执行跳跃动作
        self.is_jumping = torch.zeros(self.num_envs, device=self.device, dtype=torch.bool)
        
        # 相位累加器
        self.phase_acc = torch.zeros(self.num_envs, device=self.device, dtype=torch.float)
        
        # Trigger 自动关闭计时器
        # -1.0 表示“长按模式” (Trigger 保持开启)
        # >0.0 表示“脉冲模式” (倒计时结束后 Trigger 归零)
        self.command_timers = torch.zeros(self.num_envs, device=self.device, dtype=torch.float)
        
        # 固定周期时间
        self.cycle_time = self.cfg.rewards.cycle_time

    def _resample_commands(self, env_ids):
        """
        采样命令。
        """
        # 1. 采样方向命令 (手动实现父类逻辑)
        self.commands[env_ids, 0] = torch_rand_float(self.command_ranges["lin_vel_x"][0],
                                                     self.command_ranges["lin_vel_x"][1], (len(env_ids), 1),
                                                     device=self.device).squeeze(1)
        self.commands[env_ids, 1] = torch_rand_float(self.command_ranges["lin_vel_y"][0],
                                                     self.command_ranges["lin_vel_y"][1], (len(env_ids), 1),
                                                     device=self.device).squeeze(1)
        self.commands[env_ids, 2] = torch_rand_float(self.command_ranges["ang_vel_yaw"][0],
                                                     self.command_ranges["ang_vel_yaw"][1], (len(env_ids), 1),
                                                     device=self.device).squeeze(1)
        
        self.commands[env_ids, :2] *= (torch.norm(self.commands[env_ids, :2], dim=1) > 0.2).unsqueeze(1)
        
        # 2. 覆盖 commands[3] 为 Trigger 信号
        # 三种模式：
        # A. 连续跳跃 (Trigger=1, Timer=-1) - 40%
        # B. 单次跳跃 (Trigger=1, Timer=0.3s) - 40%
        # C. 站立 (Trigger=0, Timer=-1) - 20%
        
        rand_mode = torch.rand(len(env_ids), device=self.device)
        
        # 默认为站立 (Mode C)
        trigger = torch.zeros(len(env_ids), device=self.device)
        timers = torch.ones(len(env_ids), device=self.device) * -1.0
        
        # Mode A: 连续跳跃 (0.0 - 0.4)
        mode_a = rand_mode < 0.4
        trigger[mode_a] = 1.0
        timers[mode_a] = -1.0 # 保持开启
        
        # Mode B: 单次跳跃 (0.4 - 0.8)
        mode_b = (rand_mode >= 0.4) & (rand_mode < 0.8)
        trigger[mode_b] = 1.0
        # 随机脉冲长度 0.2s - 0.6s (确保小于一个周期 1.5s，让 Policy 体验 Trigger=0 Phase>0)
        timers[mode_b] = torch_rand_float(0.2, 0.6, (len(env_ids), 1), device=self.device).squeeze(1)[mode_b]
        
        self.commands[env_ids, 3] = trigger
        self.command_timers[env_ids] = timers
        
        # 处理 Reset 状态重置
        is_resetting = self.reset_buf[env_ids]
        if torch.any(is_resetting):
            is_resetting_bool = is_resetting.to(dtype=torch.bool)
            reset_indices = env_ids[is_resetting_bool]
            
            self.phase_acc[reset_indices] = 0.0
            # Reset 时，状态跟随 Trigger
            self.is_jumping[reset_indices] = (trigger[is_resetting_bool] > 0.5)

    def _post_physics_step_callback(self):
        """
        在每步物理模拟后调用。用于更新计时器和处理 Trigger 自动归零。
        """
        # 调用父类 (处理 heading 等)
        super()._post_physics_step_callback()
        
        # 更新计时器
        # 只有 timer > 0 的才需要倒计时
        active_timers = self.command_timers > 0.0
        self.command_timers[active_timers] -= self.dt
        
        # 检查是否应该关闭 Trigger
        # 条件：timer 刚刚变为负数 (<= 0) 且原本是正数的 (active_timers)
        # 简化逻辑：只要 timer <= 0 且 timer > -0.5 (排除 -1.0 的长按模式)，就把 trigger 设为 0
        # 更简单：每次 step，凡是 timer <= 0 且 timer != -1.0 的，Trigger 设为 0
        
        should_turn_off = (self.command_timers <= 0.0) & (self.command_timers != -1.0)
        self.commands[should_turn_off, 3] = 0.0
        
        # 注意：这里改了 commands[3] 为 0，但 is_jumping 依然由 _get_phase 控制
        # _get_phase 会发现 trigger=0，但如果 phase < 1.0，它会继续维持 is_jumping=True
        # 这正是我们想要的训练效果！

    def _get_phase(self):
        trigger = self.commands[:, 3] > 0.5
        
        # 1. 尝试启动跳跃 (Start Condition)
        start_jump = trigger & (~self.is_jumping)
        self.is_jumping[start_jump] = True
        
        # 2. 更新相位 (Update Phase)
        self.phase_acc[self.is_jumping] += self.dt / self.cycle_time
        
        # 3. 检查周期结束 (End Condition)
        cycle_finished = self.phase_acc >= 1.0
        finished_envs = cycle_finished
        
        if torch.any(finished_envs):
            continue_mask = finished_envs & trigger
            self.phase_acc[continue_mask] -= 1.0
            
            stop_mask = finished_envs & (~trigger)
            self.phase_acc[stop_mask] = 0.0
            self.is_jumping[stop_mask] = False
            
        return self.phase_acc

    def _get_gait_phase(self):
        phase = self.phase_acc
        stance_mask = torch.zeros((self.num_envs, 2), device=self.device)
        stance_mask[:, 0] = phase < 0.6
        stance_mask[:, 1] = phase > 0.6 
        return stance_mask

    # ================= 奖励函数 (保持不变) =================
    
    def _reward_jump_velocity(self):
        vel_reward = torch.clip(self.base_lin_vel[:, 2], min=0.0)
        return vel_reward * self.is_jumping.float()

    def _reward_feet_air_time(self):
        contact = self.contact_forces[:, self.feet_indices, 2] > 1.
        is_air = 1.0 - contact.float()
        rew_airTime = torch.sum(is_air, dim=1)
        return rew_airTime * self.is_jumping.float()

    def _reward_jump(self):
        contact = self.contact_forces[:, self.feet_indices, 2] > 5.
        stance_mask = self._get_gait_phase()
        JUMP = (contact[:,0] == contact[:,1]) & \
            (contact[:,1] == contact[:,2]) & \
            (contact[:,2] == contact[:,3]) & \
            (contact[:,3] == stance_mask[:,0])
        return JUMP.float() * self.is_jumping.float()

    def _reward_tracking_lin_vel(self):
        lin_vel_error = torch.sum(torch.square(self.commands[:, :2] - self.base_lin_vel[:, :2]), dim=1)
        rew = torch.exp(-lin_vel_error / self.cfg.rewards.tracking_sigma)
        return rew * self.is_jumping.float()

    def _reward_tracking_ang_vel(self):
        ang_vel_error = torch.square(self.commands[:, 2] - self.base_ang_vel[:, 2])
        rew = torch.exp(-ang_vel_error / self.cfg.rewards.tracking_sigma)
        return rew * self.is_jumping.float()

    def _reward_stand_still(self):
        is_standing = ~self.is_jumping
        dof_pos_error = torch.sum(torch.abs(self.dof_pos - self.default_dof_pos), dim=1)
        vel_error = torch.norm(self.base_lin_vel, dim=1) + torch.norm(self.base_ang_vel, dim=1)
        rew = torch.exp(-dof_pos_error * 5.0) * torch.exp(-vel_error * 5.0)
        return rew * is_standing.float()