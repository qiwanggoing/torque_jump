from legged_gym.envs.go2.go2_jump.go2_jump_env import GO2JumpEnv
import torch

class GO2JumpMinEnv(GO2JumpEnv):
    """
    GO2JumpMinEnv - Restored to EXACTLY match the state of successful Torque training (Dec 16).
    
    Verified state from logs:
    1. No `check_termination` override (used parent GO2Torque's logic).
    2. No `compute_observations` override (used parent GO2JumpEnv's logic).
    3. Has `_compute_torques` with PD gain multipliers and offsets.
    4. Has `_reward_default_hip_pos` with L1 penalty.
    5. Has `_reward_jump_velocity` / `_reward_feet_air_time` without command thresholds.
    """

    def __init__(self, cfg, sim_params, physics_engine, sim_device, headless):
        super().__init__(cfg, sim_params, physics_engine, sim_device, headless)
        
        # --- Debug prints were present in the old version, keeping them for verification ---
        body_names = self.gym.get_actor_rigid_body_names(self.envs[0], self.actor_handles[0])
        term_indices = self.termination_contact_indices.cpu().numpy()
        # (Simplified print to match file structure)

    def _compute_torques(self, actions):
        # 15th Dec / 16th Dec Logic
        # Apply domain randomization multipliers
        p_mult = getattr(self, 'p_gains_multiplier', 1.0)
        d_mult = getattr(self, 'd_gains_multiplier', 1.0)
        
        p_gains = self.p_gains * p_mult
        d_gains = self.d_gains * d_mult
        
        # Include motor_zero_offsets
        motor_offsets = getattr(self, 'motor_zero_offsets', torch.zeros_like(self.default_dof_pos))
        pd_torques = p_gains * (self.default_dof_pos + motor_offsets - self.dof_pos) - d_gains * self.dof_vel
        
        actions_scaled = actions * self.cfg.control.action_scale
        torques = pd_torques + actions_scaled
            
        return torch.clip(torques, -self.torque_limits, self.torque_limits)

    # NO compute_observations override (Inherit from GO2JumpEnv)
    # NO check_termination override (Inherit from GO2Torque)

    def _reward_feet_air_time(self):
        # [RESTORED] Dense air time reward, no command threshold
        contact = self.contact_forces[:, self.feet_indices, 2] > 1.
        is_air = 1.0 - contact.float()
        rew_airTime = torch.sum(is_air, dim=1)
        return rew_airTime

    def _reward_jump_velocity(self):
        # [RESTORED] Reward upward velocity, no command threshold
        return torch.clip(self.base_lin_vel[:, 2], min=0.0)

    def _reward_forward_velocity(self):
        return torch.clip(self.base_lin_vel[:, 0], min=0.0)

    def _reward_feet_clearance(self):
        # [RESTORED]
        self.feet_height = self.rigid_state[:, self.feet_indices, 2] - 0.02
        swing_mask = 1 - self._get_gait_phase()
        rew_pos = torch.clip(self.feet_height, min=0, max=0.05)
        rew_pos = torch.sum(rew_pos * swing_mask[:,:1].repeat(1,4), dim=1)
        return rew_pos

    def _reward_jump(self):
        # [RESTORED]
        contact = self.contact_forces[:, self.feet_indices, 2] > 5.
        stance_mask = self._get_gait_phase()
        JUMP = (contact[:,0] == contact[:,1]) & \
            (contact[:,1] == contact[:,2]) & \
            (contact[:,2] == contact[:,3]) & \
            (contact[:,3] == stance_mask[:,0])
        return JUMP.float()

    def _reward_default_hip_pos(self):
        # Strict L1 penalty for hip deviation
        hip_indices = [0, 3, 6, 9]
        return torch.sum(torch.abs(self.dof_pos[:, hip_indices] - self.default_dof_pos[:, hip_indices]), dim=1)