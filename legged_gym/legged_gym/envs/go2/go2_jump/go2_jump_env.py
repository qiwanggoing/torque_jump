from legged_gym.envs.go2.go2_torque.go2_torque import GO2Torque
from isaacgym import gymtorch
from isaacgym.torch_utils import torch_rand_float
import torch
import numpy as np
from isaacgym.torch_utils import get_euler_xyz

class GO2JumpEnv(GO2Torque):
    def __init__(self, cfg, sim_params, physics_engine, sim_device, headless):
        super().__init__(cfg, sim_params, physics_engine, sim_device, headless)
        
        # Explicitly initialize add_noise if not handled by parent correctly
        self.add_noise = self.cfg.noise.add_noise

        # Re-initialize P/D gains for P_Residual
        self.p_gains = torch.zeros(self.num_actions, dtype=torch.float, device=self.device, requires_grad=False)
        self.d_gains = torch.zeros(self.num_actions, dtype=torch.float, device=self.device, requires_grad=False)
        
        for i in range(self.num_dofs):
            self.p_gains[i] = self.cfg.control.stiffness['joint']
            self.d_gains[i] = self.cfg.control.damping['joint']
            
        self.commands_scale = torch.tensor([self.obs_scales.lin_vel, self.obs_scales.lin_vel, self.obs_scales.ang_vel], device=self.device, requires_grad=False,)
        
    def _reset_dofs(self, env_ids):
        # Override GO2Torque._reset_dofs to remove fatigue logic and dependency on self.add_noise (which caused AttributeError)
        self.dof_pos[env_ids] = (
                self.default_dof_pos * torch_rand_float(0.95, 1.05, (len(env_ids), self.num_dof), device=self.device))
        self.dof_vel[env_ids] = 0.
        # We don't use activation_sign or motor_fatigue in jump env
        
        env_ids_int32 = env_ids.to(dtype=torch.int32)
        self.gym.set_dof_state_tensor_indexed(self.sim,
                                              gymtorch.unwrap_tensor(self.dof_state),
                                              gymtorch.unwrap_tensor(env_ids_int32), len(env_ids_int32))
        
    def _compute_torques(self, actions):
        # PD + Residual logic
        pd_torques = self.p_gains * (self.default_dof_pos - self.dof_pos) - self.d_gains * self.dof_vel
        actions_scaled = actions * self.cfg.control.action_scale
        
        torques = pd_torques + actions_scaled
        return torch.clip(torques, -self.torque_limits, self.torque_limits)

    def _get_phase(self):
        cycle_time = self.cfg.rewards.cycle_time
        phase = self.episode_length_buf * self.dt / cycle_time
        return phase

    def _get_gait_phase(self):
        phase = self._get_phase()
        phase_mod = phase % 1.0
        stance_mask = torch.zeros((self.num_envs, 2), device=self.device)
        # 60% duty cycle for jump
        stance_mask[:, 0] = phase_mod < 0.6
        stance_mask[:, 1] = phase_mod > 0.6 
        return stance_mask

    def compute_observations(self):
        base_lin_vel = self.base_lin_vel
        motor_fatigue = torch.zeros_like(self.dof_pos) # Zero filled as we disabled fatigue
        
        # Phase signal
        phase = self._get_phase()
        sin_pos = torch.sin(2 * torch.pi * phase).unsqueeze(1)
        cos_pos = torch.cos(2 * torch.pi * phase).unsqueeze(1)
        
        obs_buf = torch.cat((
            base_lin_vel * self.obs_scales.lin_vel,             # 3
            self.base_ang_vel * self.obs_scales.ang_vel,        # 3
            self.projected_gravity,                             # 3
            (self.dof_pos - self.default_dof_pos) * self.obs_scales.dof_pos, # 12
            self.dof_vel * self.obs_scales.dof_vel,             # 12
            self.commands[:, :3] * self.commands_scale,         # 3
            self.torques,                                       # 12
            motor_fatigue,                                      # 12
            sin_pos,                                            # 1
            cos_pos                                             # 1
        ), dim=-1) 
        
        if self.add_noise:
            noise_vec = self._get_noise_scale_vec(None) 
            obs_buf += (2 * torch.rand_like(obs_buf) - 1) * noise_vec

        self.obs_buf = obs_buf

    def _get_noise_scale_vec(self, cfg):
        noise_vec = torch.zeros(62, device=self.device)
        noise_scales = self.cfg.noise.noise_scales
        noise_level = self.cfg.noise.noise_level
        
        noise_vec[:3] = noise_scales.lin_vel * noise_level * self.obs_scales.lin_vel
        noise_vec[3:6] = noise_scales.ang_vel * noise_level * self.obs_scales.ang_vel
        noise_vec[6:9] = noise_scales.gravity * noise_level
        noise_vec[9:21] = noise_scales.dof_pos * noise_level * self.obs_scales.dof_pos
        noise_vec[21:33] = noise_scales.dof_vel * noise_level * self.obs_scales.dof_vel
        noise_vec[33:] = 0. 
        
        return noise_vec

    # ================= Rewards =================
    
    def _reward_jump(self):
        contact = self.contact_forces[:, self.feet_indices, 2] > 5.
        stance_mask = self._get_gait_phase()
        # Sync Jump: All feet contact == stance_mask[0]
        # Assuming feet_indices order is consistent (FL, RL, FR, RR or similar)
        # Checking consistency: (0==1) & (1==2) & (2==3) ensures all 4 are same.
        # & (3==stance) ensures they match the target phase.
        JUMP = (contact[:,0] == contact[:,1]) & \
            (contact[:,1] == contact[:,2]) & \
            (contact[:,2] == contact[:,3]) & \
            (contact[:,3] == stance_mask[:,0])
        return JUMP.float() * (torch.norm(self.commands[:, :2], dim=1) > 0.2)

    def _reward_feet_clearance(self):
        feet_height = self.rigid_body_states[:, self.feet_indices, 2] - 0.02
        stance_mask = self._get_gait_phase()
        swing_mask = 1.0 - stance_mask[:, 0].unsqueeze(1).repeat(1, 4)
        rew_pos = torch.clip(feet_height, min=0, max=0.05)
        rew_pos = torch.sum(rew_pos * swing_mask, dim=1)
        return rew_pos * (torch.norm(self.commands[:, :2], dim=1) > 0.2)

    def _reward_base_height(self):
        base_height = torch.mean(self.root_states[:, 2].unsqueeze(1) - self.measured_heights, dim=1)
        return torch.exp(-torch.abs(base_height - self.cfg.rewards.base_height_target) * 10)

    def _reward_orientation(self):
        return torch.exp(-torch.norm(self.projected_gravity[:, :2], dim=1) * 10)

    def _reward_lin_vel_z(self):
        return torch.exp(-torch.abs(self.base_lin_vel[:, 2]))
    
    def _reward_ang_vel_xy(self):
        return torch.exp(-torch.norm(torch.abs(self.base_ang_vel[:, :2]), dim=1))

    def _reward_tracking_lin_vel(self):
        lin_vel_error = torch.sum(torch.square(self.commands[:, :2] - self.base_lin_vel[:, :2]), dim=1)
        return torch.exp(-lin_vel_error/self.cfg.rewards.tracking_sigma)

    def _reward_tracking_ang_vel(self):
        ang_vel_error = torch.square(self.commands[:, 2] - self.base_ang_vel[:, 2])
        return torch.exp(-ang_vel_error/self.cfg.rewards.tracking_sigma)
    
    def _reward_feet_contact_forces(self):
        return torch.sum((torch.norm(self.contact_forces[:, self.feet_indices, :], dim=-1) -  self.cfg.rewards.max_contact_force).clip(min=0.), dim=1)

    def _reward_default_pos(self):
        return torch.sum(torch.abs(self.dof_pos - self.default_dof_pos), dim=1)

    def _reward_default_hip_pos(self):
        joint_diff = torch.abs(self.dof_pos[:,0])+torch.abs(self.dof_pos[:,3])+torch.abs(self.dof_pos[:,6])+torch.abs(self.dof_pos[:,9])
        return torch.exp(-joint_diff * 4)

    def _reward_feet_air_time(self):
        # Reward long steps
        # Need to filter the contacts because the contact reporting of PhysX is unreliable on meshes
        contact = self.contact_forces[:, self.feet_indices, 2] > 1.
        contact_filt = torch.logical_or(contact, self.last_contacts) 
        self.last_contacts = contact
        first_contact = (self.feet_air_time > 0.) * contact_filt
        self.feet_air_time += self.dt
        rew_airTime = torch.sum((self.feet_air_time - 0.3) * first_contact, dim=1) # Threshold lowered to 0.3
        rew_airTime *= torch.norm(self.commands[:, :2], dim=1) > 0.1 #no reward for zero command
        self.feet_air_time *= ~contact_filt
        return rew_airTime

    # Disable old rewards explicitly
    def _reward_forward(self): return 0.
    def _reward_head_height(self): return 0.
    def _reward_moving_y(self): return 0.
    def _reward_moving_yaw(self): return 0.
    def _reward_motor_fatigue(self): return 0.
    def _reward_roll(self): return 0.
