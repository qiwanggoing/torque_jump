# SPDX-FileCopyrightText: Copyright (c) 2021 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice, this
# list of conditions and the following disclaimer. 
#
# 2. Redistributions in binary form must reproduce the above copyright notice,
# this list of conditions and the following disclaimer in the documentation
# and/or other materials provided with the distribution.
#
# 3. Neither the name of the copyright holder nor the names of its
# contributors may be used to endorse or promote products derived from
# this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
#
# Copyright (c) 2021 ETH Zurich, Nikita Rudin

from legged_gym import LEGGED_GYM_ROOT_DIR
import os

import isaacgym
from legged_gym.envs import *
from legged_gym.utils import get_args, export_policy_as_jit, task_registry, Logger

import numpy as np
import torch

# For keyboard input (non-blocking)
import sys
import select
import termios
import tty
import time

def get_non_blocking_input():
    # Helper to get non-blocking keyboard input
    fd = sys.stdin.fileno()
    old_settings = termios.tcgetattr(fd)
    try:
        tty.setraw(sys.stdin.fileno())
        ch = sys.stdin.read(1)
    finally:
        termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)
    return ch if ch else None

def play(args):
    # Set the task to our new trigger jump environment
    args.task = 'go2_trigger_jump' 

    env_cfg, train_cfg = task_registry.get_cfgs(name=args.task)
    # override some parameters for testing
    env_cfg.env.num_envs = min(env_cfg.env.num_envs, 1) # Only one environment for interactive play
    env_cfg.env.episode_length_s = 60 # Longer episodes for play
    # env_cfg.control.control_type = 'T' # This should be handled by the env's config
    env_cfg.test.use_test = True
    env_cfg.test.checkpoint = 3000 # Default checkpoint to load, can be overridden by args
    env_cfg.test.vel = torch.tensor([0.0, 0.0, 0., 0.0], dtype=torch.float32)

    # Disable complex actuator models that might cause fatigue/performance drop for jumping tasks
    env_cfg.control.activation_process = False
    env_cfg.control.hill_model = False
    env_cfg.control.motor_fatigue = False
    env_cfg.commands.heading_command = False # We control angular velocity directly

    
    # Disable domain randomization for consistent play experience
    env_cfg.noise.add_noise = False
    env_cfg.domain_rand.randomize_friction = False
    env_cfg.domain_rand.push_robots = False
    env_cfg.domain_rand.randomize_base_mass = False
    
    # Ensure terrain is flat for easier testing
    env_cfg.terrain.mesh_type = 'plane'  
    env_cfg.terrain.num_rows = 1
    env_cfg.terrain.num_cols = 1
    env_cfg.terrain.curriculum = False
    env_cfg.terrain.terrain_proportions = [1, 0, 0, 0, 0] # flat terrain
    
    # Disable command resampling to prevent interference with our script control
    env_cfg.commands.resampling_time = 1000.0 
    
    # prepare environment
    env, _ = task_registry.make_env(name=args.task, args=args, env_cfg=env_cfg)
    obs = env.get_observations()
    
    # load policy
    train_cfg.runner.resume = True
    
    # --- AUTO LOAD LATEST MODEL LOGIC ---
    if args.load_run == -1 and args.checkpoint == -1: # If no specific run/checkpoint is specified
        log_dir = os.path.join(LEGGED_GYM_ROOT_DIR, 'logs', train_cfg.runner.experiment_name)
        if os.path.exists(log_dir):
            all_runs = sorted(os.listdir(log_dir))
            if all_runs:
                latest_run = all_runs[-1] # Get the latest run (last in sorted list)
                train_cfg.runner.load_run = latest_run
                
                # Try to find the latest checkpoint in that run
                run_path = os.path.join(log_dir, latest_run)
                all_checkpoints = sorted([f for f in os.listdir(run_path) if f.startswith('model_')])
                if all_checkpoints:
                    latest_checkpoint_file = all_checkpoints[-1]
                    # Extract checkpoint number (e.g., 'model_3000.pt' -> 3000)
                    train_cfg.runner.checkpoint = int(latest_checkpoint_file.replace('model_', '').replace('.pt', ''))
                    print(f"Automatically loaded latest model: {latest_run}/{latest_checkpoint_file}")
                else:
                    print(f"No checkpoints found in latest run '{latest_run}'. Please specify --load_run and --checkpoint manually.")
                    exit()
            else:
                print(f"No runs found in '{log_dir}'. Please train the model first or specify --load_run and --checkpoint manually.")
                exit()
        else:
            print(f"Log directory '{log_dir}' not found. Please train the model first or specify --load_run and --checkpoint manually.")
            exit()
    # --- END AUTO LOAD LATEST MODEL LOGIC ---

    ppo_runner, train_cfg = task_registry.make_alg_runner(env=env, name=args.task, args=args, train_cfg=train_cfg)
    policy = ppo_runner.get_inference_policy(device=env.device)

    # export policy as a jit module (used to run it from C++)
    EXPORT_POLICY = False # Set to True if you want to export the policy
    if EXPORT_POLICY:
        path = os.path.join(LEGGED_GYM_ROOT_DIR, 'logs', train_cfg.runner.experiment_name, 'exported', 'policies')
        export_policy_as_jit(ppo_runner.alg.actor_critic, path)
        print('Exported policy as jit script to: ', path)

    # --- Automatic Play Setup ---
    print("\n--- Automatic Play Mode (Single Jump Verification) ---")
    print("  Robot will execute a single jump every 3.5 seconds.")
    print("  Esc: Exit")
    print("---------------------------\n")

    single_jump_interval_s = 3.5 # Total time for one jump cycle (jump + rest)
    trigger_pulse_duration_s = 1.5 # Hold trigger for full jump cycle (1.5s) to verify mechanics
    
    # Initialize current_commands BEFORE the loop
    current_commands = torch.zeros(env.num_envs, env.cfg.commands.num_commands, dtype=torch.float, device=env.device)

    old_settings = termios.tcgetattr(sys.stdin) # Save terminal settings
    
    try:
        tty.setraw(sys.stdin.fileno()) # Set terminal to raw mode for non-blocking input
        
        for i in range(10 * int(env.max_episode_length)): # Run for a long time
            # 1. Handle Exit Key (Esc)
            if select.select([sys.stdin], [], [], 0)[0]:
                key_pressed = sys.stdin.read(1)
                if key_pressed == '\x1b': # Esc key
                    print("Exiting...")
                    break

            # 2. Single Jump Verification Logic
            current_time = i * env.dt
            cycle_phase = current_time % single_jump_interval_s
            
            # Reset commands
            current_commands[:] = 0.0 # Clear all commands first
            
            # Trigger jump at the start of each single_jump_interval
            if cycle_phase < trigger_pulse_duration_s:
                current_commands[:, 3] = 1.0 # Trigger ON
                current_commands[:, 0] = 1.0 # Forward velocity for jump (adjust as needed)
                mode_str = "TRIGGER ON"
            else:
                current_commands[:, 3] = 0.0 # Trigger OFF
                mode_str = "STAND/RESET"
            
            # Apply commands
            env.commands[:] = current_commands 

            actions = policy(obs.detach())
            obs, _, rews, dones, infos = env.step(actions.detach())
            
            # Debugging Output - adjust frequency as needed
            if i % 50 == 0:
                print(f"T={current_time:.2f}s | CmdTrig={env.commands[0, 3].item():.0f} | CmdVelX={env.commands[0,0].item():.1f} | Phase={env.phase_acc[0].item():.2f} | Jumping={env.is_jumping[0].item()}")

            if dones: 
                print(f"--- Environment Reset at T={current_time:.2f}s ---")
                env.reset()

                
            # Optional: Move camera to follow robot
            MOVE_CAMERA = True
            if MOVE_CAMERA:
                robot_pos = env.root_states[0, :3].cpu().numpy()
                camera_position = robot_pos + np.array([1, 1, 1])
                env.set_camera(camera_position, robot_pos)

    finally:
        termios.tcsetattr(sys.stdin, termios.TCSADRAIN, old_settings) # Restore terminal settings

if __name__ == '__main__':
    args = get_args()
    play(args)
