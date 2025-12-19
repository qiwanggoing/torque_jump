# Torque Jump Experiment (Based on SATA)

> Built upon the [SATA Framework](https://arxiv.org/abs/2502.12674), this project implements a high-performance **Residual Torque Control** policy for vertical jumping on the Unitree Go2.

---

## 🎯 Objective

To train a robust, explosive vertical jumping policy using pure torque control, demonstrating the superiority of **force-based feedback** over traditional position control for dynamic aerial maneuvers.

---

## 💡 Key Implementation Concepts

### 1. Observation Space: The Power of Torque Feedback
Instead of relying solely on joint positions and velocities, our policy explicitly observes the **Applied Torques** (62-dim observation vector).
*   **Why?** In a residual control setup, knowing the exact torque currently being applied allows the network to "feel" the ground reaction forces and the robot's internal stress.
*   This is critical for learning compliant landings and precise takeoff timing, which position-based observations often miss due to stiffness.

### 2. Minimalist & Strict Reward Structure
We moved away from complex, feature-engineered reward functions (which often lead to local minima like "kneeling" or "shaking"). Instead, we use a **"Constraint-Based" Minimalist approach**:

*   **Survival is Priority #1**:
    *   Massive penalties for crashing (`Termination = -200`).
    *   Strict enforcement of upright posture (`Orientation Reward`).
    *   Any contact with non-foot body parts forces an immediate reset.
*   **Simple Drive**:
    *   A single, clean signal for vertical velocity drives the jump.
    *   No complex "phase matching" or "foot trajectory" shaping—we let the physics engine and the torque controller discover the optimal gait naturally.

### 3. Advanced Jump Control Experiments

#### A. Directional & Frequency Control (`go2_jump_control`)
This environment extends the base torque jumping policy to allow explicit control over:
*   **Direction**: `lin_vel_x` (forward/backward) and `lin_vel_y` (left/right).
*   **Jump Frequency**: `commands[3]` controls the jump frequency (Hz).
    *   `freq > 0.1`: Robot jumps at the specified frequency.
    *   `freq < 0.1`: Robot stands still.

**Train:**
```bash
python legged_gym/scripts/train.py --task go2_jump_control
```

#### B. Single-Shot Trigger Jump (`go2_trigger_jump`) [WIP]
An experimental environment designed for precise, "one-shot" jumping control via a trigger signal.
*   **Command Structure**:
    *   `commands[0-2]`: Directional Velocity (x, y, yaw).
    *   `commands[3]`: **Trigger Signal** (0 or 1).
*   **Behavior**:
    *   **Standby**: When Trigger is 0, the robot maintains a stable standing posture.
    *   **Fire**: When Trigger becomes 1, the robot executes **exactly one** full jump cycle.
    *   **Atomic Action**: Once a jump starts, it will complete the full cycle even if the trigger is released mid-air.
*   **Status**: Basic logic implemented. Training stability for the "short pulse" scenario is still being tuned. The robot can jump but may require further reward tuning for perfect single-jump consistency.

**Train:**
```bash
python legged_gym/scripts/train.py --task go2_trigger_jump
```

**Play (Verification):**
```bash
python legged_gym/scripts/play_trigger_jump.py
```
This script automatically verifies the single-jump behavior by sending short trigger pulses every few seconds.

---

## 🚀 Usage

**Train:**
```bash
python scripts/train.py --task=go2_jump_torque_min --headless
```

**Visualize:**
```bash
python scripts/play.py --task=go2_jump_torque_min
```

---

## 📊 Status (Dec 16, 2025)
*   **Experiment Focus**: Pure Torque Control (Position Control comparison experiments have been deprecated).
*   **Current Performance**: Stable vertical jumping with safe, compliant landings.
*   **Codebase**: Cleaned and refactored to remove legacy dependencies and ensure a pure torque learning pipeline.
