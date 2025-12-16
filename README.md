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
