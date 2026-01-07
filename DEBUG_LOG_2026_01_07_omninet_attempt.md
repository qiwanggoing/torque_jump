# 开发日志 - 2026年1月7日：OmniNet 架构复现与双优化器尝试

## 1. 目标
复现 OmniNet 的核心架构，具体包括：
1.  **历史观测 (Frame Stacking)**：引入过去 20 帧的历史信息 ($H=20$)。
2.  **高度估计器 (Height Estimator)**：利用历史信息训练一个独立的高度估计网络。
3.  **超越性能**：在 `go2_jump_torque_min` 任务上，利用历史信息提升跳跃的物理边界。

## 2. 实施方案 (Architecture Implementation)

为了解决“估计器 Loss 爆炸影响策略”的问题，我们实施了一套复杂的**隔离架构**：

### A. 观测空间分层 (Observation Splitting)
*   **Actor (策略)**: 只看当前帧 (62维)。
    *   *目的*: 保持输入简单，快速收敛，兼容旧有的成功经验。
*   **Critic (价值)**: 只看当前帧 (62维)。
    *   *修改*: 在 `ActorCritic.evaluate` 中对输入进行了切片 (`[:, :62]`)。
*   **Estimator (估计)**: 看全量历史 (1302维 = 62 * 21)。
    *   *目的*: 利用时序信息准确预测高度。

### B. 双优化器隔离 (Dual Optimizers)
为了彻底防止估计器的梯度污染策略网络，我们修改了 `PPO` 算法：
*   **Main Optimizer**: 仅优化 Actor, Critic 和 Std。
*   **Estimator Optimizer**: 仅优化 Height Estimator。
*   **物理隔离**: 在 `update` 步骤中，两个优化器分别执行 `backward()` 和 `step()`，梯度完全不互通。

### C. 鲁棒性增强
*   **异常过滤**: 在计算 Estimation Loss 时，自动忽略 `abs(true_height) > 10.0` 的物理引擎异常数据。

## 3. 实验结果：策略坍缩 (Policy Collapse)

尽管架构设计理论上很完美，但在 `go2_jump_torque_min` 上训练时出现了典型的问题：

*   **现象**:
    *   **0-500 Iter**: 机器人学会了跳跃，Reward 正常上升。
    *   **500-800 Iter**: 机器人逐渐倾向于“站立不动”。
    *   **800+ Iter**: 策略彻底坍缩，机器人选择“踮起脚尖直立”，完全放弃跳跃。
*   **原因分析**:
    1.  **惩罚过重**: `termination` (摔倒) 惩罚高达 `-200.0`。
    2.  **风险规避**: 在 Critic 尚未完全收敛（或因架构变动导致价值估计不准）时，机器人发现“跳跃”风险太大（容易摔倒），而“踮脚站立”既能拿 `base_height` 奖励又绝对安全。
    3.  **价值函数滞后**: Critic 被限制在 62 维，可能在处理高动态动作（需要预判）时不如 Estimator 准确，导致 Advantage 计算偏差。

## 4. 结论与后续
*   已回退代码至 `3b660ee` (Trigger Jump 版本)，即 1月6日的稳定状态。
*   今天的尝试代码已备份在分支 `backup_2026_01_07_attempt` 中。