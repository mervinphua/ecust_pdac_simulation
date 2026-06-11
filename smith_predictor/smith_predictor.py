"""
Smith Predictor 时延补偿器仿真
Homework 16.7: Design a time-delay compensator (Smith predictor)

Process:       G_p(s) = e^(-θs) / (5s+1)
Actuator:      G_v = 1
Measurement:   G_m = 1
Disturbance:   G_d(s) = G_p(s)
Controller:    G_c = K_c = 1 (P control)
Delay:         θ = 1
"""

import numpy as np
import matplotlib.pyplot as plt
from collections import deque
import os

# 切换到脚本所在目录
os.chdir(os.path.dirname(os.path.abspath(__file__)))

# ============================================================
# 仿真参数
# ============================================================
Kc = 1          # 比例增益
tau = 5         # 过程时间常数
theta = 1       # 纯延迟时间
Ts = 0.01       # 采样周期 (s)
t_end = 30      # 仿真时长 (s)

# 离散化参数
N = int(t_end / Ts)                # 总步数
delay_steps = int(theta / Ts)      # 延迟步数 (100 steps)

# 一阶惯性环节离散化: y_k = a * y_{k-1} + b * u_k
a = np.exp(-Ts / tau)              # a = e^(-Ts/τ)
b = 1 - a                          # b = 1 - a

print(f"Sampling period Ts = {Ts:.3f}")
print(f"Total steps N = {N}")
print(f"Delay steps = {delay_steps}")
print(f"Discrete params: a = {a:.6f}, b = {b:.6f}")

# 时间向量
t = np.arange(N) * Ts


# ============================================================
# 辅助函数
# ============================================================
def first_order_step(y_prev, u, a, b):
    """一阶惯性环节一步更新 y_k = a * y_{k-1} + b * u_k"""
    return a * y_prev + b * u


def delay_buffer():
    """创建纯延迟用的 FIFO 缓冲区"""
    buf = deque([0.0] * delay_steps, maxlen=delay_steps)
    return buf


def push_delay(buf, val):
    """将当前值推入延迟缓冲区，返回延迟后的输出"""
    delayed = buf[0]  # 最旧的值 = 延迟 θ 后的输出
    buf.append(val)   # 新值入队
    return delayed


# ============================================================
# 场景 1: 单位阶跃设定值变化 (Set-point change)
# ============================================================
def simulate_setpoint():
    """
    Smith Predictor 结构:
    
    Y_sp ──→[+]──→[G_c=Kc]──→[G_v=1]──→[G_p(s)]──────────────→ Y_actual
               ↑   [-]                          ↑
               │     └── y_comp ←──[y_model* - y_model]──┘
               │                    (Smith Predictor内部)
    """
    # 状态变量初始化
    Y_sp = np.ones(N)       # 单位阶跃设定值

    # --- 实际过程 ---
    y_actual = np.zeros(N)
    y_actual_inner = 0.0    # 实际过程无延迟部分的状态
    buf_actual = delay_buffer()

    # --- Smith Predictor 内部模型 ---
    y_model_star = np.zeros(N)  # 模型无延迟输出 G_p*(s)
    y_model = np.zeros(N)       # 模型带延迟输出
    y_model_inner = 0.0
    buf_model = delay_buffer()

    # --- 控制量 ---
    u = np.zeros(N)
    y_comp = np.zeros(N)    # 补偿信号

    for k in range(N - 1):
        # Smith Predictor 补偿信号: y_comp = y_model* - y_model
        # (这一步用上一步的值, 因为 y_model 已经包含了延迟)
        y_comp[k] = y_model_star[k] - y_model[k]

        # 控制器输入误差: e = Y_sp - (y_actual - y_comp)
        # Smith Predictor 把 y_comp 加回去, 等效于控制器看到无延迟对象
        e = Y_sp[k] - (y_actual[k] - y_comp[k])

        # P 控制器
        u[k] = Kc * e

        # --- 实际过程: G_p(s) = e^{-θs} / (τs+1) ---
        # Step 1: 一阶惯性环节
        y_actual_inner = first_order_step(y_actual_inner, u[k], a, b)
        # Step 2: 纯延迟
        y_actual[k + 1] = push_delay(buf_actual, y_actual_inner)

        # --- Smith Predictor 内部模型 ---
        # 无延迟模型输出 G_p*(s) = 1/(τs+1)
        y_model_inner = first_order_step(y_model_inner, u[k], a, b)
        y_model_star[k + 1] = y_model_inner
        # 带延迟模型输出
        y_model[k + 1] = push_delay(buf_model, y_model_inner)

    return t, Y_sp, y_actual, y_model, y_comp, u


# ============================================================
# 场景 1b: 无 Smith Predictor 的普通反馈 (用于对比)
# ============================================================
def simulate_setpoint_no_sp():
    """普通反馈: 控制器直接基于 Y_sp - Y_actual 计算"""
    Y_sp = np.ones(N)
    y_actual = np.zeros(N)
    y_inner = 0.0
    buf = delay_buffer()
    u = np.zeros(N)

    for k in range(N - 1):
        e = Y_sp[k] - y_actual[k]
        u[k] = Kc * e
        y_inner = first_order_step(y_inner, u[k], a, b)
        y_actual[k + 1] = push_delay(buf, y_inner)

    return t, Y_sp, y_actual, u


# ============================================================
# 场景 2: 单位阶跃扰动变化 (Disturbance change)
# ============================================================
def simulate_disturbance():
    """
    扰动 D(s) 加到过程输出端, G_d(s) = G_p(s)
    
    Y_sp=0, D(t)=1(t) 从 t=0 开始作用
    扰动信号经过 G_d(s) = e^{-θs}/(τs+1) 后叠加到 Y_actual
    """
    Y_sp = np.zeros(N)          # 设定值为 0
    D = np.ones(N)              # 单位阶跃扰动

    # --- 实际过程 ---
    y_actual = np.zeros(N)
    y_actual_inner = 0.0
    buf_actual = delay_buffer()

    # --- 扰动通道 G_d(s) = e^{-θs}/(τs+1) ---
    y_dist = np.zeros(N)
    y_dist_inner = 0.0
    buf_dist = delay_buffer()

    # --- Smith Predictor 内部模型 ---
    y_model_star = np.zeros(N)
    y_model = np.zeros(N)
    y_model_inner = 0.0
    buf_model = delay_buffer()

    # --- 控制量 ---
    u = np.zeros(N)
    y_comp = np.zeros(N)

    for k in range(N - 1):
        # Smith Predictor 补偿
        y_comp[k] = y_model_star[k] - y_model[k]

        # 控制器输入误差 (Y_sp=0)
        e = Y_sp[k] - (y_actual[k] - y_comp[k])

        # P 控制器
        u[k] = Kc * e

        # --- 实际过程 (由控制量驱动) + 扰动 ---
        y_actual_inner = first_order_step(y_actual_inner, u[k], a, b)
        # 扰动的无延迟部分 (输入=扰动信号 D)
        y_dist_inner = first_order_step(y_dist_inner, D[k], a, b)

        # 过程输出的延迟部分 + 扰动输出的延迟部分
        actual_delayed = push_delay(buf_actual, y_actual_inner)
        dist_delayed = push_delay(buf_dist, y_dist_inner)

        y_actual[k + 1] = actual_delayed + dist_delayed

        # --- Smith Predictor 内部模型 (只含过程, 不含扰动) ---
        y_model_inner = first_order_step(y_model_inner, u[k], a, b)
        y_model_star[k + 1] = y_model_inner
        y_model[k + 1] = push_delay(buf_model, y_model_inner)

    return t, Y_sp, D, y_actual, y_model, y_comp, u


# ============================================================
# 场景 2b: 无 Smith Predictor 的扰动响应 (对比)
# ============================================================
def simulate_disturbance_no_sp():
    """普通反馈扰动抑制"""
    Y_sp = np.zeros(N)
    D = np.ones(N)

    y_actual = np.zeros(N)
    y_inner = 0.0
    buf = delay_buffer()

    y_dist = np.zeros(N)
    y_dist_inner = 0.0
    buf_dist = delay_buffer()

    u = np.zeros(N)

    for k in range(N - 1):
        e = Y_sp[k] - y_actual[k]
        u[k] = Kc * e

        y_inner = first_order_step(y_inner, u[k], a, b)
        y_dist_inner = first_order_step(y_dist_inner, D[k], a, b)

        actual_delayed = push_delay(buf, y_inner)
        dist_delayed = push_delay(buf_dist, y_dist_inner)

        y_actual[k + 1] = actual_delayed + dist_delayed

    return t, Y_sp, D, y_actual, u


# ============================================================
# 运行仿真
# ============================================================
print("\n=== Running Set-point response simulation ===")
t_sp, Y_sp, y_sp, y_model_sp, y_comp_sp, u_sp = simulate_setpoint()
t_sp2, Y_sp2, y_sp_nosp, u_sp_nosp = simulate_setpoint_no_sp()

print("=== Running Disturbance response simulation ===")
t_dist, Y_sp_d, D_dist, y_dist, y_model_dist, y_comp_dist, u_dist = simulate_disturbance()
t_dist2, Y_sp_d2, D_dist2, y_dist_nosp, u_dist_nosp = simulate_disturbance_no_sp()

# ============================================================
# 绘图
# ============================================================
plt.rcParams.update({
    'font.size': 12,
    'axes.titlesize': 14,
    'axes.labelsize': 13,
    'legend.fontsize': 11,
    'figure.dpi': 150,
    'savefig.dpi': 150,
    'savefig.bbox': 'tight',
})

# ---------- 图 1: 设定值响应 ----------
fig, axes = plt.subplots(2, 1, figsize=(12, 9))

ax = axes[0]
ax.plot(t_sp, Y_sp, 'k--', linewidth=1.5, label=r'Set-point $Y_{sp}(t)$')
ax.plot(t_sp, y_sp, 'b-', linewidth=2, label=r'With Smith Predictor')
ax.plot(t_sp, y_sp_nosp, 'r--', linewidth=1.8, label=r'Without Smith Predictor')
ax.axvline(x=theta, color='gray', linestyle=':', alpha=0.7, label=f'θ = {theta}')
ax.set_xlabel('Time (s)')
ax.set_ylabel(r'Output $Y(t)$')
ax.set_title(f'Set-point Response (Unit Step)\n$K_c={Kc}, \\tau={tau}, \\theta={theta}$')
ax.legend(loc='lower right')
ax.set_xlim([0, t_end])
ax.grid(True, alpha=0.3)

ax = axes[1]
ax.plot(t_sp, u_sp, 'b-', linewidth=1.5, label='Control signal (with SP)')
ax.plot(t_sp, u_sp_nosp, 'r--', linewidth=1.5, label='Control signal (without SP)')
ax.set_xlabel('Time (s)')
ax.set_ylabel(r'Control $u(t)$')
ax.set_title('Control Signal Comparison')
ax.legend()
ax.set_xlim([0, t_end])
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('smith_predictor_setpoint.png')
print("\nSaved: smith_predictor_setpoint.png")

# ---------- 图 2: 扰动响应 ----------
fig, axes = plt.subplots(2, 1, figsize=(12, 9))

ax = axes[0]
ax.plot(t_dist, D_dist, 'g--', linewidth=1.5, alpha=0.7, label=r'Disturbance $D(t)=1(t)$')
ax.plot(t_dist, y_dist, 'b-', linewidth=2, label=r'With Smith Predictor')
ax.plot(t_dist, y_dist_nosp, 'r--', linewidth=1.8, label=r'Without Smith Predictor')
ax.axhline(y=0, color='k', linestyle='-', alpha=0.3)
ax.axvline(x=theta, color='gray', linestyle=':', alpha=0.7, label=f'θ = {theta}')
ax.set_xlabel('Time (s)')
ax.set_ylabel(r'Output $Y(t)$')
ax.set_title(f'Disturbance Rejection (Unit Step)\n$K_c={Kc}, \\tau={tau}, \\theta={theta}$')
ax.legend(loc='upper right')
ax.set_xlim([0, t_end])
ax.grid(True, alpha=0.3)

ax = axes[1]
ax.plot(t_dist, u_dist, 'b-', linewidth=1.5, label='Control signal (with SP)')
ax.plot(t_dist, u_dist_nosp, 'r--', linewidth=1.5, label='Control signal (without SP)')
ax.set_xlabel('Time (s)')
ax.set_ylabel(r'Control $u(t)$')
ax.set_title('Control Signal Comparison')
ax.legend()
ax.set_xlim([0, t_end])
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('smith_predictor_disturbance.png')
print("Saved: smith_predictor_disturbance.png")

# ---------- 图 3: 综合对比大图 ----------
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# 设定值响应 - 输出
ax = axes[0, 0]
ax.plot(t_sp, Y_sp, 'k--', linewidth=1.5, label=r'$Y_{sp}$')
ax.plot(t_sp, y_sp, 'b-', linewidth=2, label='With Smith Predictor')
ax.plot(t_sp, y_sp_nosp, 'r--', linewidth=1.8, label='Without Smith Predictor')
ax.axvline(x=theta, color='gray', linestyle=':', alpha=0.7)
ax.set_xlabel('Time (s)')
ax.set_ylabel(r'$Y(t)$')
ax.set_title('Set-point Response — Output')
ax.legend(loc='lower right')
ax.set_xlim([0, t_end])
ax.grid(True, alpha=0.3)

# 设定值响应 - 控制量
ax = axes[0, 1]
ax.plot(t_sp, u_sp, 'b-', linewidth=1.5, label='With Smith Predictor')
ax.plot(t_sp, u_sp_nosp, 'r--', linewidth=1.5, label='Without Smith Predictor')
ax.set_xlabel('Time (s)')
ax.set_ylabel(r'$u(t)$')
ax.set_title('Set-point Response — Control Signal')
ax.legend()
ax.set_xlim([0, t_end])
ax.grid(True, alpha=0.3)

# 扰动响应 - 输出
ax = axes[1, 0]
ax.plot(t_dist, D_dist, 'g--', linewidth=1.5, alpha=0.7, label=r'$D(t)$')
ax.plot(t_dist, y_dist, 'b-', linewidth=2, label='With Smith Predictor')
ax.plot(t_dist, y_dist_nosp, 'r--', linewidth=1.8, label='Without Smith Predictor')
ax.axhline(y=0, color='k', linestyle='-', alpha=0.3)
ax.axvline(x=theta, color='gray', linestyle=':', alpha=0.7)
ax.set_xlabel('Time (s)')
ax.set_ylabel(r'$Y(t)$')
ax.set_title('Disturbance Rejection — Output')
ax.legend(loc='upper right')
ax.set_xlim([0, t_end])
ax.grid(True, alpha=0.3)

# 扰动响应 - 控制量
ax = axes[1, 1]
ax.plot(t_dist, u_dist, 'b-', linewidth=1.5, label='With Smith Predictor')
ax.plot(t_dist, u_dist_nosp, 'r--', linewidth=1.5, label='Without Smith Predictor')
ax.set_xlabel('Time (s)')
ax.set_ylabel(r'$u(t)$')
ax.set_title('Disturbance Rejection — Control Signal')
ax.legend()
ax.set_xlim([0, t_end])
ax.grid(True, alpha=0.3)

fig.suptitle(f'Smith Predictor — Comprehensive Comparison ($K_c={Kc}, \\tau={tau}, \\theta={theta}$)',
             fontsize=16, y=1.01)
plt.tight_layout()
plt.savefig('smith_predictor_comparison.png')
print("Saved: smith_predictor_comparison.png")

plt.show()
print("\n=== Simulation complete! ===")
