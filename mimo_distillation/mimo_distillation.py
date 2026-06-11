"""
18.1 Luyben & Vinante 精馏塔模型 (2×2 MIMO)
Ziegler-Nichols PI 控制 + 单回路/双回路设定值响应仿真

过程传递函数矩阵:
[T17(s)]   [G11(s)  G12(s)] [R(s)]
[T4(s) ] = [G21(s)  G22(s)] [S(s)]

G11 = -2.16 e^{-1.0s} / (8.25s+1)    (T17/R, Loop 1 对角)
G12 =  1.26 e^{-0.3s} / (7.05s+1)    (T17/S, 耦合)
G21 = -2.75 e^{-1.8s} / (8.25s+1)    (T4/R,  耦合)
G22 =  4.28 e^{-0.35s} / (9.0s+1)    (T4/S,  Loop 2 对角)
"""

import numpy as np
import matplotlib.pyplot as plt
from collections import deque
from scipy.optimize import fsolve
import os

# 切换到脚本所在目录，确保图片保存在代码目录下
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
os.chdir(SCRIPT_DIR)

# ============================================================
# 仿真参数
# ============================================================
Ts = 0.01           # 采样周期 (s)
t_end = 100         # 仿真时长 (s)
N = int(t_end / Ts)

# ---------- 过程模型 (FOPDT: [K, theta, tau]) ----------
# G11: T17/R
g11_K, g11_theta, g11_tau = -2.16, 1.0, 8.25
# G12: T17/S
g12_K, g12_theta, g12_tau =  1.26, 0.3, 7.05
# G21: T4/R
g21_K, g21_theta, g21_tau = -2.75, 1.8, 8.25
# G22: T4/S
g22_K, g22_theta, g22_tau =  4.28, 0.35, 9.0

print("=" * 65)
print("Luyben & Vinante Distillation Column — 2×2 MIMO")
print("=" * 65)

# ============================================================
# Part 1: Ziegler-Nichols 连续振荡法整定
# ============================================================
print("\n" + "=" * 65)
print("Ziegler-Nichols Tuning (Continuous Cycling Method)")
print("=" * 65)


def fopdt_phase_eqn(omega, theta, tau):
    """Phase crossover equation: omega*theta + arctan(omega*tau) = pi"""
    return omega * theta + np.arctan(omega * tau) - np.pi


def zn_tune_fopdt(K, theta, tau, label):
    """
    ZN tuning for a FOPDT element.
    Solves: ω_u*θ + arctan(ω_u*τ) = π
    Then: K_u = sqrt(1 + (ω_u*τ)^2) / |K|
          P_u = 2π/ω_u
          K_c = 0.45*K_u, τ_I = P_u/1.2
    """
    # 用 fsolve 数值求解 ω_u
    # 初始猜测：对于纯延迟主导，ω_u ≈ (π/2)/θ
    # 对于一阶惯性主导，ω_u ≈ π/(2*theta)...
    omega_guess = np.pi / (2 * max(theta, 0.01))
    try:
        omega_u = fsolve(lambda w: fopdt_phase_eqn(w, theta, tau), omega_guess)[0]
    except Exception:
        # 如果 fsolve 失败，用手动搜索
        omega_u = omega_guess
        for _ in range(50):
            err = fopdt_phase_eqn(omega_u, theta, tau)
            omega_u += 0.01 * err

    # 验证解
    phase_check = omega_u * theta + np.arctan(omega_u * tau)
    phase_check_deg = np.degrees(phase_check)

    # 临界增益
    Ku = np.sqrt(1 + (omega_u * tau)**2) / abs(K)

    # 临界周期
    Pu = 2 * np.pi / omega_u

    # ZN PI 参数
    Kc = 0.45 * Ku
    tau_I = Pu / 1.2

    print(f"\n{label}: K={K:.3f}, θ={theta:.3f}, τ={tau:.3f}")
    print(f"  ω_u = {omega_u:.4f} rad/s  → phase at crossover = {phase_check_deg:.2f}° (should be 180°)")
    print(f"  K_u = {Ku:.4f}        (critical gain)")
    print(f"  P_u = {Pu:.4f} s      (critical period)")
    print(f"  ZN PI:  K_c = {Kc:.4f},  τ_I = {tau_I:.4f} s")

    return Kc, tau_I, Ku, Pu, omega_u


# Loop 1 (T17—R): use G11
Kc1, tauI1, Ku1, Pu1, wu1 = zn_tune_fopdt(g11_K, g11_theta, g11_tau, "Loop 1 (T17—R) [G11]")
# Loop 2 (T4—S):  use G22
Kc2, tauI2, Ku2, Pu2, wu2 = zn_tune_fopdt(g22_K, g22_theta, g22_tau, "Loop 2 (T4—S)  [G22]")


# ============================================================
# 辅助函数
# ============================================================
def first_order_coeffs(tau):
    """一阶惯性离散化系数: a = e^{-Ts/tau}, b = 1-a"""
    a = np.exp(-Ts / tau)
    b = 1.0 - a
    return a, b


def make_delay(steps):
    """创建纯延迟 FIFO 缓冲区"""
    return deque([0.0] * steps, maxlen=steps)


def push_delay(buf, val):
    """推入值并返回延迟后的输出"""
    out = buf[0]
    buf.append(val)
    return out


def first_order_step(y_prev, u, a, b):
    """一阶惯性一步: y_k = a*y_{k-1} + b*u_k"""
    return a * y_prev + b * u


class FOPDT:
    """一阶惯性+纯延迟 (FOPDT) 子系统"""
    def __init__(self, K, theta, tau):
        self.K = K
        self.theta = theta
        self.tau = tau
        self.a, self.b = first_order_coeffs(tau)
        delay_steps = max(1, int(theta / Ts))
        self.buf = make_delay(delay_steps)
        self.y_inner = 0.0  # 无延迟部分状态

    def step(self, u):
        """一步仿真: u -> y (含延迟)"""
        self.y_inner = first_order_step(self.y_inner, u, self.a, self.b)
        return self.K * push_delay(self.buf, self.y_inner)

    def reset(self):
        self.buf = make_delay(max(1, int(self.theta / Ts)))
        self.y_inner = 0.0


class PIController:
    """PI 控制器 (增量式)"""
    def __init__(self, Kc, tau_I, Ts):
        self.Kc = Kc
        self.tau_I = tau_I
        self.Ts = Ts
        self.e_prev = 0.0
        self.u = 0.0

    def step(self, e):
        """
        增量式: Δu = Kc * [(e_k - e_{k-1}) + (Ts/τ_I) * e_k]
        """
        de = e - self.e_prev
        self.u += self.Kc * (de + (self.Ts / self.tau_I) * e)
        self.e_prev = e
        return self.u

    def reset(self):
        self.e_prev = 0.0
        self.u = 0.0


# ============================================================
# 场景 (a): 单回路设定值变化，另一回路手动
# ============================================================
def simulate_scenario_a1():
    """
    (a1) Loop 1 (T17—R) 设定值阶跃+1, PI自动
         Loop 2 (T4—S) 手动 (控制器输出u2=0)
    耦合仍然存在: G12 会把 S 的影响传到 T17, G21 把 R 传到 T4
    """
    t = np.arange(N) * Ts

    # 设定值
    Ysp1 = np.ones(N)        # Loop 1 单位阶跃
    Ysp2 = np.zeros(N)       # Loop 2 设定值=0

    # 控制器
    pi1 = PIController(Kc1, tauI1, Ts)

    # 过程子系统
    g11 = FOPDT(g11_K, g11_theta, g11_tau)
    g12 = FOPDT(g12_K, g12_theta, g12_tau)
    g21 = FOPDT(g21_K, g21_theta, g21_tau)
    g22 = FOPDT(g22_K, g22_theta, g22_tau)

    # 记录
    T17 = np.zeros(N)
    T4 = np.zeros(N)
    R_sig = np.zeros(N)
    S_sig = np.zeros(N)

    for k in range(N - 1):
        # Loop 1: PI 自动
        e1 = Ysp1[k] - T17[k]
        u1 = pi1.step(e1)

        # Loop 2: 手动 (输出为0, 即稳态值)
        u2 = 0.0

        # 过程计算
        t17_from_r = g11.step(u1)    # G11: R -> T17
        t17_from_s = g12.step(u2)    # G12: S -> T17
        t4_from_r   = g21.step(u1)   # G21: R -> T4
        t4_from_s   = g22.step(u2)   # G22: S -> T4

        T17[k + 1] = t17_from_r + t17_from_s
        T4[k + 1]  = t4_from_r + t4_from_s
        R_sig[k] = u1
        S_sig[k] = u2

    return t, Ysp1, Ysp2, T17, T4, R_sig, S_sig


def simulate_scenario_a2():
    """
    (a2) Loop 1 手动 (u1=0), Loop 2 (T4—S) 设定值阶跃+1, PI自动
    """
    t = np.arange(N) * Ts
    Ysp1 = np.zeros(N)
    Ysp2 = np.ones(N)        # Loop 2 单位阶跃

    pi2 = PIController(Kc2, tauI2, Ts)

    g11 = FOPDT(g11_K, g11_theta, g11_tau)
    g12 = FOPDT(g12_K, g12_theta, g12_tau)
    g21 = FOPDT(g21_K, g21_theta, g21_tau)
    g22 = FOPDT(g22_K, g22_theta, g22_tau)

    T17 = np.zeros(N)
    T4 = np.zeros(N)
    R_sig = np.zeros(N)
    S_sig = np.zeros(N)

    for k in range(N - 1):
        u1 = 0.0  # Loop 1 手动

        e2 = Ysp2[k] - T4[k]
        u2 = pi2.step(e2)

        T17[k + 1] = g11.step(u1) + g12.step(u2)
        T4[k + 1]  = g21.step(u1) + g22.step(u2)
        R_sig[k] = u1
        S_sig[k] = u2

    return t, Ysp1, Ysp2, T17, T4, R_sig, S_sig


# ============================================================
# 场景 (b): 双回路同时自动
# ============================================================
def simulate_scenario_b(Ysp1_val, Ysp2_val, label):
    """
    双回路 PI 自动
    Ysp1_val: Loop 1 设定值阶跃幅值
    Ysp2_val: Loop 2 设定值阶跃幅值
    """
    t = np.arange(N) * Ts
    Ysp1_arr = np.full(N, Ysp1_val)
    Ysp2_arr = np.full(N, Ysp2_val)

    pi1 = PIController(Kc1, tauI1, Ts)
    pi2 = PIController(Kc2, tauI2, Ts)

    g11 = FOPDT(g11_K, g11_theta, g11_tau)
    g12 = FOPDT(g12_K, g12_theta, g12_tau)
    g21 = FOPDT(g21_K, g21_theta, g21_tau)
    g22 = FOPDT(g22_K, g22_theta, g22_tau)

    T17 = np.zeros(N)
    T4 = np.zeros(N)
    R_sig = np.zeros(N)
    S_sig = np.zeros(N)

    for k in range(N - 1):
        e1 = Ysp1_arr[k] - T17[k]
        e2 = Ysp2_arr[k] - T4[k]

        u1 = pi1.step(e1)
        u2 = pi2.step(e2)

        T17[k + 1] = g11.step(u1) + g12.step(u2)
        T4[k + 1]  = g21.step(u1) + g22.step(u2)
        R_sig[k] = u1
        S_sig[k] = u2

    return t, Ysp1_arr, Ysp2_arr, T17, T4, R_sig, S_sig


# ============================================================
# 绘图设置
# ============================================================
plt.rcParams.update({
    'font.size': 11,
    'axes.titlesize': 13,
    'axes.labelsize': 12,
    'legend.fontsize': 9,
    'figure.dpi': 150,
    'savefig.dpi': 150,
    'savefig.bbox': 'tight',
})


def plot_scenario(t, Ysp1, Ysp2, T17, T4, R, S, title, filename,
                  show_sp1=True, show_sp2=True):
    """绘制四子图：T17, T4, R, S"""
    fig, axes = plt.subplots(4, 1, figsize=(12, 12), sharex=True)

    colors = plt.cm.tab10.colors

    # --- T17 ---
    ax = axes[0]
    ax.plot(t, T17, color=colors[0], linewidth=1.8, label=r'$T_{17}$ (top temp)')
    if show_sp1 and np.any(Ysp1 != 0):
        ax.plot(t, Ysp1, 'k--', linewidth=1.2, alpha=0.6, label=r'$Y_{sp1}$')
    ax.set_ylabel(r'$T_{17}$')
    ax.set_title(f'{title} — $T_{{17}}$ (Loop 1 CV)')
    ax.legend(loc='best')
    ax.grid(True, alpha=0.3)
    ax.set_xlim([0, t_end])

    # --- T4 ---
    ax = axes[1]
    ax.plot(t, T4, color=colors[1], linewidth=1.8, label=r'$T_{4}$ (bottom temp)')
    if show_sp2 and np.any(Ysp2 != 0):
        ax.plot(t, Ysp2, 'k--', linewidth=1.2, alpha=0.6, label=r'$Y_{sp2}$')
    ax.set_ylabel(r'$T_{4}$')
    ax.set_title(f'{title} — $T_{{4}}$ (Loop 2 CV)')
    ax.legend(loc='best')
    ax.grid(True, alpha=0.3)
    ax.set_xlim([0, t_end])

    # --- R (回流比) ---
    ax = axes[2]
    ax.plot(t, R, color=colors[2], linewidth=1.5, label=r'$R$ (reflux, MV1)')
    ax.set_ylabel(r'$R$')
    ax.set_title(f'{title} — Manipulated Variables')
    ax.legend(loc='best')
    ax.grid(True, alpha=0.3)
    ax.set_xlim([0, t_end])

    # --- S (蒸汽流量) ---
    ax = axes[3]
    ax.plot(t, S, color=colors[3], linewidth=1.5, label=r'$S$ (steam, MV2)')
    ax.set_xlabel('Time (s)')
    ax.set_ylabel(r'$S$')
    ax.legend(loc='best')
    ax.grid(True, alpha=0.3)
    ax.set_xlim([0, t_end])

    plt.tight_layout()
    plt.savefig(filename)
    print(f"Saved: {filename}")
    plt.close(fig)


# ============================================================
# 运行所有场景仿真
# ============================================================
print("\n" + "=" * 65)
print("Running Simulations...")
print("=" * 65)

# --- 场景 (a1) ---
print("\n>>> Scenario (a1): Loop 1 auto, Loop 2 manual")
t_a1, Ysp1_a1, Ysp2_a1, T17_a1, T4_a1, R_a1, S_a1 = simulate_scenario_a1()
plot_scenario(t_a1, Ysp1_a1, Ysp2_a1, T17_a1, T4_a1, R_a1, S_a1,
              'Scenario (a1): Loop 1 (T₁₇–R) Auto, Loop 2 Manual',
              'mimo_scenario_a1.png')

# --- 场景 (a2) ---
print("\n>>> Scenario (a2): Loop 1 manual, Loop 2 auto")
t_a2, Ysp1_a2, Ysp2_a2, T17_a2, T4_a2, R_a2, S_a2 = simulate_scenario_a2()
plot_scenario(t_a2, Ysp1_a2, Ysp2_a2, T17_a2, T4_a2, R_a2, S_a2,
              'Scenario (a2): Loop 1 Manual, Loop 2 (T₄–S) Auto',
              'mimo_scenario_a2.png')

# --- 场景 (b1) ---
print("\n>>> Scenario (b1): Both auto, Loop 1 SP=+1, Loop 2 SP=0")
t_b1, Ysp1_b1, Ysp2_b1, T17_b1, T4_b1, R_b1, S_b1 = simulate_scenario_b(1.0, 0.0, 'b1')
plot_scenario(t_b1, Ysp1_b1, Ysp2_b1, T17_b1, T4_b1, R_b1, S_b1,
              'Scenario (b1): Both Auto, SP₁=+1, SP₂=0',
              'mimo_scenario_b1.png')

# --- 场景 (b2) ---
print("\n>>> Scenario (b2): Both auto, Loop 1 SP=0, Loop 2 SP=+1")
t_b2, Ysp1_b2, Ysp2_b2, T17_b2, T4_b2, R_b2, S_b2 = simulate_scenario_b(0.0, 1.0, 'b2')
plot_scenario(t_b2, Ysp1_b2, Ysp2_b2, T17_b2, T4_b2, R_b2, S_b2,
              'Scenario (b2): Both Auto, SP₁=0, SP₂=+1',
              'mimo_scenario_b2.png')

# --- 场景 (b3) ---
print("\n>>> Scenario (b3): Both auto, both SP=+1")
t_b3, Ysp1_b3, Ysp2_b3, T17_b3, T4_b3, R_b3, S_b3 = simulate_scenario_b(1.0, 1.0, 'b3')
plot_scenario(t_b3, Ysp1_b3, Ysp2_b3, T17_b3, T4_b3, R_b3, S_b3,
              'Scenario (b3): Both Auto, SP₁=+1, SP₂=+1',
              'mimo_scenario_b3.png')


# ============================================================
# 综合对比图 (Cross-scenario Comparison)
# ============================================================
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# 左上: T17 across scenarios (a1, b1, b3)
ax = axes[0, 0]
ax.plot(t_a1, T17_a1, linewidth=1.8, label='(a1) Loop 1 auto, L2 manual')
ax.plot(t_b1, T17_b1, '--', linewidth=1.8, label='(b1) Both auto, SP₁=+1')
ax.plot(t_b3, T17_b3, '-.', linewidth=1.8, label='(b3) Both auto, both SP=+1')
ax.axhline(y=1.0, color='k', linestyle=':', alpha=0.5)
ax.set_xlabel('Time (s)')
ax.set_ylabel(r'$T_{17}$')
ax.set_title(r'$T_{17}$ — Set-point Tracking Comparison')
ax.legend(fontsize=8)
ax.set_xlim([0, t_end])
ax.grid(True, alpha=0.3)

# 右上: T4 across scenarios (a2, b2, b3)
ax = axes[0, 1]
ax.plot(t_a2, T4_a2, linewidth=1.8, label='(a2) L1 manual, L2 auto')
ax.plot(t_b2, T4_b2, '--', linewidth=1.8, label='(b2) Both auto, SP₂=+1')
ax.plot(t_b3, T4_b3, '-.', linewidth=1.8, label='(b3) Both auto, both SP=+1')
ax.axhline(y=1.0, color='k', linestyle=':', alpha=0.5)
ax.set_xlabel('Time (s)')
ax.set_ylabel(r'$T_{4}$')
ax.set_title(r'$T_{4}$ — Set-point Tracking Comparison')
ax.legend(fontsize=8)
ax.set_xlim([0, t_end])
ax.grid(True, alpha=0.3)

# 左下: Coupling effect on T4 when Loop 1 steps
ax = axes[1, 0]
ax.plot(t_a1, T4_a1, linewidth=1.8, color='tab:orange',
        label='(a1) T₄ when T₁₇ step (L2 manual)')
ax.plot(t_b1, T4_b1, '--', linewidth=1.8, color='tab:red',
        label='(b1) T₄ when T₁₇ step (L2 auto)')
ax.axhline(y=0, color='k', linestyle='-', alpha=0.3)
ax.set_xlabel('Time (s)')
ax.set_ylabel(r'$T_{4}$')
ax.set_title(r'Coupling: $T_{4}$ response to Loop 1 set-point change')
ax.legend(fontsize=8)
ax.set_xlim([0, t_end])
ax.grid(True, alpha=0.3)

# 右下: Coupling effect on T17 when Loop 2 steps
ax = axes[1, 1]
ax.plot(t_a2, T17_a2, linewidth=1.8, color='tab:blue',
        label='(a2) T₁₇ when T₄ step (L1 manual)')
ax.plot(t_b2, T17_b2, '--', linewidth=1.8, color='tab:purple',
        label='(b2) T₁₇ when T₄ step (L1 auto)')
ax.axhline(y=0, color='k', linestyle='-', alpha=0.3)
ax.set_xlabel('Time (s)')
ax.set_ylabel(r'$T_{17}$')
ax.set_title(r'Coupling: $T_{17}$ response to Loop 2 set-point change')
ax.legend(fontsize=8)
ax.set_xlim([0, t_end])
ax.grid(True, alpha=0.3)

fig.suptitle(
    f'MIMO Distillation Column — Comparison Across Scenarios\n'
    f'ZN PI: Loop 1 $K_c$={Kc1:.3f}, $\\tau_I$={tauI1:.2f}s | '
    f'Loop 2 $K_c$={Kc2:.3f}, $\\tau_I$={tauI2:.2f}s',
    fontsize=13, y=1.01
)
plt.tight_layout()
plt.savefig('mimo_cross_comparison.png')
print("\nSaved: mimo_cross_comparison.png")
plt.close(fig)


# ============================================================
# 总结输出
# ============================================================
print("\n" + "=" * 65)
print("Simulation Complete!")
print("=" * 65)
print(f"\nController Parameters (Ziegler-Nichols PI):")
print(f"  Loop 1 (T₁₇—R):  K_c = {Kc1:.4f},  τ_I = {tauI1:.4f} s")
print(f"    (Critical: K_u = {Ku1:.4f}, P_u = {Pu1:.4f} s, ω_u = {wu1:.4f})")
print(f"  Loop 2 (T₄—S):   K_c = {Kc2:.4f},  τ_I = {tauI2:.4f} s")
print(f"    (Critical: K_u = {Ku2:.4f}, P_u = {Pu2:.4f} s, ω_u = {wu2:.4f})")
print(f"\nOutput files in: {os.getcwd()}")
