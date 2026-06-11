"""
15.5 前馈-反馈控制系统设计 (Feedforward-Feedback Control)

过程模型:
  G_p(s) = 1/(s+1)              (被控过程)
  G_d(s) = 2/[(s+1)(5s+1)]      (扰动通道)
  G_v = 1, G_m = 1, G_t = 1     (执行器/变送器)

控制结构 (Fig 15.11):
  P = P_FF + P_FB  →  G_v  →  G_p  →  Y
  P_FF = G_f * D_m,  D_m = G_t * D
  P_FB = G_c * (Y_sp - G_m * Y)

============================================================
PART A: 解析推导
============================================================
"""

import numpy as np
import matplotlib.pyplot as plt
from collections import deque
import os

# 切换到脚本目录
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
os.chdir(SCRIPT_DIR)

# ============================================================
# 仿真参数
# ============================================================
Ts = 0.01          # 采样周期
t_end = 30         # 仿真时长 (s)
N = int(t_end / Ts)
t = np.arange(N) * Ts

D_step = 1.0       # 单位阶跃扰动

print("=" * 70)
print("15.5 前馈-反馈控制系统设计")
print("=" * 70)


# ============================================================
# (a) 稳态前馈控制器 (Steady-state FF)
# ============================================================
print("\n" + "-" * 50)
print("(a) 稳态前馈控制器 (Steady-State Feedforward)")
print("-" * 50)
print("""
G_f(0) = - G_d(0) / [G_p(0) * G_v(0) * G_t(0)]

G_p(0) = 1/(0+1) = 1
G_d(0) = 2/[(0+1)(5*0+1)] = 2
G_v(0) = 1, G_t(0) = 1

G_f(0) = -2 / (1 * 1 * 1) = -2
""")
Kf_ss = -2.0
print(f"  => G_f^ss = K_f = {Kf_ss:.1f}  (纯增益常数)")


# ============================================================
# (b) 动态前馈控制器 (Dynamic FF)
# ============================================================
print("\n" + "-" * 50)
print("(b) 动态前馈控制器 (Dynamic Feedforward)")
print("-" * 50)
print("""
理想动态补偿条件:
  G_f(s) = - G_d(s) / [G_p(s) * G_v(s) * G_t(s)]

代入:
  G_f(s) = - [2/((s+1)(5s+1))] / [(1/(s+1)) * 1 * 1]
         = - [2/((s+1)(5s+1))] * (s+1)
         = - 2/(5s+1)

分析可实现性:
  分子阶次 = 0, 分母阶次 = 1
  分子阶次 ≤ 分母阶次 → 物理可实现！(严格正则)
  这是一个一阶惯性环节 (first-order lag)，无需近似。
""")
print("  => G_f(s) = -2 / (5s + 1)   [一阶惯性，物理可实现]")


# ============================================================
# (c) IMC 反馈控制器设计
# ============================================================
print("\n" + "-" * 50)
print("(c) IMC 反馈控制器设计 (τ_c = 2)")
print("-" * 50)
print("""
Step 1: 分解 G_p(s)
  G_p(s) = 1/(s+1), 无 RHP 零点、无时延
  → G_p+ = 1, G_p- = 1/(s+1)

Step 2: IMC 控制器
  G_IMC(s) = (1/G_p-) * 1/(τ_c*s + 1)
           = (s+1) / (2s+1)

Step 3: 等效反馈控制器
  G_c(s) = G_IMC / (1 - G_IMC * G_p)
         = [(s+1)/(2s+1)] / [1 - (s+1)/(2s+1) * 1/(s+1)]
         = [(s+1)/(2s+1)] / [1 - 1/(2s+1)]
         = [(s+1)/(2s+1)] / [2s/(2s+1)]
         = (s+1) / (2s)
         = 1/2 * (1 + 1/s)
         = 0.5 + 0.5/s

Step 4: 识别控制器类型
  G_c(s) = K_c * (1 + 1/(τ_I * s))
  其中: K_c = 0.5, τ_I = 1
  → 这是一个 PI 控制器！
""")
Kc_imc = 0.5
tauI_imc = 1.0
print(f"  => PI: K_c = {Kc_imc}, τ_I = {tauI_imc}")


# ============================================================
# 离散化参数
# ============================================================
print("\n" + "-" * 50)
print("离散化参数")
print("-" * 50)

# G_p(s) = 1/(s+1): τ=1
a_p = np.exp(-Ts / 1.0)
b_p = 1.0 - a_p

# G_d(s) = 2/[(s+1)(5s+1)]: two first-order stages
a_d1 = np.exp(-Ts / 1.0)   # τ=1
b_d1 = 1.0 - a_d1
a_d2 = np.exp(-Ts / 5.0)   # τ=5
b_d2 = 1.0 - a_d2

# Dynamic FF: G_f(s) = -2/(5s+1): τ=5
a_ff = np.exp(-Ts / 5.0)
b_ff = 1.0 - a_ff

print(f"  G_p (τ=1):    a={a_p:.6f}, b={b_p:.6f}")
print(f"  G_d stage1 (τ=1): a={a_d1:.6f}, b={b_d1:.6f}")
print(f"  G_d stage2 (τ=5): a={a_d2:.6f}, b={b_d2:.6f}")
print(f"  G_f dyn (τ=5):   a={a_ff:.6f}, b={b_ff:.6f}")
print(f"  PI: K_c={Kc_imc}, τ_I={tauI_imc}, Ts/τ_I={Ts/tauI_imc:.4f}")


# ============================================================
# 辅助函数
# ============================================================
def first_order_step(y_prev, u, a, b):
    """一阶惯性: y_{k+1} = a*y_k + b*u_k"""
    return a * y_prev + b * u


class PIController:
    """增量式 PI 控制器"""
    def __init__(self, Kc, tau_I, Ts):
        self.Kc = Kc
        self.tau_I = tau_I
        self.Ts = Ts
        self.reset()

    def reset(self):
        self.e_prev = 0.0
        self.u = 0.0

    def step(self, e):
        """Δu = Kc * [de + (Ts/τ_I) * e]"""
        de = e - self.e_prev
        self.u += self.Kc * (de + (self.Ts / self.tau_I) * e)
        self.e_prev = e
        return self.u


# ============================================================
# (d) 纯前馈控制仿真 (Pure FF, no feedback)
# ============================================================
def simulate_pure_ff(ff_type='none'):
    """
    ff_type: 'none' | 'steady' | 'dynamic'
    反馈控制器 G_c = 0 (开环)
    扰动 D(t) = 1(t) 从 t=0 开始
    """
    D = np.full(N, D_step)

    # 状态变量
    y = np.zeros(N)                     # 输出 Y(t)

    # 过程 G_p(s) = 1/(s+1): 输入 = P_FF (因为无反馈，P_FB=0)
    y_p_inner = 0.0

    # 扰动通道 G_d(s) = 2/[(s+1)(5s+1)]
    y_d1_inner = 0.0   # stage 1 (τ=1)
    y_d2_inner = 0.0   # stage 2 (τ=5)

    # 动态 FF 状态
    y_ff_inner = 0.0

    # 记录
    P_FF_arr = np.zeros(N)
    P_arr = np.zeros(N)

    for k in range(N - 1):
        # 前馈补偿计算
        if ff_type == 'steady':
            P_FF = Kf_ss * D[k]           # P_FF = -2 * D
        elif ff_type == 'dynamic':
            y_ff_inner = first_order_step(y_ff_inner, D[k], a_ff, b_ff)
            P_FF = -2.0 * y_ff_inner       # G_f(s) = -2/(5s+1)
        else:  # none
            P_FF = 0.0

        P = P_FF  # 无反馈，P = P_FF only

        # 过程通道: P → G_p(s) → Y
        y_p_inner = first_order_step(y_p_inner, P, a_p, b_p)

        # 扰动通道: D → G_d(s) → 叠加到 Y
        y_d1_inner = first_order_step(y_d1_inner, D[k], a_d1, b_d1)
        y_d2_inner = first_order_step(y_d2_inner, y_d1_inner, a_d2, b_d2)

        y[k + 1] = y_p_inner + 2.0 * y_d2_inner  # 注意 Gd 增益=2, 已包含

        P_FF_arr[k] = P_FF
        P_arr[k] = P

    return t, D, y, P_FF_arr, P_arr


# ============================================================
# (e) 前馈-反馈联合控制仿真
# ============================================================
def simulate_ff_fb(ff_type='none'):
    """
    ff_type: 'none' | 'steady' | 'dynamic'
    反馈: IMC PI (K_c=0.5, τ_I=1)
    """
    Y_sp = np.zeros(N)       # 设定值恒定=0 (只看扰动抑制)
    D = np.full(N, D_step)

    # PI 控制器
    pi = PIController(Kc_imc, tauI_imc, Ts)

    # 过程 G_p
    y_p_inner = 0.0

    # 扰动通道 G_d
    y_d1_inner = 0.0
    y_d2_inner = 0.0

    # 动态 FF
    y_ff_inner = 0.0

    # 输出
    y = np.zeros(N)
    P_FF_arr = np.zeros(N)
    P_FB_arr = np.zeros(N)
    P_arr = np.zeros(N)

    for k in range(N - 1):
        # 前馈
        if ff_type == 'steady':
            P_FF = Kf_ss * D[k]
        elif ff_type == 'dynamic':
            y_ff_inner = first_order_step(y_ff_inner, D[k], a_ff, b_ff)
            P_FF = -2.0 * y_ff_inner
        else:
            P_FF = 0.0

        # 反馈 (IMC PI)
        e = Y_sp[k] - y[k]    # Y_sp - Y_m, G_m=1
        P_FB = pi.step(e)

        P = P_FF + P_FB

        # 过程
        y_p_inner = first_order_step(y_p_inner, P, a_p, b_p)

        # 扰动
        y_d1_inner = first_order_step(y_d1_inner, D[k], a_d1, b_d1)
        y_d2_inner = first_order_step(y_d2_inner, y_d1_inner, a_d2, b_d2)

        y[k + 1] = y_p_inner + 2.0 * y_d2_inner

        P_FF_arr[k] = P_FF
        P_FB_arr[k] = P_FB
        P_arr[k] = P

    return t, Y_sp, D, y, P_FF_arr, P_FB_arr, P_arr


# ============================================================
# 运行仿真
# ============================================================
print("\n" + "=" * 70)
print("Running Simulations...")
print("=" * 70)

# --- (d) 纯前馈 ---
print("\n>>> (d) Pure FF: steady-state, dynamic, no compensation")
t_d, D_d, y_none, P_FF_none, P_none = simulate_pure_ff('none')
t_d, D_d, y_ss, P_FF_ss, P_ss = simulate_pure_ff('steady')
t_d, D_d, y_dyn, P_FF_dyn, P_dyn = simulate_pure_ff('dynamic')

# --- (e) 前馈+反馈 ---
print(">>> (e) FF + FB: IMC PI only, steady FF + PI, dynamic FF + PI")
t_e, Y_sp_e, D_e, y_pi_only, P_FF_0, P_FB_pi, P_pi = simulate_ff_fb('none')
t_e, Y_sp_e, D_e, y_ss_pi, P_FF_ss_pi, P_FB_ss_pi, P_ss_pi = simulate_ff_fb('steady')
t_e, Y_sp_e, D_e, y_dyn_pi, P_FF_dyn_pi, P_FB_dyn_pi, P_dyn_pi = simulate_ff_fb('dynamic')


# ============================================================
# 绘图设置
# ============================================================
plt.rcParams.update({
    'font.size': 11,
    'axes.titlesize': 13,
    'axes.labelsize': 12,
    'legend.fontsize': 10,
    'figure.dpi': 150,
    'savefig.dpi': 150,
    'savefig.bbox': 'tight',
})

# ============================================================
# 图 1: 纯前馈控制 — 扰动抑制对比
# ============================================================
fig, axes = plt.subplots(2, 1, figsize=(12, 8), sharex=True)

# 输出响应
ax = axes[0]
ax.plot(t_d, D_d, 'k--', linewidth=1.3, alpha=0.5, label=r'Disturbance $D(t)=1(t)$')
ax.plot(t_d, y_none, 'r-', linewidth=2, label='No Compensation (open-loop)')
ax.plot(t_d, y_ss, 'orange', linewidth=2, label=f'Steady-State FF ($K_f={Kf_ss}$)')
ax.plot(t_d, y_dyn, 'g-', linewidth=2, label='Dynamic FF $G_f(s)=-2/(5s+1)$')
ax.axhline(y=0, color='gray', linestyle=':', alpha=0.4)
ax.set_ylabel(r'Output $Y(t)$')
ax.set_title('(d) Pure Feedforward — Disturbance Rejection Comparison')
ax.legend(loc='upper right')
ax.set_ylim([-0.5, 2.2])
ax.grid(True, alpha=0.3)

# 控制器输出
ax = axes[1]
ax.plot(t_d, P_ss, 'orange', linewidth=1.5, label=f'$P(t)$ (Steady-State FF)')
ax.plot(t_d, P_dyn, 'g-', linewidth=1.5, label='$P(t)$ (Dynamic FF)')
ax.set_xlabel('Time (s)')
ax.set_ylabel(r'Controller Output $P(t)$')
ax.set_title('Feedforward Controller Output')
ax.legend()
ax.set_xlim([0, t_end])
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('feedforward_pure_ff.png')
print("Saved: feedforward_pure_ff.png")
plt.close(fig)


# ============================================================
# 图 2: 前馈+反馈联合控制 — 扰动抑制对比
# ============================================================
fig, axes = plt.subplots(2, 1, figsize=(12, 8), sharex=True)

ax = axes[0]
ax.plot(t_e, D_e, 'k--', linewidth=1.3, alpha=0.5, label=r'Disturbance $D(t)=1(t)$')
ax.plot(t_e, y_pi_only, 'r-', linewidth=2, label='IMC PI only (no FF)')
ax.plot(t_e, y_ss_pi, 'orange', linewidth=2, label=f'Steady-State FF ($K_f={Kf_ss}$) + IMC PI')
ax.plot(t_e, y_dyn_pi, 'g-', linewidth=2, label='Dynamic FF $G_f(s)=-2/(5s+1)$ + IMC PI')
ax.axhline(y=0, color='gray', linestyle=':', alpha=0.4)
ax.set_ylabel(r'Output $Y(t)$')
ax.set_title('(e) Feedforward + Feedback — Disturbance Rejection\n'
             f'IMC PI: $K_c={Kc_imc}$, $\\tau_I={tauI_imc}$')
ax.legend(loc='upper right')
ax.grid(True, alpha=0.3)

# 控制器输出分解 (以动态FF+PI为例)
ax = axes[1]
ax.plot(t_e, P_FB_pi, 'r-', linewidth=1.3, alpha=0.7, label='$P_{FB}$ (PI only)')
ax.plot(t_e, P_FB_dyn_pi, 'b-', linewidth=1.3, alpha=0.7, label='$P_{FB}$ (Dynamic FF + PI)')
ax.plot(t_e, P_FF_dyn_pi, 'g--', linewidth=1.5, label='$P_{FF}$ (Dynamic FF)')
ax.plot(t_e, P_dyn_pi, 'k-', linewidth=1.8, label='$P = P_{FF} + P_{FB}$ (Total)')
ax.set_xlabel('Time (s)')
ax.set_ylabel(r'Controller Signals')
ax.set_title('Controller Signal Decomposition — Dynamic FF + IMC PI')
ax.legend(loc='lower right', ncol=2)
ax.set_xlim([0, t_end])
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('feedforward_ff_fb.png')
print("Saved: feedforward_ff_fb.png")
plt.close(fig)


# ============================================================
# 图 3: 综合对比大图 (6 条曲线合并)
# ============================================================
fig, axes = plt.subplots(2, 1, figsize=(14, 10), sharex=True)

# 上：全部 6 种方案的 Y(t) 对比
ax = axes[0]
ax.plot(t_d, D_d, 'k--', linewidth=1.3, alpha=0.5, label=r'$D(t)=1(t)$')
ax.plot(t_d, y_none, color='gray', linewidth=1.5, linestyle=':', label='No control')
ax.plot(t_d, y_ss, color='orange', linewidth=1.8, linestyle='--', label='Steady FF only')
ax.plot(t_d, y_dyn, color='gold', linewidth=1.8, linestyle='--', label='Dynamic FF only')
ax.plot(t_e, y_pi_only, color='red', linewidth=2, label='IMC PI only')
ax.plot(t_e, y_ss_pi, color='dodgerblue', linewidth=2, label='Steady FF + IMC PI')
ax.plot(t_e, y_dyn_pi, color='green', linewidth=2.5, label='Dynamic FF + IMC PI')
ax.axhline(y=0, color='k', linestyle='-', alpha=0.2)
ax.set_ylabel(r'Output $Y(t)$')
ax.set_title('Feedforward-Feedback Control — Comprehensive Comparison\n'
             f'IMC PI: $K_c={Kc_imc}$, $\\tau_I={tauI_imc}$ | '
             f'Steady FF: $K_f={Kf_ss}$ | Dynamic FF: $G_f(s)=-2/(5s+1)$')
ax.legend(loc='upper right', ncol=2, fontsize=8.5)
ax.grid(True, alpha=0.3)

# 下：各方案的峰值偏差对比（柱状图）
ax = axes[1]
schemes = ['No\ncontrol', 'Steady\nFF', 'Dynamic\nFF',
           'IMC\nPI only', 'Steady FF\n+ IMC PI', 'Dynamic FF\n+ IMC PI']
peak_dev = [np.max(np.abs(y_none)),
            np.max(np.abs(y_ss)),
            np.max(np.abs(y_dyn)),
            np.max(np.abs(y_pi_only)),
            np.max(np.abs(y_ss_pi)),
            np.max(np.abs(y_dyn_pi))]
colors_bar = ['gray', 'orange', 'gold', 'red', 'dodgerblue', 'green']
bars = ax.bar(schemes, peak_dev, color=colors_bar, edgecolor='black', alpha=0.85)
for bar, val in zip(bars, peak_dev):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
            f'{val:.3f}', ha='center', va='bottom', fontsize=9)
ax.set_ylabel('Peak Deviation |Y_max|')
ax.set_title('Maximum Deviation Comparison')
ax.grid(True, alpha=0.3, axis='y')

plt.tight_layout()
plt.savefig('feedforward_comprehensive.png')
print("Saved: feedforward_comprehensive.png")
plt.close(fig)


# ============================================================
# 图 4: 稳态/动态 FF + PI 的控制信号分解对比
# ============================================================
fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))

# 稳态 FF + PI
ax = axes[0]
ax.plot(t_e, P_FB_ss_pi, 'r-', linewidth=1.3, label='$P_{FB}$ (PI)')
ax.plot(t_e, P_FF_ss_pi, 'orange', linewidth=1.5, label=f'$P_{{FF}}$ (Steady, $K_f={Kf_ss}$)')
ax.plot(t_e, P_ss_pi, 'k-', linewidth=1.8, label='$P = P_{FF} + P_{FB}$')
ax.set_xlabel('Time (s)')
ax.set_ylabel('Signal')
ax.set_title('Steady-State FF + IMC PI')
ax.legend()
ax.set_xlim([0, t_end])
ax.grid(True, alpha=0.3)

# 动态 FF + PI
ax = axes[1]
ax.plot(t_e, P_FB_dyn_pi, 'r-', linewidth=1.3, label='$P_{FB}$ (PI)')
ax.plot(t_e, P_FF_dyn_pi, 'g-', linewidth=1.5, label='$P_{FF}$ (Dynamic)')
ax.plot(t_e, P_dyn_pi, 'k-', linewidth=1.8, label='$P = P_{FF} + P_{FB}$')
ax.set_xlabel('Time (s)')
ax.set_ylabel('Signal')
ax.set_title('Dynamic FF $G_f(s)=-2/(5s+1)$ + IMC PI')
ax.legend()
ax.set_xlim([0, t_end])
ax.grid(True, alpha=0.3)

fig.suptitle('Controller Signal Decomposition — FF + FB Combined',
             fontsize=14, y=1.02)
plt.tight_layout()
plt.savefig('feedforward_signal_decomposition.png')
print("Saved: feedforward_signal_decomposition.png")
plt.close(fig)


# ============================================================
# 总结
# ============================================================
print("\n" + "=" * 70)
print("Simulation Complete!")
print("=" * 70)

# 性能指标
print(f"\n{'Scheme':<30} {'Peak |Y|':>12} {'SS |Y|':>12}")
print("-" * 56)
print(f"{'No control (open-loop)':<30} {np.max(np.abs(y_none)):>12.4f} {abs(y_none[-1]):>12.4f}")
print(f"{'Steady FF only':<30} {np.max(np.abs(y_ss)):>12.4f} {abs(y_ss[-1]):>12.4f}")
print(f"{'Dynamic FF only':<30} {np.max(np.abs(y_dyn)):>12.4f} {abs(y_dyn[-1]):>12.4f}")
print(f"{'IMC PI only':<30} {np.max(np.abs(y_pi_only)):>12.4f} {abs(y_pi_only[-1]):>12.4f}")
print(f"{'Steady FF + IMC PI':<30} {np.max(np.abs(y_ss_pi)):>12.4f} {abs(y_ss_pi[-1]):>12.4f}")
print(f"{'Dynamic FF + IMC PI':<30} {np.max(np.abs(y_dyn_pi)):>12.4f} {abs(y_dyn_pi[-1]):>12.4f}")

print(f"\nDesign Summary:")
print(f"  (a) Steady-State FF:  K_f = {Kf_ss}")
print(f"  (b) Dynamic FF:        G_f(s) = -2/(5s+1)  [physically realizable]")
print(f"  (c) IMC PI (τ_c=2):   K_c = {Kc_imc}, τ_I = {tauI_imc}")
print(f"\nKey Insight:")
print(f"  Dynamic FF achieves perfect steady-state compensation (SS error = 0)")
print(f"  because G_f(s) = -2/(5s+1) exactly cancels the Gd/Gp ratio.")
print(f"  Adding IMC PI feedback further improves transient response.")
