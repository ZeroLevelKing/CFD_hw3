import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
mpl.rcParams['font.sans-serif'] = ['SimHei']
mpl.rcParams['axes.unicode_minus'] = False

# ================== 通用函数 ==================
def initial_condition(x, case='smooth'):
    """生成初始条件"""
    if case == 'smooth':
        return np.sin(2 * np.pi * x / L)  # 光滑初值用于精度分析
    elif case == 'square':
        u = np.zeros_like(x)
        u[(x > 2) & (x < 4)] = 1.0        # 方波初值用于耗散分析
        return u
    else:
        raise ValueError("无效的初始条件类型")

def periodic_roll(u, shift):
    """处理周期性边界条件的滚动操作"""
    return np.roll(u, shift, axis=0)

# ================== 数值格式实现 ==================
def lax_wendroff(u, cfl):
    """Lax-Wendroff格式单步更新"""
    u_prev = periodic_roll(u, 1)
    u_next = periodic_roll(u, -1)
    return u - 0.5*cfl*(u_next - u_prev) + 0.5*cfl**2*(u_next - 2*u + u_prev)

def warming_beam(u, cfl):
    """Warming-Beam格式单步更新"""
    u_prev1 = periodic_roll(u, 1)
    u_prev2 = periodic_roll(u, 2)
    return u - cfl*(u - u_prev1) + 0.5*cfl*(cfl-1)*(u - 2*u_prev1 + u_prev2)

def leap_frog(u_prev, u_current, cfl):
    """Leap-frog格式单步更新"""
    u_next_val = periodic_roll(u_current, -1)
    u_prev_val = periodic_roll(u_current, 1)
    return u_prev - cfl*(u_next_val - u_prev_val)

# ================== 模拟运行函数 ==================
def simulate(scheme, u0, cfl, nt, **kwargs):
    """通用模拟函数"""
    u = u0.copy()
    history = [u.copy()]
    
    # 处理特殊格式的初始化
    if scheme.__name__ == 'leap_frog':
        u_prev = u0.copy()
        u_current = lax_wendroff(u_prev, cfl)  # 用Lax-Wendroff启动第一步
        history.append(u_current.copy())
        for _ in range(1, nt):
            u_next = scheme(u_prev, u_current, cfl)
            u_prev, u_current = u_current, u_next
            history.append(u_next.copy())
        return np.array(history)
    else:
        for _ in range(nt):
            u = scheme(u, cfl)
            history.append(u.copy())
        return np.array(history)

# ================== 参数设置 ==================
L = 10.0          # 空间域长度
T = 5.0           # 总模拟时间
cases = [
    {'name': 'Lax-Wendroff稳定', 'cfl': 0.8, 'scheme': lax_wendroff},
    {'name': 'Lax-Wendroff失稳', 'cfl': 1.1, 'scheme': lax_wendroff},
    {'name': 'Warming-Beam稳定', 'cfl': 1.5, 'scheme': warming_beam},
    {'name': 'Warming-Beam失稳', 'cfl': 2.2, 'scheme': warming_beam},
    {'name': 'Leap-frog稳定', 'cfl': 0.95, 'scheme': leap_frog},
    {'name': 'Leap-frog失稳', 'cfl': 1.05, 'scheme': leap_frog},
]

# ================== 稳定性验证 ==================
plt.figure(figsize=(12, 8), dpi=100)
x = np.linspace(0, L, 100, endpoint=False)
u0_square = initial_condition(x, case='square')

for i, case in enumerate(cases):
    nx = 100
    dx = L / nx
    dt = case['cfl'] * dx
    nt = int(T / dt)
    
    history = simulate(case['scheme'], u0_square, case['cfl'], nt)
    
    plt.subplot(2, 3, i+1)
    plt.plot(x, history[0], 'k-', lw=1, label='初始')
    plt.plot(x, history[-1], 'r--', lw=1, label='终态')
    plt.title(f"{case['name']} (σ={case['cfl']})")
    plt.ylim(-0.5, 1.5)
    plt.legend(loc='upper right')

plt.tight_layout()

# ================== 精度分析 ==================
plt.figure(figsize=(8, 6), dpi=100)
cfl_lw = 0.8
nx_list = [50, 100, 200, 400]
colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']

for i, nx in enumerate(nx_list):
    x = np.linspace(0, L, nx, endpoint=False)
    dx = L / nx
    dt = cfl_lw * dx
    nt = int(T / dt)
    u0 = initial_condition(x, case='smooth')
    history = simulate(lax_wendroff, u0, cfl_lw, nt)
    u_exact = initial_condition((x - T) % L, case='smooth')
    error = np.sqrt(np.mean((history[-1] - u_exact)**2))
    plt.loglog(1/nx, error, 'o', color=colors[i], markersize=8, 
               label=f'nx={nx}')

# 绘制参考线
ref_x = [1/50, 1/100]
ref_y = [0.1*(x/ref_x[0])**2 for x in ref_x]
plt.loglog(ref_x, ref_y, 'k--', lw=1, label='二阶收敛')
plt.xlabel('网格间距Δx')
plt.ylabel('L2误差')
plt.title('Lax-Wendroff格式收敛性分析')
plt.legend()

# ================== 耗散与相位分析 ==================
plt.figure(figsize=(12, 6), dpi=100)
x = np.linspace(0, L, 200, endpoint=False)
u0_square = initial_condition(x, case='square')

# Lax-Wendroff
history_lw = simulate(lax_wendroff, u0_square, 0.8, int(5/(0.8*(L/200))))
# Warming-Beam
history_wb = simulate(warming_beam, u0_square, 1.5, int(5/(1.5*(L/200))))
# Leap-frog
history_lf = simulate(leap_frog, u0_square, 0.95, int(5/(0.95*(L/200))))

plt.plot(x, history_lw[0], 'k-', lw=2, label='初始')
plt.plot(x, history_lw[-1], 'r--', lw=1.5, label='Lax-Wendroff')
plt.plot(x, history_wb[-1], 'g-.', lw=1.5, label='Warming-Beam')
plt.plot(x, history_lf[-1], 'b:', lw=1.5, label='Leap-frog')
plt.ylim(-0.5, 1.5)
plt.legend()
plt.title('耗散与相位特性比较 (t=5)')

plt.tight_layout()
plt.show()