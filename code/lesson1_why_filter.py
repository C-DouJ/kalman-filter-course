import numpy as np
import matplotlib.pyplot as plt
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'PingFang SC', 'Noto Sans CJK SC']
plt.rcParams['axes.unicode_minus'] = False

# 1. 设定：小车沿直线开，名义速度 u = 1 m/s，每 Δt = 1 s 读一次传感器
N, dt, u = 200, 1.0, 1.0
Q = 0.5**2        # 运动扰动的方差（打滑、路面不平），σ_w = 0.5 m
R = 2.0**2        # 传感器噪声的方差，σ_R = 2 m
rng = np.random.default_rng()

# 2. 生成真实位置和传感器读数（真实位置只有我们知道，滤波器看不到）
x_true = np.zeros(N)
for k in range(1, N):
    x_true[k] = x_true[k-1] + u*dt + rng.normal(0, np.sqrt(Q))
z = x_true + rng.normal(0, np.sqrt(R), N)

# 3. 两种“偷懒”的办法
x_dr = np.arange(N) * u * dt      # 只信预测：从起点按名义速度一路推算
x_meas = z                        # 只信传感器：读数是多少就是多少

# 4. 卡尔曼滤波：每一步按“谁更靠谱”把预测和读数加权（这五行怎么来的，第 2、3 课讲）
x_kf = np.zeros(N)
P = 0.0                           # 起点已知，所以初始误差方差为 0
for k in range(1, N):
    x_pred = x_kf[k-1] + u*dt     # 预测位置
    P_pred = P + Q                # 预测的不确定性
    K = P_pred / (P_pred + R)     # 给读数的权重
    x_kf[k] = x_pred + K * (z[k] - x_pred)   # 用读数修正
    P = (1 - K) * P_pred          # 修正后的不确定性

for name, est in [('只信预测', x_dr), ('只信传感器', x_meas), ('卡尔曼滤波', x_kf)]:
    rmse = np.sqrt(np.mean((est - x_true)**2))
    print(f'{name}: RMSE = {rmse:.2f} m')
print(f'理论交叉点 k_c = R/Q = {R/Q:.0f} 步')

k = np.arange(N)
plt.plot(k, x_dr - x_true, label='只信预测的误差')
plt.plot(k, z - x_true, '.', label='只信传感器的误差')
plt.plot(k, x_kf - x_true, lw=2, label='卡尔曼滤波的误差')
plt.plot(k, np.sqrt(k*Q), 'k:', k, -np.sqrt(k*Q), 'k:')
plt.axhline(0, color='gray', lw=0.5)
plt.xlabel('k（步）'); plt.ylabel('误差 (m)'); plt.legend(); plt.show()
